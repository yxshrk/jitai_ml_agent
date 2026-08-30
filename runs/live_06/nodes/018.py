"""node_000 -- the official FM baseline, ported to the harness contract.

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
Loss    Same-user BPR with one uniformly sampled negative per eligible positive each epoch.
Ensemble averages tie-free normalized within-user ranks from five independently early-stopped seeds.
"""
import argparse, csv, json, os, time
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def read_rows(path, cols):
    """Rows of `path` restricted to `cols`, as lists of strings, in file order."""
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r); idx = [head.index(c) for c in cols]
        return [[rec[i] for i in idx] for rec in r]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def normalized_ranks(users, scores, tiebreak=None):
    users = np.asarray(users); scores = np.asarray(scores)
    if len(scores) == 0:
        return np.empty(0, dtype=np.float64)
    if tiebreak is None:
        tiebreak = np.arange(len(scores))
    order = np.lexsort((np.arange(len(scores)), np.asarray(tiebreak), scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    counts = np.diff(np.r_[starts, len(scores)])
    pos = np.arange(len(scores)) - np.repeat(starts, counts)
    den = np.maximum(np.repeat(counts, counts) - 1, 1)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = pos / den
    return ranks

def average_ranks(users, ranks):
    mean_rank = np.mean(np.stack(ranks), axis=0)
    return normalized_ranks(users, mean_rank, ranks[0])

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with same-user BPR + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # one k-vector per feature value
        self.T = rng.normal(0, 0.05, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mT = np.zeros_like(self.T); self.vT = np.zeros_like(self.T)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                              # (B, F, k)
        S = E.sum(1)                                               # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))    # sum of all pairwise dot products
        C = self.T[X[:, (0, 1, 3)]]
        third = (C[:, 0] * C[:, 1] * C[:, 2]).sum(1)
        return self.b + self.W[X].sum(1) + inter + third, E, S, C

    def step(self, Xp, Xn):
        B = len(Xp)
        zp, Ep, Sp, Cp = self.logits(Xp)
        zn, En, Sn, Cn = self.logits(Xn)
        d = zp - zn
        g = (-sigmoid(-d) / B).astype(np.float32)                  # d(BPR loss)/d(score_pos)
        gV = np.zeros_like(self.V); gT = np.zeros_like(self.T); gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, g[:, None])
        np.add.at(gW, Xn, -g[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
        np.add.at(gT, Xp[:, 0], g[:, None] * Cp[:, 1] * Cp[:, 2])
        np.add.at(gT, Xp[:, 1], g[:, None] * Cp[:, 0] * Cp[:, 2])
        np.add.at(gT, Xp[:, 3], g[:, None] * Cp[:, 0] * Cp[:, 1])
        np.add.at(gT, Xn[:, 0], -g[:, None] * Cn[:, 1] * Cn[:, 2])
        np.add.at(gT, Xn[:, 1], -g[:, None] * Cn[:, 0] * Cn[:, 2])
        np.add.at(gT, Xn[:, 3], -g[:, None] * Cn[:, 0] * Cn[:, 1])
        gV += self.l2 * self.V; gT += self.l2 * self.T; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.T, gT, self.mT, self.vT),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(sigmoid(d) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def write_predictions(path, rows, scores):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for x, s in zip(rows, scores):
            w.writerow([x[0], x[1], x[2], f'{float(s):.9g}'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--score-extra', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    a = ap.parse_args()
    smoke = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    epochs = min(a.epochs, smoke) if smoke > 0 else a.epochs
    os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()

    # ---- load ----
    vid2author = dict(read_rows(f'{a.data_dir}/video_features_basic.csv', ['video_id', 'author_id']))
    tr = read_rows(f'{a.data_dir}/train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view'])
    va = read_rows(f'{a.data_dir}/valid.csv', ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view'])

    # ---- encode: 5 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur):
        return [user, video, vid2author.get(video, 'UNK'), tab, str(int(np.searchsorted(edges, float(dur))))]
    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32); dim = int(sum(dims))
    def encode(rows, ui, vi, ti, di):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr = encode(tr, 0, 1, 2, 3); ytr = np.array([1.0 if x[4] != '0' else 0.0 for x in tr], dtype=np.float32)
    Xva = encode(va, 1, 2, 3, 4); yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
    neg_idx = np.flatnonzero(ytr == 0)
    neg_count = np.bincount(Xtr[neg_idx, 0], minlength=dims[0])
    neg_start = np.cumsum(np.r_[0, neg_count[:-1]])
    neg_sorted = neg_idx[np.argsort(Xtr[neg_idx, 0], kind='stable')]
    pair_pos = np.flatnonzero((ytr > 0) & (neg_count[Xtr[:, 0]] > 0))
    pair_users = Xtr[pair_pos, 0]
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    # ---- train with early stopping on valid primary ----
    models = [FM(dim, k=a.k, lr=a.lr, seed=a.seed + s) for s in range(5)]
    rngs = [np.random.default_rng(a.seed + s) for s in range(5)]
    best = [-1.0] * 5; best_state = [None] * 5; best_ranks = [None] * 5
    bad = [0] * 5; active = [True] * 5; history = []
    for ep in range(1, epochs + 1):
        if not any(active):
            break
        epoch_losses = []
        for s, m in enumerate(models):
            if not active[s]:
                continue
            pair_neg = neg_sorted[neg_start[pair_users] + rngs[s].integers(0, neg_count[pair_users])]
            idx = rngs[s].permutation(len(pair_pos))
            losses = [m.step(Xtr[pair_pos[idx[i:i + a.batch]]], Xtr[pair_neg[idx[i:i + a.batch]]])
                      for i in range(0, len(idx), a.batch)]
            epoch_losses.extend(losses)
            pred = m.predict(Xva); rm = evaluate(uva, yva, pred)
            if rm['primary'] > best[s] + 1e-5:
                best[s], bad[s] = rm['primary'], 0
                best_state[s] = (m.V.copy(), m.T.copy(), m.W.copy(), np.float32(m.b))
                best_ranks[s] = normalized_ranks(uva, pred)
            else:
                bad[s] += 1
                if bad[s] >= a.patience:
                    active[s] = False
        r = evaluate(uva, yva, average_ranks(uva, best_ranks))
        history.append({'epoch': ep, 'train_loss': float(np.mean(epoch_losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {np.mean(epoch_losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
    for m, state in zip(models, best_state):
        m.V, m.T, m.W, m.b = state

    # ---- outputs ----
    sva = average_ranks(uva, best_ranks); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        Xex = encode(ex, 1, 2, 3, 4); uex = [x[1] for x in ex]
        rex = [normalized_ranks(uex, m.predict(Xex)) for m in models]
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, average_ranks(uex, rex))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
