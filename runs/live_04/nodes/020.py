"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
        The standard-FM ensemble branch adds a rank-8 factorized DCN cross head.
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
    if tiebreak is None:
        order = np.lexsort((np.arange(len(scores)), scores, users))
    else:
        order = np.lexsort((np.arange(len(scores)), np.asarray(tiebreak), scores, users))
    su = users[order]
    starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(order)])
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = (np.arange(len(order)) - np.repeat(starts, counts)) / np.repeat(np.maximum(counts - 1, 1), counts)
    return ranks

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with logloss + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, standard=False):
        rng = np.random.default_rng(seed)
        self.standard = standard
        self.V = rng.normal(0, 0.01, (dim, k) if standard else (dim, len(FIELDS), k)).astype(np.float32)   # one k-vector per partner field
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        if self.standard:
            width = len(FIELDS) * k
            self.U = rng.normal(0, 0.01, (width, 8)).astype(np.float32)
            self.R = rng.normal(0, 0.01, (8, width)).astype(np.float32)
            self.cb = np.zeros(width, dtype=np.float32)
            self.v = rng.normal(0, 0.01, width).astype(np.float32)
            self.mU = np.zeros_like(self.U); self.vU = np.zeros_like(self.U)
            self.mR = np.zeros_like(self.R); self.vR = np.zeros_like(self.R)
            self.mcb = np.zeros_like(self.cb); self.vcb = np.zeros_like(self.cb)
            self.mv = np.zeros_like(self.v); self.vv = np.zeros_like(self.v)
        self.t = 0

    def logits(self, X):
        if self.standard:
            E = self.V[X]
            inter = 0.5 * ((E.sum(1) ** 2 - (E * E).sum(1)).sum(1))
            x0 = E.reshape(len(X), -1)
            cross = (x0 @ self.U) @ self.R + self.cb
            inter += (x0 * (cross + 1.0)) @ self.v
            return self.b + self.W[X].sum(1) + inter, x0, cross
        else:
            inter = np.zeros(len(X), dtype=np.float32)
            for i in range(len(FIELDS)):
                for j in range(i + 1, len(FIELDS)):
                    inter += (self.V[X[:, i], j] * self.V[X[:, j], i]).sum(1)
        return self.b + self.W[X].sum(1) + inter, None, None

    def step(self, Xp, Xn):
        B = len(Xp)
        zp, x0p, cp = self.logits(Xp); zn, x0n, cn = self.logits(Xn)
        d = zp - zn
        g = ((sigmoid(d) - 1.0) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        if self.standard:
            gU = np.zeros_like(self.U); gR = np.zeros_like(self.R)
            gcb = np.zeros_like(self.cb); gv = np.zeros_like(self.v)
        for X, h, x0, cross in ((Xp, g, x0p, cp), (Xn, -g, x0n, cn)):
            np.add.at(gW, X, h[:, None])
            if self.standard:
                E = self.V[X]; S = E.sum(1)
                np.add.at(gV, X, h[:, None, None] * (S[:, None, :] - E))
                dx1 = h[:, None] * self.v
                dcross = dx1 * x0
                hidden = x0 @ self.U
                gv += (x0 * (cross + 1.0)).T @ h
                gR += hidden.T @ dcross
                gU += x0.T @ (dcross @ self.R.T)
                gcb += dcross.sum(0)
                dx0 = dx1 * (cross + 1.0) + (dcross @ self.R.T) @ self.U.T
                np.add.at(gV, X, dx0.reshape(len(X), len(FIELDS), -1))
            else:
                for i in range(len(FIELDS)):
                    for j in range(i + 1, len(FIELDS)):
                        Ei = self.V[X[:, i], j]
                        Ej = self.V[X[:, j], i]
                        np.add.at(gV, (X[:, i], j), h[:, None] * Ej)
                        np.add.at(gV, (X[:, j], i), h[:, None] * Ei)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        if self.standard:
            gU += 1e-4 * self.U; gR += 1e-4 * self.R
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [(self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)]
        if self.standard:
            params += [(self.U, gU, self.mU, self.vU), (self.R, gR, self.mR, self.vR),
                       (self.cb, gcb, self.mcb, self.vcb), (self.v, gv, self.mv, self.vv)]
        for P, G, M, Vv in params:
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
    neg_rows = np.flatnonzero(ytr == 0)
    neg_order = neg_rows[np.argsort(Xtr[neg_rows, 0], kind='stable')]
    neg_users, neg_starts, neg_counts = np.unique(Xtr[neg_order, 0], return_index=True, return_counts=True)
    neg_start = np.zeros(dims[0], dtype=np.int64); neg_count = np.zeros(dims[0], dtype=np.int64)
    neg_start[neg_users] = neg_starts; neg_count[neg_users] = neg_counts
    pair_pos = np.flatnonzero(ytr != 0)
    pair_pos = pair_pos[neg_count[Xtr[pair_pos, 0]] > 0]
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    # ---- train with early stopping on valid primary ----
    pred_cache = [[], []]; loss_cache = [[], []]; models = [[], []]
    for branch in range(2):
        for member in range(5):
            member_seed = a.seed + branch * 5 + member
            m = FM(dim, k=a.k, lr=a.lr, seed=member_seed, standard=bool(branch))
            rng = np.random.default_rng(member_seed)
            best, best_state, best_pred, bad = -1.0, None, None, 0
            pred_epochs, loss_epochs = [], []
            for ep in range(1, epochs + 1):
                pidx = pair_pos[rng.permutation(len(pair_pos))]
                pu = Xtr[pidx, 0]
                nidx = neg_order[neg_start[pu] + (rng.random(len(pidx)) * neg_count[pu]).astype(np.int64)]
                losses = [m.step(Xtr[pidx[i:i + a.batch]], Xtr[nidx[i:i + a.batch]]) for i in range(0, len(pidx), a.batch)]
                current_pred = m.predict(Xva)
                r = evaluate(uva, yva, current_pred)
                print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
                      f"primary {r['primary']:.4f}", flush=True)
                if r['primary'] > best + 1e-5:
                    if m.standard:
                        best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), m.U.copy(), m.R.copy(),
                                      m.cb.copy(), m.v.copy())
                    else:
                        best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
                    best, bad = r['primary'], 0
                    best_pred = current_pred.copy()
                else:
                    bad += 1
                pred_epochs.append(best_pred.copy())
                loss_epochs.append(float(np.mean(losses)))
                if bad >= a.patience:
                    break
            if m.standard:
                m.V, m.W, m.b, m.U, m.R, m.cb, m.v = best_state
                m.mU = m.vU = m.mR = m.vR = m.mcb = m.vcb = m.mv = m.vv = None
            else:
                m.V, m.W, m.b = best_state
            m.mV = m.vV = m.mW = m.vW = None
            pred_cache[branch].append(pred_epochs)
            loss_cache[branch].append(loss_epochs)
            models[branch].append(m)

    history = []
    synchronized_epochs = max(len(x) for branch in pred_cache for x in branch)
    for ep in range(synchronized_epochs):
        branch_scores = [
            np.mean([normalized_ranks(uva, pred_cache[branch][member][min(ep, len(pred_cache[branch][member]) - 1)])
                     for member in range(5)], axis=0)
            for branch in range(2)
        ]
        blend = 0.6 * branch_scores[0] + 0.4 * branch_scores[1]
        sva = normalized_ranks(uva, blend, branch_scores[0])
        r = evaluate(uva, yva, sva)
        train_loss = float(np.mean([
            loss_cache[branch][member][min(ep, len(loss_cache[branch][member]) - 1)]
            for branch in range(2) for member in range(5)
        ]))
        history.append({'epoch': ep + 1, 'train_loss': train_loss, 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})

    # ---- outputs ----
    r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        Xex = encode(ex, 1, 2, 3, 4); uex = [x[1] for x in ex]
        branch_scores = [
            np.mean([normalized_ranks(uex, m.predict(Xex)) for m in models[branch]], axis=0)
            for branch in range(2)
        ]
        blend = 0.6 * branch_scores[0] + 0.4 * branch_scores[1]
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, normalized_ranks(uex, blend, branch_scores[0]))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
