"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
Loss    Same-user logistic BPR with one uniformly sampled negative per positive each epoch.
Time    Continuous global, tab-specific, and video-specific date slopes are added to the score.
"""
import argparse, csv, json, os, time
import datetime
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

class FM:
    """score = FM + date * (global + video + tab slope), trained with BPR + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # one k-vector per feature value
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.D = np.zeros(dim, dtype=np.float32)
        self.G = np.zeros((), dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mD = np.zeros_like(self.D); self.vD = np.zeros_like(self.D)
        self.mG = np.zeros_like(self.G); self.vG = np.zeros_like(self.G)
        self.t = 0

    def logits(self, X, date):
        E = self.V[X]                                              # (B, F, k)
        S = E.sum(1)                                               # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))    # sum of all pairwise dot products
        drift = date * (self.G + self.D[X[:, 1]] + self.D[X[:, 3]])
        return self.b + self.W[X].sum(1) + inter + drift, E, S

    def step(self, Xp, Xn, tp, tn):
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp, tp)
        zn, En, Sn = self.logits(Xn, tn)
        g = (-sigmoid(-(zp - zn)) / B).astype(np.float32)          # d(BPR loss)/d(positive logit)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        gD = np.zeros_like(self.D); gG = np.zeros_like(self.G)
        for X, E, S, date, h in ((Xp, Ep, Sp, tp, g), (Xn, En, Sn, tn, -g)):
            np.add.at(gW, X, h[:, None])
            np.add.at(gV, X, h[:, None, None] * (S[:, None, :] - E))
            np.add.at(gD, X[:, 1], h * date)
            np.add.at(gD, X[:, 3], h * date)
            gG += np.dot(h, date)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        gD += self.l2 * self.D; gG += self.l2 * self.G
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW),
                            (self.D, gD, self.mD, self.vD), (self.G, gG, self.mG, self.vG)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(sigmoid(zp - zn) + 1e-9)))

    def predict(self, X, date, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs], date[i:i + bs])[0] for i in range(0, len(X), bs)])

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
    tr = read_rows(f'{a.data_dir}/train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'date'])
    va = read_rows(f'{a.data_dir}/valid.csv', ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'date'])

    # ---- encode: 5 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur):
        return [user, video, vid2author.get(video, 'UNK'), tab, str(int(np.searchsorted(edges, float(dur))))]
    def ordinal_date(value):
        value = value.replace('-', '')
        return datetime.date(int(value[:4]), int(value[4:6]), int(value[6:8])).toordinal()
    train_days = np.array([ordinal_date(x[5]) for x in tr], dtype=np.float32)
    date_origin = train_days.max(); date_scale = max(float(date_origin - train_days.min()), 1.0)
    ttr = ((train_days - date_origin) / date_scale).astype(np.float32)
    tva = np.array([(ordinal_date(x[6]) - date_origin) / date_scale for x in va], dtype=np.float32)
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
    train_users = Xtr[:, 0]
    negative_pool = np.flatnonzero(ytr == 0)
    negative_pool = negative_pool[np.argsort(train_users[negative_pool], kind='stable')]
    negative_count = np.bincount(train_users[negative_pool], minlength=dim)
    negative_start = np.cumsum(np.r_[0, negative_count[:-1]])
    positive_rows = np.flatnonzero((ytr == 1) & (negative_count[train_users] > 0))
    pair_users = train_users[positive_rows]
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    # ---- train with early stopping on valid primary ----
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed); rng = np.random.default_rng(a.seed)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        sampled_negatives = negative_pool[negative_start[pair_users] +
            (rng.random(len(positive_rows)) * negative_count[pair_users]).astype(np.int64)]
        idx = rng.permutation(len(positive_rows))
        losses = [m.step(Xtr[positive_rows[idx[i:i + a.batch]]], Xtr[sampled_negatives[idx[i:i + a.batch]]],
                         ttr[positive_rows[idx[i:i + a.batch]]], ttr[sampled_negatives[idx[i:i + a.batch]]])
                  for i in range(0, len(idx), a.batch)]
        r = evaluate(uva, yva, m.predict(Xva, tva))
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if r['primary'] > best + 1e-5:
            best, bad, best_state = r['primary'], 0, (m.V.copy(), m.W.copy(), np.float32(m.b),
                                                      m.D.copy(), m.G.copy())
        else:
            bad += 1
            if bad >= a.patience:
                break
    m.V, m.W, m.b, m.D, m.G = best_state

    # ---- outputs ----
    sva = m.predict(Xva, tva); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'date'])
        tex = np.array([(ordinal_date(x[5]) - date_origin) / date_scale for x in ex], dtype=np.float32)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, m.predict(encode(ex, 1, 2, 3, 4), tex))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
