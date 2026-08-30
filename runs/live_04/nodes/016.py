"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
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

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with logloss + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, len(FIELDS), k)).astype(np.float32)   # one k-vector per partner field
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.Wt = np.zeros(dim, dtype=np.float32)
        self.bt = np.zeros(1, dtype=np.float32)
        self.watch_weight = 0.05
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mWt = np.zeros_like(self.Wt); self.vWt = np.zeros_like(self.Wt)
        self.mbt = np.zeros_like(self.bt); self.vbt = np.zeros_like(self.bt)
        self.t = 0

    def logits(self, X):
        inter = np.zeros(len(X), dtype=np.float32)
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                inter += (self.V[X[:, i], j] * self.V[X[:, j], i]).sum(1)
        return self.b + self.W[X].sum(1) + inter, None, None

    def watch_logits(self, X):
        inter = np.zeros(len(X), dtype=np.float32)
        for i in range(len(FIELDS)):
            for j in range(i + 1, len(FIELDS)):
                inter += (self.V[X[:, i], j] * self.V[X[:, j], i]).sum(1)
        return self.bt[0] + self.Wt[X].sum(1) + inter

    def step(self, Xp, Xn, tp, tn, cp, cn, wp, wn):
        B = len(Xp)
        zp = self.logits(Xp)[0]; zn = self.logits(Xn)[0]
        d = zp - zn
        g = ((sigmoid(d) - 1.0) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        gWt = np.zeros_like(self.Wt); gbt = np.zeros_like(self.bt)
        for X, h in ((Xp, g), (Xn, -g)):
            np.add.at(gW, X, h[:, None])
            for i in range(len(FIELDS)):
                for j in range(i + 1, len(FIELDS)):
                    Ei = self.V[X[:, i], j]
                    Ej = self.V[X[:, j], i]
                    np.add.at(gV, (X[:, i], j), h[:, None] * Ej)
                    np.add.at(gV, (X[:, j], i), h[:, None] * Ei)
        aux_loss = 0.0
        if np.any(wp) or np.any(wn):
            Xw = np.concatenate((Xp[wp], Xn[wn]))
            tw = np.concatenate((tp[wp], tn[wn]))
            cw = np.concatenate((cp[wp], cn[wn]))
            resid = (self.watch_logits(Xw) - tw).astype(np.float32)
            resid[cw & (resid > 0.0)] = 0.0
            h = (2.0 * self.watch_weight * resid / len(Xw)).astype(np.float32)
            np.add.at(gWt, Xw, h[:, None])
            gbt[0] = h.sum()
            for i in range(len(FIELDS)):
                for j in range(i + 1, len(FIELDS)):
                    Ei = self.V[Xw[:, i], j]
                    Ej = self.V[Xw[:, j], i]
                    np.add.at(gV, (Xw[:, i], j), h[:, None] * Ej)
                    np.add.at(gV, (Xw[:, j], i), h[:, None] * Ei)
            aux_loss = self.watch_weight * float(np.mean(resid * resid))
        gV += self.l2 * self.V; gW += self.l2 * self.W; gWt += self.l2 * self.Wt
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW),
                            (self.Wt, gWt, self.mWt, self.vWt), (self.bt, gbt, self.mbt, self.vbt)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(sigmoid(d) + 1e-9))) + aux_loss

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
    tr = read_rows(f'{a.data_dir}/train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'play_time_ms'])
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
    play_ms = np.array([float(x[5]) for x in tr], dtype=np.float32)
    dur_ms = np.array([float(x[3]) for x in tr], dtype=np.float32)
    watch_t = np.log1p(np.maximum(play_ms, 0.0) / 1000.0).astype(np.float32)
    watch_valid = dur_ms > 0.0; watch_censored = watch_valid & (play_ms >= dur_ms)
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
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed); rng = np.random.default_rng(a.seed)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        pidx = pair_pos[rng.permutation(len(pair_pos))]
        pu = Xtr[pidx, 0]
        nidx = neg_order[neg_start[pu] + (rng.random(len(pidx)) * neg_count[pu]).astype(np.int64)]
        losses = []
        for i in range(0, len(pidx), a.batch):
            pi = pidx[i:i + a.batch]; ni = nidx[i:i + a.batch]
            losses.append(m.step(Xtr[pi], Xtr[ni], watch_t[pi], watch_t[ni],
                                 watch_censored[pi], watch_censored[ni], watch_valid[pi], watch_valid[ni]))
        r = evaluate(uva, yva, m.predict(Xva))
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if r['primary'] > best + 1e-5:
            best, bad, best_state = r['primary'], 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                break
    m.V, m.W, m.b = best_state

    # ---- outputs ----
    sva = m.predict(Xva); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, m.predict(encode(ex, 1, 2, 3, 4)))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
