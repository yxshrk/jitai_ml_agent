"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
Loss    Same-user logistic BPR with one uniformly sampled negative per positive impression.
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
    users, scores = np.asarray(users), np.asarray(scores)
    tie = np.zeros(len(scores)) if tiebreak is None else np.asarray(tiebreak)
    order = np.lexsort((-np.arange(len(scores)), tie, scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    out = np.empty(len(scores), dtype=np.float64)
    for start, end in zip(starts, np.r_[starts[1:], len(scores)]):
        out[order[start:end]] = np.arange(end - start) / max(end - start - 1, 1)
    return out

def rank_blend(users, bpr_scores, point_scores):
    br = normalized_ranks(users, bpr_scores)
    pr = normalized_ranks(users, point_scores)
    return normalized_ranks(users, 0.85 * br + 0.15 * pr, br)

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with same-user BPR + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # one k-vector per feature value
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.float32(0.0); self.vb = np.float32(0.0)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                              # (B, F, k)
        S = E.sum(1)                                               # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))    # sum of all pairwise dot products
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, Xp, Xn):
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = zp - zn
        g = (-sigmoid(-d) / B).astype(np.float32)                  # d(-log sigmoid(d))/d(d)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, g[:, None])
        np.add.at(gW, Xn, -g[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(np.log(sigmoid(d) + 1e-9)))

    def step_pointwise(self, X, y):
        z, E, S = self.logits(X)
        p = sigmoid(z)
        g = ((p - y) / len(X)).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        gb = np.float32(g.sum())
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.mb = b1 * self.mb + (1 - b1) * gb
        self.vb = b2 * self.vb + (1 - b2) * gb * gb
        self.b -= self.lr * (self.mb / (1 - b1 ** self.t)) / (np.sqrt(self.vb / (1 - b2 ** self.t)) + eps)
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

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
    vinfo = read_rows(f'{a.data_dir}/video_features_basic.csv',
                      ['video_id', 'author_id', 'tag', 'music_id', 'video_type'])
    vid2author = {x[0]: x[1] for x in vinfo}
    def clean(v):
        v = v.strip().strip("[]'\" ")
        return v if v and v.lower() not in ('nan', 'none', 'null') else 'UNK'
    def first_tag(v):
        v = clean(v)
        return clean(v.split(',')[0]) if v != 'UNK' else v
    vid2attrs = {x[0]: (first_tag(x[2]), clean(x[3]), clean(x[4])) for x in vinfo}
    tr = read_rows(f'{a.data_dir}/train.csv',
                   ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])
    va = read_rows(f'{a.data_dir}/valid.csv',
                   ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])

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

    rec_edges = np.array([60_000, 600_000, 3_600_000, 21_600_000,
                          86_400_000, 259_200_000, 604_800_000], dtype=np.int64)
    def rec_value(video, previous, delta):
        if previous is None:
            return (0, 0, 0, 0)
        cur, prev = vid2attrs.get(video, ('UNK',) * 3), vid2attrs.get(previous, ('UNK',) * 3)
        matches = [1 if c == 'UNK' or p == 'UNK' else (3 if c == p else 2) for c, p in zip(cur, prev)]
        return tuple(matches + [1 + int(np.searchsorted(rec_edges, max(int(delta), 0)))])

    tr_time = np.array([int(float(x[5])) for x in tr], dtype=np.int64)
    last_time = np.full(dims[0], -1, dtype=np.int64)
    last_video = np.full(dims[0], None, dtype=object)
    Rtr = np.empty((len(tr), 4), dtype=np.uint8)
    order = np.lexsort((np.arange(len(tr)), tr_time, Xtr[:, 0]))
    p = 0
    while p < len(order):
        q = p + 1; u = Xtr[order[p], 0]; tm = tr_time[order[p]]
        while q < len(order) and Xtr[order[q], 0] == u and tr_time[order[q]] == tm:
            q += 1
        for j in order[p:q]:
            Rtr[j] = rec_value(tr[j][1], last_video[u], tr_time[j] - last_time[u])
        for j in order[p:q]:
            if ytr[j] != 0:
                last_video[u], last_time[u] = tr[j][1], tr_time[j]
        p = q

    def recurrence(rows, ui, vi, timei):
        R = np.empty((len(rows), 4), dtype=np.uint8)
        for n, x in enumerate(rows):
            u = vocabs[0].get(x[ui], unk[0]); tm = int(float(x[timei]))
            R[n] = rec_value(x[vi], last_video[u], tm - last_time[u])
        return R
    rdims = np.array([4, 4, 4, 9], dtype=np.int32)
    roff = (dim + np.cumsum(np.r_[0, rdims[:-1]])).astype(np.int32)
    pdim = int(dim + rdims.sum())
    def with_rec(X, R):
        return np.concatenate((X, R.astype(np.int32) + roff), axis=1)
    Ptr = with_rec(Xtr, Rtr); Pva = with_rec(Xva, recurrence(va, 1, 2, 6))
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    neg_rows = np.flatnonzero(ytr == 0)
    neg_rows = neg_rows[np.argsort(Xtr[neg_rows, 0], kind='stable')]
    neg_count = np.bincount(Xtr[neg_rows, 0], minlength=dims[0])
    neg_start = np.cumsum(np.r_[0, neg_count[:-1]])
    pos_rows = np.flatnonzero(ytr == 1)
    pos_users = Xtr[pos_rows, 0]
    keep = neg_count[pos_users] > 0
    pos_rows, pos_users = pos_rows[keep], pos_users[keep]
    pos_neg_start, pos_neg_count = neg_start[pos_users], neg_count[pos_users]

    # ---- train with early stopping on valid primary ----
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed); rng = np.random.default_rng(a.seed)
    best, best_state, best_pred, best_loss, bad, bcurve = -1.0, None, None, None, 0, []
    for ep in range(1, epochs + 1):
        neg_idx = neg_rows[pos_neg_start + (rng.random(len(pos_rows)) * pos_neg_count).astype(np.int64)]
        idx = rng.permutation(len(pos_rows))
        losses = [m.step(Xtr[pos_rows[idx[i:i + a.batch]]], Xtr[neg_idx[idx[i:i + a.batch]]])
                  for i in range(0, len(idx), a.batch)]
        sva_ep = m.predict(Xva); r = evaluate(uva, yva, sva_ep)
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if r['primary'] > best + 1e-5:
            best, bad, best_pred, best_loss = r['primary'], 0, sva_ep, float(np.mean(losses))
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
        bcurve.append((best_loss, best_pred))
        if bad > 0 and bad >= a.patience:
            break
    m.V, m.W, m.b = best_state

    pm = FM(pdim, k=a.k, lr=a.lr, seed=a.seed); prng = np.random.default_rng(a.seed)
    best, best_state, best_pred, best_loss, bad, pcurve = -1.0, None, None, None, 0, []
    for ep in range(1, epochs + 1):
        idx = prng.permutation(len(ytr))
        losses = [pm.step_pointwise(Ptr[idx[i:i + a.batch]], ytr[idx[i:i + a.batch]])
                  for i in range(0, len(idx), a.batch)]
        sva_ep = pm.predict(Pva); r = evaluate(uva, yva, sva_ep)
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if r['primary'] > best + 1e-5:
            best, bad, best_pred, best_loss = r['primary'], 0, sva_ep, float(np.mean(losses))
            best_state = (pm.V.copy(), pm.W.copy(), np.float32(pm.b))
        else:
            bad += 1
        pcurve.append((best_loss, best_pred))
        if bad > 0 and bad >= a.patience:
            break
    pm.V, pm.W, pm.b = best_state

    history = []
    for ep in range(1, max(len(bcurve), len(pcurve)) + 1):
        bs = bcurve[min(ep, len(bcurve)) - 1]; ps = pcurve[min(ep, len(pcurve)) - 1]
        sva_ep = rank_blend(uva, bs[1], ps[1]); r = evaluate(uva, yva, sva_ep)
        history.append({'epoch': ep, 'train_loss': float(0.85 * bs[0] + 0.15 * ps[0]), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})

    # ---- outputs ----
    sva = rank_blend(uva, bcurve[-1][1], pcurve[-1][1]); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(len(history)), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'time_ms'])
        Xex = encode(ex, 1, 2, 3, 4)
        Pex = with_rec(Xex, recurrence(ex, 1, 2, 5))
        sex = rank_blend([x[1] for x in ex], m.predict(Xex), pm.predict(Pex))
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, sex)
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
