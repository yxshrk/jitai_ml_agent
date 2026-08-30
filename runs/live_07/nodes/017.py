"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
This node keeps those model settings but replaces pointwise logloss with same-user positive-negative BPR.
Ensemble Five independently seeded BPR models combined by normalized within-user rank averaging.
Attention Each member adds an 8-dimensional masked residual over the 20 latest strictly earlier positive attributes.
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
    tie = np.arange(len(scores)) if tiebreak is None else np.asarray(tiebreak)
    order = np.lexsort((np.arange(len(scores)), tie, scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    counts = np.diff(starts)
    rank = np.arange(len(order)) - np.repeat(starts[:-1], counts)
    out = np.empty(len(order), dtype=np.float64)
    out[order] = rank / np.maximum(np.repeat(counts, counts) - 1, 1)
    return out

def average_ranks(users, predictions):
    ranks = [normalized_ranks(users, scores) for scores in predictions]
    return normalized_ranks(users, np.mean(ranks, axis=0), ranks[0])

def first_tag(value):
    tag = value.strip().strip('[]').split(',')[0].strip().strip("'\"")
    return tag or 'UNK'

def positive_history_layout(rows, attrs, labels, ui, ti):
    users = np.asarray([x[ui] for x in rows])
    times = np.asarray([int(float(x[ti])) for x in rows], dtype=np.int64)
    order = np.lexsort((np.arange(len(rows)), times, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    history = [np.zeros((1, attrs.shape[1]), dtype=np.int32)]
    hstart = np.empty(len(rows), dtype=np.int32); hend = np.empty(len(rows), dtype=np.int32)
    totals = {}; cursor = 1
    for lo, hi in zip(starts[:-1], starts[1:]):
        ix = order[lo:hi]
        positive = ix[labels[ix] > 0]
        count = np.searchsorted(times[positive], times[ix], side='left')
        hstart[ix] = cursor; hend[ix] = cursor + count
        totals[str(users[ix[0]])] = (cursor, cursor + len(positive))
        if len(positive):
            history.append(attrs[positive])
        cursor += len(positive)
    return np.concatenate(history, axis=0), hstart, hend, totals

def fixed_history(rows, ui, totals):
    bounds = np.asarray([totals.get(str(x[ui]), (0, 0)) for x in rows], dtype=np.int32).reshape(-1, 2)
    return bounds[:, 0], bounds[:, 1]

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with BPR + Adam."""
    def __init__(self, dim, att_dim, history_attr, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # one k-vector per feature value
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.A = rng.normal(0, 0.03, (att_dim, 8)).astype(np.float32)
        self.history_attr = history_attr
        self.back = np.arange(20, 0, -1, dtype=np.int32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mA = np.zeros_like(self.A); self.vA = np.zeros_like(self.A)
        self.t = 0

    def attention(self, attrs, hstart, hend):
        idx = hend[:, None] - self.back
        mask = idx >= hstart[:, None]
        idx = np.maximum(idx, 0)
        hattrs = self.history_attr[idx]
        q = np.zeros((len(attrs), 8), dtype=np.float32)
        k = np.zeros((len(attrs), 20, 8), dtype=np.float32)
        for j in range(attrs.shape[1]):
            q += self.A[attrs[:, j]]
            k += self.A[hattrs[:, :, j]]
        q *= np.float32(0.25); k *= np.float32(0.25)
        scale = np.float32(1.0 / np.sqrt(8.0))
        alog = (k * q[:, None, :]).sum(2) * scale
        masked = np.where(mask, alog, np.float32(-1e9))
        peak = masked.max(1, keepdims=True)
        weight = np.exp(np.where(mask, alog - peak, np.float32(-80.0))).astype(np.float32)
        weight *= mask
        weight /= np.maximum(weight.sum(1, keepdims=True), np.float32(1.0))
        return (weight * alog).sum(1), (attrs, hattrs, q, k, weight, alog)

    def attention_grad(self, cache, grad, gA):
        attrs, hattrs, q, k, weight, alog = cache
        score = (weight * alog).sum(1)
        dl = grad[:, None] * weight * (np.float32(1.0) + alog - score[:, None])
        scale = np.float32(1.0 / np.sqrt(8.0))
        gq = (dl[:, :, None] * k).sum(1) * scale
        gk = dl[:, :, None] * q[:, None, :] * scale
        ids = np.concatenate((attrs.reshape(-1), hattrs.reshape(-1)))
        for j in range(8):
            values = np.concatenate((np.repeat(gq[:, j] * np.float32(0.25), attrs.shape[1]),
                                     np.repeat(gk[:, :, j].reshape(-1) * np.float32(0.25),
                                               hattrs.shape[2])))
            gA[:, j] += np.bincount(ids, weights=values, minlength=len(gA)).astype(np.float32)

    def logits(self, X, attrs, hstart, hend):
        E = self.V[X]                                              # (B, F, k)
        S = E.sum(1)                                               # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))    # sum of all pairwise dot products
        att, cache = self.attention(attrs, hstart, hend)
        return self.b + self.W[X].sum(1) + inter + att, E, S, cache

    def step(self, Xp, Ap, hsp, hep, Xn, An, hsn, hen):
        B = len(Xp)
        zp, Ep, Sp, Cp = self.logits(Xp, Ap, hsp, hep)
        zn, En, Sn, Cn = self.logits(Xn, An, hsn, hen)
        d = zp - zn
        g = (-sigmoid(-d) / B).astype(np.float32)                  # d(BPR loss)/d(score_pos-score_neg)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W); gA = np.zeros_like(self.A)
        np.add.at(gW, Xp, g[:, None])
        np.add.at(gW, Xn, -g[:, None])
        np.add.at(gV, Xp, g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, -g[:, None, None] * (Sn[:, None, :] - En))
        self.attention_grad(Cp, g, gA); self.attention_grad(Cn, -g, gA)
        gV += self.l2 * self.V; gW += self.l2 * self.W; gA += self.l2 * self.A
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW),
                            (self.A, gA, self.mA, self.vA)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(np.mean(np.logaddexp(0.0, -d)))

    def predict(self, X, attrs, hstart, hend, bs=20_000):
        return np.concatenate([self.logits(X[i:i + bs], attrs[i:i + bs], hstart[i:i + bs],
                                           hend[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

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
    video_rows = read_rows(f'{a.data_dir}/video_features_basic.csv', ['video_id', 'author_id', 'tag'])
    vid2author = {x[0]: x[1] for x in video_rows}
    vid2tag = {x[0]: first_tag(x[2]) for x in video_rows}
    tr = read_rows(f'{a.data_dir}/train.csv',
                   ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])
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
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    tag_vocab = {}
    for x in tr:
        tag = vid2tag.get(x[1], 'UNK')
        if tag not in tag_vocab:
            tag_vocab[tag] = len(tag_vocab)
    tag_unk = len(tag_vocab); att_dim = dim + tag_unk + 1
    def attention_attrs(rows, X, vi):
        tags = np.fromiter((tag_vocab.get(vid2tag.get(x[vi], 'UNK'), tag_unk) + dim for x in rows),
                           dtype=np.int32, count=len(rows))
        return np.column_stack((X[:, 2], tags, X[:, 4], X[:, 3])).astype(np.int32)
    Atr = attention_attrs(tr, Xtr, 1); Ava = attention_attrs(va, Xva, 2)
    history_attr, hstr, hend, history_totals = positive_history_layout(tr, Atr, ytr, 0, 5)
    hsv, hev = fixed_history(va, 1, history_totals)

    # ---- precompute positive rows and per-user negative pools for BPR ----
    by_user = {}
    for i, x in enumerate(tr):
        by_user.setdefault(x[0], ([], []))[int(ytr[i])].append(i)
    pair_pos, neg_pool, neg_start, neg_count = [], [], [], []
    for neg, pos in by_user.values():
        if pos and neg:
            start = len(neg_pool)
            neg_pool.extend(neg); pair_pos.extend(pos)
            neg_start.extend([start] * len(pos)); neg_count.extend([len(neg)] * len(pos))
    pair_pos = np.asarray(pair_pos, dtype=np.int32)
    neg_pool = np.asarray(neg_pool, dtype=np.int32)
    neg_start = np.asarray(neg_start, dtype=np.int32)
    neg_count = np.asarray(neg_count, dtype=np.int32)
    del by_user

    # ---- train with early stopping on valid primary ----
    models = [FM(dim, att_dim, history_attr, k=a.k, lr=a.lr, seed=a.seed + s) for s in range(5)]
    rngs = [np.random.default_rng(a.seed + s) for s in range(5)]
    best = np.full(5, -1.0); best_state = [None] * 5; best_preds = [None] * 5
    bad = np.zeros(5, dtype=np.int32); active = np.ones(5, dtype=bool); history = []
    for ep in range(1, epochs + 1):
        losses = []
        for s, m in enumerate(models):
            if not active[s]:
                continue
            neg_idx = neg_pool[neg_start + (rngs[s].random(len(pair_pos)) * neg_count).astype(np.int32)]
            idx = rngs[s].permutation(len(pair_pos))
            for i in range(0, len(idx), a.batch):
                p = pair_pos[idx[i:i + a.batch]]; n = neg_idx[idx[i:i + a.batch]]
                losses.append(m.step(Xtr[p], Atr[p], hstr[p], hend[p],
                                     Xtr[n], Atr[n], hstr[n], hend[n]))
            member_scores = m.predict(Xva, Ava, hsv, hev); member_r = evaluate(uva, yva, member_scores)
            if member_r['primary'] > best[s] + 1e-5:
                best[s], bad[s] = member_r['primary'], 0
                best_state[s] = (m.V.copy(), m.W.copy(), np.float32(m.b), m.A.copy())
                best_preds[s] = member_scores.copy()
            else:
                bad[s] += 1
                if bad[s] >= a.patience:
                    active[s] = False
        r = evaluate(uva, yva, average_ranks(uva, best_preds))
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if not np.any(active):
            break
    for s, m in enumerate(models):
        m.V, m.W, m.b, m.A = best_state[s]

    # ---- outputs ----
    sva = average_ranks(uva, [m.predict(Xva, Ava, hsv, hev) for m in models]); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        Xex = encode(ex, 1, 2, 3, 4)
        Aex = attention_attrs(ex, Xex, 2); hse, hee = fixed_history(ex, 1, history_totals)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex,
                          average_ranks([x[1] for x in ex],
                                        [m.predict(Xex, Aex, hse, hee) for m in models]))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
