"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
This node keeps those model settings but replaces pointwise logloss with same-user positive-negative BPR.
Ensemble Five independently seeded BPR models combined by normalized within-user rank averaging.
Adds strictly prior video and author 24-hour exposure-count and previous-gap categorical fields.
"""
import argparse, csv, json, os, time
from collections import deque
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
          'video_24h_count', 'video_previous_gap', 'author_24h_count', 'author_previous_gap']

def read_rows(path, cols):
    """Rows of `path` restricted to `cols`, as lists of strings, in file order."""
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r); idx = [head.index(c) for c in cols]
        return [[rec[i] for i in idx] for rec in r]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def momentum_features(rows, vi, ti, vid2author, initial=None):
    videos = [x[vi] for x in rows]
    authors = [vid2author.get(video, 'UNK') for video in videos]
    times = np.fromiter((int(x[ti]) for x in rows), dtype=np.int64, count=len(rows))
    order = np.argsort(times, kind='stable')
    features = np.empty((len(rows), 4), dtype=np.int8)
    if initial is None:
        video_recent, video_last, author_recent, author_last = {}, {}, {}, {}
    else:
        vr, vl, ar, al = initial
        video_recent = {k: deque(v) for k, v in vr.items()}; video_last = vl.copy()
        author_recent = {k: deque(v) for k, v in ar.items()}; author_last = al.copy()
    window = 86_400_000
    p = 0
    while p < len(order):
        now = int(times[order[p]]); q = p + 1
        while q < len(order) and times[order[q]] == now:
            q += 1
        video_values, author_values = {}, {}
        for oi0 in order[p:q]:
            oi = int(oi0); video = videos[oi]; author = authors[oi]
            vv = video_values.get(video)
            if vv is None:
                recent = video_recent.get(video)
                if recent is None:
                    recent = deque(); video_recent[video] = recent
                while recent and now - recent[0] > window:
                    recent.popleft()
                gap = None if video not in video_last else now - video_last[video]
                vv = (min(len(recent).bit_length(), 12),
                      0 if gap is None else min(max(gap // 1000, 1).bit_length(), 20))
                video_values[video] = vv
            av = author_values.get(author)
            if av is None:
                recent = author_recent.get(author)
                if recent is None:
                    recent = deque(); author_recent[author] = recent
                while recent and now - recent[0] > window:
                    recent.popleft()
                gap = None if author not in author_last else now - author_last[author]
                av = (min(len(recent).bit_length(), 12),
                      0 if gap is None else min(max(gap // 1000, 1).bit_length(), 20))
                author_values[author] = av
            features[oi] = (vv[0], vv[1], av[0], av[1])
        for oi0 in order[p:q]:
            oi = int(oi0); video = videos[oi]; author = authors[oi]
            video_recent[video].append(now); video_last[video] = now
            author_recent[author].append(now); author_last[author] = now
        p = q
    if len(order):
        end = int(times[order[-1]])
        for recent in video_recent.values():
            while recent and end - recent[0] > window:
                recent.popleft()
        for recent in author_recent.values():
            while recent and end - recent[0] > window:
                recent.popleft()
    state = ({k: tuple(v) for k, v in video_recent.items() if v}, video_last,
             {k: tuple(v) for k, v in author_recent.items() if v}, author_last)
    return features, state

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

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with BPR + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)   # one k-vector per feature value
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
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
        g = (-sigmoid(-d) / B).astype(np.float32)                  # d(BPR loss)/d(score_pos-score_neg)
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
        return float(np.mean(np.logaddexp(0.0, -d)))

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
    tr = read_rows(f'{a.data_dir}/train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])
    va = read_rows(f'{a.data_dir}/valid.csv', ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])
    Mtr, train_momentum_state = momentum_features(tr, 1, 5, vid2author)
    Mva, _ = momentum_features(va, 2, 6, vid2author, train_momentum_state)

    # ---- encode: 9 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur, momentum):
        return [user, video, vid2author.get(video, 'UNK'), tab, str(int(np.searchsorted(edges, float(dur))))] + list(momentum)
    vocabs = [dict() for _ in FIELDS]
    for n, x in enumerate(tr):
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3], Mtr[n])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32); dim = int(sum(dims))
    def encode(rows, momentum, ui, vi, ti, di):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di], momentum[n])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr = encode(tr, Mtr, 0, 1, 2, 3); ytr = np.array([1.0 if x[4] != '0' else 0.0 for x in tr], dtype=np.float32)
    Xva = encode(va, Mva, 1, 2, 3, 4); yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

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
    models = [FM(dim, k=a.k, lr=a.lr, seed=a.seed + s) for s in range(5)]
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
            losses.extend([m.step(Xtr[pair_pos[idx[i:i + a.batch]]], Xtr[neg_idx[idx[i:i + a.batch]]])
                           for i in range(0, len(idx), a.batch)])
            member_scores = m.predict(Xva); member_r = evaluate(uva, yva, member_scores)
            if member_r['primary'] > best[s] + 1e-5:
                best[s], bad[s] = member_r['primary'], 0
                best_state[s] = (m.V.copy(), m.W.copy(), np.float32(m.b))
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
        m.V, m.W, m.b = best_state[s]

    # ---- outputs ----
    sva = average_ranks(uva, [m.predict(Xva) for m in models]); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'time_ms'])
        Mex, _ = momentum_features(ex, 2, 5, vid2author, train_momentum_state)
        Xex = encode(ex, Mex, 1, 2, 3, 4)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex,
                          average_ranks([x[1] for x in ex], [m.predict(Xex) for m in models]))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
