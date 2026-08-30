"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
This node keeps those model settings but replaces pointwise logloss with same-user positive-negative BPR.
Ensemble Five independently seeded BPR models combined by normalized within-user rank averaging.
Adds strictly prior session position, recent-impression density, and previous-gap categorical fields.
Fusion substitutes session-model ranks only for duration >180s or tab-4 rows, retaining base ranks elsewhere.
"""
import argparse, csv, json, os, time
from collections import deque
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket', 'session_pos', 'recent_10m', 'previous_gap']

def read_rows(path, cols):
    """Rows of `path` restricted to `cols`, as lists of strings, in file order."""
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r); idx = [head.index(c) for c in cols]
        return [[rec[i] for i in idx] for rec in r]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def exposure_features(rows, ui, ti, initial=None):
    users = np.fromiter((int(x[ui]) for x in rows), dtype=np.int64, count=len(rows))
    times = np.fromiter((int(x[ti]) for x in rows), dtype=np.int64, count=len(rows))
    order = np.lexsort((np.arange(len(rows), dtype=np.int64), times, users))
    features = np.empty((len(rows), 3), dtype=np.int8)
    state = {} if initial is None else initial.copy()
    p = 0
    while p < len(order):
        oi = int(order[p]); user = int(users[oi])
        last, pos, prior = state.get(user, (None, 0, ()))
        recent = deque(prior)
        while p < len(order) and users[order[p]] == user:
            oi = int(order[p]); now = int(times[oi]); q = p + 1
            while q < len(order) and users[order[q]] == user and times[order[q]] == now:
                q += 1
            gap = None if last is None else now - last
            if gap is None or gap > 1_800_000:
                pos = 0
            while recent and now - recent[0] > 600_000:
                recent.popleft()
            pos_bucket = 0 if pos == 0 else 1 if pos <= 2 else 2 if pos <= 9 else 3 if pos <= 29 else 4
            density_bucket = 0 if not recent else 1 if len(recent) <= 3 else 2 if len(recent) <= 10 else 3
            gap_bucket = (5 if gap is None else 0 if gap < 30_000 else 1 if gap < 120_000 else
                          2 if gap < 600_000 else 3 if gap <= 3_600_000 else 4)
            features[order[p:q]] = (pos_bucket, density_bucket, gap_bucket)
            pos += q - p
            recent.extend([now] * (q - p))
            last = now
            p = q
        state[user] = (last, pos, tuple(recent))
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

def gated_ranks(users, base_predictions, session_predictions, mask):
    base_rank = average_ranks(users, base_predictions)
    session_rank = average_ranks(users, session_predictions)
    fused = base_rank + np.asarray(mask, dtype=np.float64) * (session_rank - base_rank)
    return normalized_ranks(users, fused, base_rank)

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
    Str, train_exposure_state = exposure_features(tr, 0, 5)
    Sva, _ = exposure_features(va, 1, 6, train_exposure_state)

    # ---- encode: 8 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur, session):
        return [user, video, vid2author.get(video, 'UNK'), tab, str(int(np.searchsorted(edges, float(dur))))] + list(session)
    vocabs = [dict() for _ in FIELDS]
    for n, x in enumerate(tr):
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3], Str[n])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32); dim = int(sum(dims))
    def encode(rows, session, ui, vi, ti, di):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di], session[n])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr = encode(tr, Str, 0, 1, 2, 3); ytr = np.array([1.0 if x[4] != '0' else 0.0 for x in tr], dtype=np.float32)
    Xva = encode(va, Sva, 1, 2, 3, 4); yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
    Xtr_base = Xtr[:, :5]; Xva_base = Xva[:, :5]; base_dim = int(sum(dims[:5]))
    target_va = np.fromiter((float(x[4]) > 180_000 or x[3] == '4' for x in va), dtype=bool, count=len(va))
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
    base_models = [FM(base_dim, k=a.k, lr=a.lr, seed=a.seed + s) for s in range(5)]
    base_rngs = [np.random.default_rng(a.seed + s) for s in range(5)]
    best = np.full(5, -1.0); best_state = [None] * 5; best_preds = [None] * 5
    bad = np.zeros(5, dtype=np.int32); active = np.ones(5, dtype=bool); history = []
    base_best = np.full(5, -1.0); base_best_state = [None] * 5; base_best_preds = [None] * 5
    base_bad = np.zeros(5, dtype=np.int32); base_active = np.ones(5, dtype=bool)
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
        for s, m in enumerate(base_models):
            if not base_active[s]:
                continue
            neg_idx = neg_pool[neg_start + (base_rngs[s].random(len(pair_pos)) * neg_count).astype(np.int32)]
            idx = base_rngs[s].permutation(len(pair_pos))
            losses.extend([m.step(Xtr_base[pair_pos[idx[i:i + a.batch]]], Xtr_base[neg_idx[idx[i:i + a.batch]]])
                           for i in range(0, len(idx), a.batch)])
            member_scores = m.predict(Xva_base); member_r = evaluate(uva, yva, member_scores)
            if member_r['primary'] > base_best[s] + 1e-5:
                base_best[s], base_bad[s] = member_r['primary'], 0
                base_best_state[s] = (m.V.copy(), m.W.copy(), np.float32(m.b))
                base_best_preds[s] = member_scores.copy()
            else:
                base_bad[s] += 1
                if base_bad[s] >= a.patience:
                    base_active[s] = False
        r = evaluate(uva, yva, gated_ranks(uva, base_best_preds, best_preds, target_va))
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if not np.any(active) and not np.any(base_active):
            break
    for s, m in enumerate(models):
        m.V, m.W, m.b = best_state[s]
    for s, m in enumerate(base_models):
        m.V, m.W, m.b = base_best_state[s]

    # ---- outputs ----
    sva = gated_ranks(uva, [m.predict(Xva_base) for m in base_models],
                      [m.predict(Xva) for m in models], target_va); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(np.argmax([h['val_primary'] for h in history]) + 1), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'time_ms'])
        Sex, _ = exposure_features(ex, 1, 5, train_exposure_state)
        Xex = encode(ex, Sex, 1, 2, 3, 4)
        target_ex = np.fromiter((float(x[4]) > 180_000 or x[3] == '4' for x in ex), dtype=bool, count=len(ex))
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex,
                          gated_ranks([x[1] for x in ex], [m.predict(Xex[:, :5]) for m in base_models],
                                      [m.predict(Xex) for m in models], target_ex))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
