"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
Candidate trains five field-aware and five standard BPR members and blends their within-user ranks 0.6/0.4.
Candidate gates the field-aware blend weight using train-only user-video and user-author relation support.
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
    row = np.arange(len(scores))
    if tiebreak is None:
        order = np.lexsort((row, scores, users))
    else:
        order = np.lexsort((row, np.asarray(tiebreak), scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    local = np.arange(len(order)) - np.repeat(starts, sizes)
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = local / np.repeat(np.maximum(sizes - 1, 1), sizes)
    return ranks

def rank_average(users, predictions):
    raw = np.mean(np.stack(predictions), axis=0)
    ranks = np.mean(np.stack([normalized_ranks(users, p) for p in predictions]), axis=0)
    return normalized_ranks(users, ranks, raw)

def rank_blend(users, field_predictions, standard_predictions, confidence):
    field = np.mean(np.stack([normalized_ranks(users, p) for p in field_predictions]), axis=0)
    standard = np.mean(np.stack([normalized_ranks(users, p) for p in standard_predictions]), axis=0)
    field_weight = 0.5 + 0.2 * np.asarray(confidence)
    return normalized_ranks(users, field_weight * field + (1.0 - field_weight) * standard, field)

def relation_count_tables(X, dim):
    user = X[:, 0].astype(np.int64)
    return [np.unique(user * dim + X[:, col], return_counts=True) for col in (1, 2)]

def relation_confidence(X, dim, tables):
    user = X[:, 0].astype(np.int64)
    support = []
    for col, (keys, counts) in zip((1, 2), tables):
        query = user * dim + X[:, col]
        pos = np.searchsorted(keys, query)
        clipped = np.minimum(pos, len(keys) - 1)
        found = (pos < len(keys)) & (keys[clipped] == query)
        support.append(np.where(found, counts[clipped], 0).astype(np.float64))
    combined = support[0] + 0.5 * support[1]
    return combined / (combined + 3.0)

class FM:
    """Standard or field-aware FM trained with same-user BPR and Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, standard=False):
        rng = np.random.default_rng(seed)
        self.standard = standard
        shape = (dim, k) if standard else (dim, len(FIELDS), k)
        self.V = rng.normal(0, 0.01, shape).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        if not self.standard:
            inter = np.zeros(len(X), dtype=np.float32)
            for i in range(len(FIELDS)):
                for j in range(i + 1, len(FIELDS)):
                    inter += (self.V[X[:, i], j] * self.V[X[:, j], i]).sum(1)
            return self.b + self.W[X].sum(1) + inter, None, None
        E = self.V[X]                                              # (B, F, k)
        S = E.sum(1)                                               # (B, k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))    # sum of all pairwise dot products
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, Xp, Xn):
        B = len(Xp)
        zp, _, _ = self.logits(Xp); zn, _, _ = self.logits(Xn)
        d = zp - zn
        g = ((sigmoid(d) - 1.0) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        def accumulate(X, h):
            np.add.at(gW, X, h[:, None])
            if self.standard:
                E = self.V[X]; S = E.sum(1)
                np.add.at(gV, X, h[:, None, None] * (S[:, None, :] - E))
            else:
                for i in range(len(FIELDS)):
                    for j in range(i + 1, len(FIELDS)):
                        Ei = self.V[X[:, i], j]; Ej = self.V[X[:, j], i]
                        np.add.at(gV[:, j], X[:, i], h[:, None] * Ej)
                        np.add.at(gV[:, i], X[:, j], h[:, None] * Ei)
        accumulate(Xp, g); accumulate(Xn, -g)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
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
    support_tables = relation_count_tables(Xtr, dim)
    confidence_va = relation_confidence(Xva, dim, support_tables)
    user_order = np.argsort(Xtr[:, 0], kind='stable')
    starts = np.r_[0, np.flatnonzero(Xtr[user_order[1:], 0] != Xtr[user_order[:-1], 0]) + 1]
    ends = np.r_[starts[1:], len(user_order)]
    pair_pos, neg_pool, neg_start, neg_count, neg_total = [], [], [], [], 0
    for lo, hi in zip(starts, ends):
        rows = user_order[lo:hi]
        pos = rows[ytr[rows] > 0.5]; neg = rows[ytr[rows] < 0.5]
        if len(pos) and len(neg):
            pair_pos.append(pos); neg_pool.append(neg)
            neg_start.append(np.full(len(pos), neg_total, dtype=np.int64))
            neg_count.append(np.full(len(pos), len(neg), dtype=np.int64))
            neg_total += len(neg)
    pair_pos = np.concatenate(pair_pos); neg_pool = np.concatenate(neg_pool)
    neg_start = np.concatenate(neg_start); neg_count = np.concatenate(neg_count)
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    # ---- train with early stopping on valid primary ----
    models, member_histories, member_curves, best_epochs = [], [], [], []
    for branch in range(2):
        for member in range(5):
            member_seed = a.seed + branch * 5 + member
            m = FM(dim, k=a.k, lr=a.lr, seed=member_seed, standard=bool(branch))
            rng = np.random.default_rng(member_seed)
            best, best_state, best_pred, best_epoch, bad, history, curve = -1.0, None, None, 0, 0, [], []
            for ep in range(1, epochs + 1):
                sampled_neg = neg_pool[neg_start + (rng.random(len(pair_pos)) * neg_count).astype(np.int64)]
                idx = rng.permutation(len(pair_pos))
                losses = [m.step(Xtr[pair_pos[idx[i:i + a.batch]]], Xtr[sampled_neg[idx[i:i + a.batch]]])
                          for i in range(0, len(idx), a.batch)]
                pva = m.predict(Xva); r = evaluate(uva, yva, pva)
                history.append({'epoch': ep, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                                'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
                print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
                      f"primary {r['primary']:.4f}", flush=True)
                if r['primary'] > best + 1e-5:
                    best, bad, best_epoch = r['primary'], 0, ep
                    best_state, best_pred = (m.V.copy(), m.W.copy(), np.float32(m.b)), pva
                else:
                    bad += 1
                curve.append(best_pred)
                if bad >= a.patience:
                    break
            m.V, m.W, m.b = best_state
            models.append(m); member_histories.append(history); member_curves.append(curve); best_epochs.append(best_epoch)

    history = []
    for ep in range(max(len(h) for h in member_histories)):
        pva = rank_blend(uva,
                         [c[min(ep, len(c) - 1)] for c in member_curves[:5]],
                         [c[min(ep, len(c) - 1)] for c in member_curves[5:]],
                         confidence_va)
        r = evaluate(uva, yva, pva)
        losses = [h[min(ep, len(h) - 1)]['train_loss'] for h in member_histories]
        history.append({'epoch': ep + 1, 'train_loss': float(np.mean(losses)), 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})

    # ---- outputs ----
    sva = rank_blend(uva, [m.predict(Xva) for m in models[:5]], [m.predict(Xva) for m in models[5:]], confidence_va)
    r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': max(best_epochs), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        Xex = encode(ex, 1, 2, 3, 4)
        confidence_ex = relation_confidence(Xex, dim, support_tables)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex,
                          rank_blend([x[1] for x in ex],
                                     [m.predict(Xex) for m in models[:5]],
                                     [m.predict(Xex) for m in models[5:]],
                                     confidence_ex))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
