"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
Loss    Same-user logistic BPR with one uniformly sampled negative per positive impression.
Ensemble Five field-aware and five standard members blended by tie-free within-user ranks.
Feature Session fields are restricted to field-aware members; standard members remain the five-field anchor.
"""
import argparse, csv, json, os, time
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
FIELD_FIELDS = FIELDS + ['session_pos', 'recent_10m', 'previous_gap']

def read_rows(path, cols):
    """Rows of `path` restricted to `cols`, as lists of strings, in file order."""
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r); idx = [head.index(c) for c in cols]
        return [[rec[i] for i in idx] for rec in r]

def session_buckets(history, rows, hui, hti, ui, ti):
    """Causal session fields for rows, after advancing state with strictly earlier history and row timestamps."""
    husers = np.fromiter((int(x[hui]) for x in history), dtype=np.int64, count=len(history))
    users = np.r_[husers, np.fromiter((int(x[ui]) for x in rows), dtype=np.int64, count=len(rows))]
    htimes = np.fromiter((int(x[hti]) for x in history), dtype=np.int64, count=len(history))
    times = np.r_[htimes, np.fromiter((int(x[ti]) for x in rows), dtype=np.int64, count=len(rows))]
    source = np.r_[np.full(len(history), -1, dtype=np.int64), np.arange(len(rows), dtype=np.int64)]
    order = np.lexsort((np.arange(len(users), dtype=np.int64), times, users))
    users, times, source = users[order], times[order], source[order]
    out = np.empty((len(rows), 3), dtype=np.int8)
    cuts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        left, pos, prev, j = lo, 0, None, lo
        while j < hi:
            k = j + 1
            while k < hi and times[k] == times[j]:
                k += 1
            now = times[j]
            while left < j and times[left] < now - 600_000:
                left += 1
            gap = None if prev is None else now - prev
            if prev is None or gap > 1_800_000:
                pos = 0
            p = pos + 1
            pb = 0 if p == 1 else 1 if p <= 3 else 2 if p <= 10 else 3 if p <= 30 else 4
            recent = j - left
            rb = 0 if recent == 0 else 1 if recent <= 3 else 2 if recent <= 10 else 3
            gb = 5 if gap is None else 0 if gap < 30_000 else 1 if gap < 120_000 else 2 if gap < 600_000 else 3 if gap < 3_600_000 else 4
            q = source[j:k]; q = q[q >= 0]
            out[q] = (pb, rb, gb)
            pos += k - j
            prev, j = now, k
    return out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def normalized_ranks(users, scores, tiebreak=None):
    users, scores = np.asarray(users), np.asarray(scores)
    tie = np.zeros(len(scores), dtype=np.float64) if tiebreak is None else np.asarray(tiebreak)
    order = np.lexsort((np.arange(len(scores)), tie, scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    out = np.empty(len(scores), dtype=np.float64)
    for start, end in zip(starts, np.r_[starts[1:], len(scores)]):
        out[order[start:end]] = np.arange(end - start) / max(end - start - 1, 1)
    return out

def rank_blend(users, standard_scores, field_scores):
    standard = np.mean([normalized_ranks(users, x) for x in standard_scores], axis=0)
    field = np.mean([normalized_ranks(users, x) for x in field_scores], axis=0)
    return normalized_ranks(users, 0.6 * field + 0.4 * standard, field)

class FM:
    """score = b + sum_i w[x_i] + sum_{i<j} <V[x_i], V[x_j]>, trained with same-user BPR + Adam."""
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, field_aware=False, n_fields=None):
        rng = np.random.default_rng(seed)
        self.field_aware = field_aware
        self.n_fields = len(FIELDS) if n_fields is None else n_fields
        shape = (dim, self.n_fields, k) if field_aware else (dim, k)
        self.V = rng.normal(0, 0.01, shape).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)                     # one bias per feature value
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        if self.field_aware:
            inter = np.zeros(len(X), dtype=np.float32)
            for i in range(self.n_fields):
                for j in range(i + 1, self.n_fields):
                    inter += (self.V[X[:, i], j] * self.V[X[:, j], i]).sum(1)
            return self.b + self.W[X].sum(1) + inter, None, None
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
        if self.field_aware:
            for X, h in ((Xp, g), (Xn, -g)):
                for i in range(self.n_fields):
                    for j in range(i + 1, self.n_fields):
                        Ei, Ej = self.V[X[:, i], j], self.V[X[:, j], i]
                        np.add.at(gV[:, j], X[:, i], h[:, None] * Ej)
                        np.add.at(gV[:, i], X[:, j], h[:, None] * Ei)
        else:
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
    Str = session_buckets([], tr, 0, 0, 0, 5)
    Sva = session_buckets(tr, va, 0, 5, 1, 6)

    # ---- encode: 5 base and 3 field-aware-only categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur, session):
        return [user, video, vid2author.get(video, 'UNK'), tab, str(int(np.searchsorted(edges, float(dur))))] + [str(int(v)) for v in session]
    vocabs = [dict() for _ in FIELD_FIELDS]
    for n, x in enumerate(tr):
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3], Str[n])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32)
    dim = int(sum(dims[:len(FIELDS)])); field_dim = int(sum(dims))
    def encode(rows, session, ui, vi, ti, di):
        X = np.empty((len(rows), len(FIELD_FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di], session[n])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr_field = encode(tr, Str, 0, 1, 2, 3); Xtr = Xtr_field[:, :len(FIELDS)]
    ytr = np.array([1.0 if x[4] != '0' else 0.0 for x in tr], dtype=np.float32)
    Xva_field = encode(va, Sva, 1, 2, 3, 4); Xva = Xva_field[:, :len(FIELDS)]
    yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
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
    models, member_curves, member_best_epochs = [[], []], [[], []], [[], []]
    for branch, field_aware in enumerate((True, False)):
        Xtr_branch, Xva_branch = (Xtr_field, Xva_field) if field_aware else (Xtr, Xva)
        for member in range(5):
            member_seed = a.seed + branch * 5 + member
            m = FM(field_dim if field_aware else dim, k=a.k, lr=a.lr, seed=member_seed,
                   field_aware=field_aware, n_fields=len(FIELD_FIELDS) if field_aware else None)
            rng = np.random.default_rng(member_seed)
            best, best_state, best_epoch, best_loss, best_pred, bad, curve = -1.0, None, 0, None, None, 0, []
            for ep in range(1, epochs + 1):
                neg_idx = neg_rows[pos_neg_start + (rng.random(len(pos_rows)) * pos_neg_count).astype(np.int64)]
                idx = rng.permutation(len(pos_rows))
                losses = [m.step(Xtr_branch[pos_rows[idx[i:i + a.batch]]], Xtr_branch[neg_idx[idx[i:i + a.batch]]])
                          for i in range(0, len(idx), a.batch)]
                sva_ep = m.predict(Xva_branch); r = evaluate(uva, yva, sva_ep)
                print(f"epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
                      f"primary {r['primary']:.4f}", flush=True)
                if r['primary'] > best + 1e-5:
                    best, bad, best_epoch = r['primary'], 0, ep
                    best_loss, best_pred = float(np.mean(losses)), sva_ep
                    best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
                else:
                    bad += 1
                curve.append((best_loss, best_pred))
                if bad >= a.patience:
                    break
            m.V, m.W, m.b = best_state
            m.mV = m.vV = m.mW = m.vW = None
            models[branch].append(m); member_curves[branch].append(curve)
            member_best_epochs[branch].append(best_epoch)

    history = []
    history_epochs = max(max(x) for x in member_best_epochs)
    for ep in range(1, history_epochs + 1):
        field_states = [x[min(ep, len(x)) - 1] for x in member_curves[0]]
        standard_states = [x[min(ep, len(x)) - 1] for x in member_curves[1]]
        sva_ep = rank_blend(uva, [x[1] for x in standard_states], [x[1] for x in field_states])
        r = evaluate(uva, yva, sva_ep)
        history.append({'epoch': ep, 'train_loss': float(np.mean([x[0] for x in field_states + standard_states])),
                        'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})

    # ---- outputs ----
    field_va = [m.predict(Xva_field) for m in models[0]]
    standard_va = [m.predict(Xva) for m in models[1]]
    sva = rank_blend(uva, standard_va, field_va); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': int(history_epochs), 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'time_ms'])
        Sex = session_buckets(tr, ex, 0, 5, 1, 5)
        Xex_field = encode(ex, Sex, 1, 2, 3, 4); Xex = Xex_field[:, :len(FIELDS)]
        uex = [x[1] for x in ex]
        field_ex = [m.predict(Xex_field) for m in models[0]]
        standard_ex = [m.predict(Xex) for m in models[1]]
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex,
                          rank_blend(uex, standard_ex, field_ex))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
