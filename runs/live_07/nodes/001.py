"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
History Appends latest-prior-positive tag, music and video-type matches plus bucketed recency.
"""
import argparse, csv, json, os, time
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
          'last_pos_tag_match', 'last_pos_music_match', 'last_pos_type_match', 'last_pos_recency']

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

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)              # d(logloss)/d(logit)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

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
    basic = read_rows(f'{a.data_dir}/video_features_basic.csv',
                      ['video_id', 'author_id', 'tag', 'music_id', 'video_type'])
    def clean_attr(v):
        v = v.strip().strip("'\"")
        return v if v and v.lower() not in ('nan', 'none', 'null', '-1') else 'UNK'
    def first_tag(v):
        v = v.strip().strip('[]')
        if ',' in v:
            v = v.split(',', 1)[0]
        elif ' ' in v:
            v = v.split(' ', 1)[0]
        return clean_attr(v)
    vid2author = {x[0]: x[1] for x in basic}
    vid2attrs = {x[0]: (first_tag(x[2]), clean_attr(x[3]), clean_attr(x[4])) for x in basic}
    tr = read_rows(f'{a.data_dir}/train.csv',
                   ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])
    va = read_rows(f'{a.data_dir}/valid.csv',
                   ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'time_ms'])

    recency_edges = np.array([60_000, 600_000, 3_600_000, 21_600_000, 86_400_000,
                              259_200_000, 604_800_000], dtype=np.int64)
    def match_value(current, previous):
        if current == 'UNK' or previous == 'UNK':
            return 'unknown'
        return 'yes' if current == previous else 'no'
    def prior_values(user, video, tm, state):
        previous = state.get(user)
        if previous is None:
            return ('none', 'none', 'none', 'none')
        current = vid2attrs.get(video, ('UNK', 'UNK', 'UNK'))
        recency = str(int(np.searchsorted(recency_edges, max(0, tm - previous[0]), side='right')))
        return (match_value(current[0], previous[1]), match_value(current[1], previous[2]),
                match_value(current[2], previous[3]), recency)
    def build_train_history(rows):
        n = len(rows)
        times = np.fromiter((int(x[5]) for x in rows), dtype=np.int64, count=n)
        users = np.fromiter((int(x[0]) for x in rows), dtype=np.int64, count=n)
        order = np.lexsort((np.arange(n, dtype=np.int32), times, users))
        history, state, p = [None] * n, {}, 0
        while p < n:
            i = int(order[p]); q = p + 1
            while q < n and users[order[q]] == users[i] and times[order[q]] == times[i]:
                q += 1
            user = rows[i][0]
            for k in range(p, q):
                j = int(order[k])
                history[j] = prior_values(user, rows[j][1], int(times[j]), state)
            for k in range(p, q):
                j = int(order[k])
                if rows[j][4] != '0':
                    state[user] = (int(times[j]),) + vid2attrs.get(rows[j][1], ('UNK', 'UNK', 'UNK'))
            p = q
        return history, state
    def score_history(rows, ui, vi, timei, state):
        return [prior_values(x[ui], x[vi], int(x[timei]), state) for x in rows]
    Htr, last_positive = build_train_history(tr)
    Hva = score_history(va, 1, 2, 6, last_positive)

    # ---- encode: 5 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    def raw(user, video, tab, dur, history):
        return [user, video, vid2author.get(video, 'UNK'), tab,
                str(int(np.searchsorted(edges, float(dur))))] + list(history)
    vocabs = [dict() for _ in FIELDS]
    for x, h in zip(tr, Htr):
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3], h)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32); dim = int(sum(dims))
    def encode(rows, ui, vi, ti, di, histories):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di], histories[n])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr = encode(tr, 0, 1, 2, 3, Htr); ytr = np.array([1.0 if x[4] != '0' else 0.0 for x in tr], dtype=np.float32)
    Xva = encode(va, 1, 2, 3, 4, Hva); yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(tr):,} valid {len(va):,} dim {dim:,}', flush=True)

    # ---- train with early stopping on valid primary ----
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed); rng = np.random.default_rng(a.seed)
    best, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        losses = [m.step(Xtr[idx[i:i + a.batch]], ytr[idx[i:i + a.batch]]) for i in range(0, len(idx), a.batch)]
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
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'time_ms'])
        Hex = score_history(ex, 1, 2, 5, last_positive)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, m.predict(encode(ex, 1, 2, 3, 4, Hex)))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
