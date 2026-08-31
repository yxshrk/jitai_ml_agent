"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
History Adds leakage-safe user-by-author, user-by-tab, and user-by-duration prior-rate fields.
"""
import argparse, csv, json, os, time
import numpy as np
from evaluate import evaluate   # official scorer (copied into the workspace by the harness)

FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket',
          'author_hist_rate', 'tab_hist_rate', 'duration_hist_rate']

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
    vid2author = dict(read_rows(f'{a.data_dir}/video_features_basic.csv', ['video_id', 'author_id']))
    tr = read_rows(f'{a.data_dir}/train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'time_ms', 'long_view'])
    va = read_rows(f'{a.data_dir}/valid.csv', ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view'])

    # ---- encode: 8 categorical fields -> contiguous ids; unseen values fall into a per-field UNK slot ----
    edges = np.quantile(np.array([float(x[3]) for x in tr]), np.linspace(0, 1, 11)[1:-1])   # 10 duration buckets
    ytr = np.array([1.0 if x[5] != '0' else 0.0 for x in tr], dtype=np.float32)
    times = np.fromiter((int(x[4]) for x in tr), dtype=np.int64, count=len(tr))
    authors = [vid2author.get(x[1], 'UNK') for x in tr]
    def factorize(values):
        vocab = {}
        code = np.fromiter((vocab.setdefault(v, len(vocab)) for v in values),
                           dtype=np.int64, count=len(tr))
        return code, vocab
    user_code, user_hist_vocab = factorize(x[0] for x in tr)
    author_code, author_hist_vocab = factorize(authors)
    tab_code, tab_hist_vocab = factorize(x[2] for x in tr)
    dur_code = np.fromiter((np.searchsorted(edges, float(x[3])) for x in tr),
                           dtype=np.int64, count=len(tr))

    def earlier_stats(keys):
        order = np.lexsort((np.arange(len(keys), dtype=np.int64), times, keys))
        sk, st, sy = keys[order], times[order], ytr[order]
        key_start = np.r_[True, sk[1:] != sk[:-1]]
        time_start = key_start.copy(); time_start[1:] |= st[1:] != st[:-1]
        pos = np.arange(len(keys), dtype=np.int64)
        ks = np.maximum.accumulate(np.where(key_start, pos, 0))
        ts = np.maximum.accumulate(np.where(time_start, pos, 0))
        cs = np.r_[0.0, np.cumsum(sy, dtype=np.float64)]
        n = np.empty(len(keys), dtype=np.int32); p = np.empty(len(keys), dtype=np.float32)
        n[order] = (ts - ks).astype(np.int32); p[order] = (cs[ts] - cs[ks]).astype(np.float32)
        starts = pos[key_start]; ends = np.r_[starts[1:], len(keys)]
        totals = (sk[starts].copy(), (ends - starts).astype(np.int32),
                  (cs[ends] - cs[starts]).astype(np.float32))
        return n, p, totals

    user_n, user_p, user_totals = earlier_stats(user_code)
    global_rate = float(ytr.mean())
    user_prior = np.full(len(tr), global_rate, dtype=np.float32)
    np.divide(user_p, user_n, out=user_prior, where=user_n > 0)
    hist_tr = np.zeros((len(tr), 3), dtype=np.int8)
    hist_edges, relation_totals = [], []
    relation_values = [author_code, tab_code, dur_code]
    relation_widths = [len(author_hist_vocab), len(tab_hist_vocab), 10]
    for j, (values, width) in enumerate(zip(relation_values, relation_widths)):
        key = user_code * width + values
        rn, rp, total = earlier_stats(key)
        rate = (rp + 5.0 * user_prior) / (rn + 5.0)
        seen = rn > 0
        q = np.quantile(rate[seen], np.linspace(0, 1, 11)[1:-1])
        hist_tr[seen, j] = 1 + np.searchsorted(q, rate[seen], side='right')
        hist_edges.append(q); relation_totals.append(total)

    def mapped(values, vocab, n):
        return np.fromiter((vocab.get(v, -1) for v in values), dtype=np.int64, count=n)
    def lookup(keys, totals):
        unique, counts, positives = totals
        ix = np.searchsorted(unique, keys); safe = np.minimum(ix, len(unique) - 1)
        ok = (keys >= 0) & (ix < len(unique)) & (unique[safe] == keys)
        n = np.zeros(len(keys), dtype=np.float32); p = np.zeros(len(keys), dtype=np.float32)
        n[ok] = counts[ix[ok]]; p[ok] = positives[ix[ok]]
        return n, p
    def history_fields(rows, ui, vi, ti, di):
        n = len(rows)
        uc = mapped((x[ui] for x in rows), user_hist_vocab, n)
        ac = mapped((vid2author.get(x[vi], 'UNK') for x in rows), author_hist_vocab, n)
        tc = mapped((x[ti] for x in rows), tab_hist_vocab, n)
        dc = np.fromiter((np.searchsorted(edges, float(x[di])) for x in rows), dtype=np.int64, count=n)
        un, up = lookup(uc, user_totals)
        prior = np.full(n, global_rate, dtype=np.float32)
        np.divide(up, un, out=prior, where=un > 0)
        hist = np.zeros((n, 3), dtype=np.int8)
        for j, (values, width, totals) in enumerate(zip([ac, tc, dc], relation_widths, relation_totals)):
            keys = np.where((uc >= 0) & (values >= 0), uc * width + values, -1)
            rn, rp = lookup(keys, totals)
            rate = (rp + 5.0 * prior) / (rn + 5.0)
            hist[:, j] = np.where(rn > 0, 1 + np.searchsorted(hist_edges[j], rate, side='right'), 0)
        return hist

    hist_va = history_fields(va, 1, 2, 3, 4)
    def raw(user, video, tab, dur, hist):
        return [user, video, vid2author.get(video, 'UNK'), tab,
                str(int(np.searchsorted(edges, float(dur))))] + list(hist)
    vocabs = [dict() for _ in FIELDS]
    for i in range(5, len(FIELDS)):
        vocabs[i].update((j, j) for j in range(11))
    for n, x in enumerate(tr):
        for i, v in enumerate(raw(x[0], x[1], x[2], x[3], hist_tr[n])):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]; dims = [len(v) + 1 for v in vocabs]
    off = np.cumsum([0] + dims[:-1]).astype(np.int32); dim = int(sum(dims))
    def encode(rows, ui, vi, ti, di, hist):
        X = np.empty((len(rows), len(FIELDS)), dtype=np.int32)
        for n, x in enumerate(rows):
            for i, v in enumerate(raw(x[ui], x[vi], x[ti], x[di], hist[n])):
                X[n, i] = vocabs[i].get(v, unk[i]) + off[i]
        return X
    Xtr = encode(tr, 0, 1, 2, 3, hist_tr)
    Xva = encode(va, 1, 2, 3, 4, hist_va); yva = [1 if x[5] != '0' else 0 for x in va]; uva = [x[1] for x in va]
    del tr, authors, user_code, author_code, tab_code, dur_code, relation_values
    del user_n, user_p, user_prior, hist_tr, hist_va, times, values, key, rn, rp, rate, seen, q, total
    print(f'loaded+encoded in {time.time() - t0:.0f}s: train {len(ytr):,} valid {len(va):,} dim {dim:,}', flush=True)

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
        ex = read_rows(a.score_extra, ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
        hist_ex = history_fields(ex, 1, 2, 3, 4)
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, m.predict(encode(ex, 1, 2, 3, 4, hist_ex)))
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
