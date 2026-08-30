"""node_000 -- the official FM baseline, ported to the harness contract (see workspace/CONTRACT.md).

Reads   data/train.csv, data/valid.csv, data/video_features_basic.csv
Writes  <out>/predictions.csv (valid rows, in order), <out>/metrics.json (with the per-epoch learning curve),
        <out>/predictions_extra.csv when --score-extra is given.
Model   Factorization Machine over 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket),
        k=16, Adam lr=1e-3, batch 8192, <=40 epochs, early stopping (patience 4) on valid primary --
        the same numbers as kuairand-starter-kit/baseline.py, so seed 0 reproduces valid primary 0.6015.
This node keeps those model settings but replaces pointwise logloss with same-user positive-negative BPR.
This node replaces the FM fit with LightGBM LambdaRank over legal context, history, and session features.
"""
import argparse, csv, json, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
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

def historical_stats(rows, labels, vid2author):
    video = [x[1] for x in rows]; author = [vid2author.get(x[1], 'UNK') for x in rows]
    d = pd.DataFrame({'video': video, 'author': author,
                      'time': [int(float(x[6])) for x in rows], 'label': labels})
    out, totals = [], []
    for col in ('video', 'author'):
        g = d.groupby([col, 'time'], sort=True)['label'].agg(['size', 'sum']).sort_index()
        past_n = g.groupby(level=0, sort=False)['size'].cumsum() - g['size']
        past_s = g.groupby(level=0, sort=False)['sum'].cumsum() - g['sum']
        idx = pd.MultiIndex.from_arrays([d[col], d['time']])
        n = past_n.reindex(idx).to_numpy(dtype=np.float32)
        s = past_s.reindex(idx).to_numpy(dtype=np.float32)
        out.extend((n, (s + 2.0) / (n + 6.0)))
        totals.append(d.groupby(col, sort=False)['label'].agg(['size', 'sum']))
    return np.column_stack(out).astype(np.float32), totals

def fixed_stats(rows, vi, totals, vid2author):
    keys = ([x[vi] for x in rows], [vid2author.get(x[vi], 'UNK') for x in rows])
    out = []
    for key, total in zip(keys, totals):
        n = pd.Series(key).map(total['size']).fillna(0).to_numpy(dtype=np.float32)
        s = pd.Series(key).map(total['sum']).fillna(0).to_numpy(dtype=np.float32)
        out.extend((n, (s + 2.0) / (n + 6.0)))
    return np.column_stack(out).astype(np.float32)

def session_features(rows, ui, ti, base=None):
    base = [] if base is None else base; nb = len(base)
    users = [x[0] for x in base] + [x[ui] for x in rows]
    times = [int(float(x[6])) for x in base] + [int(float(x[ti])) for x in rows]
    d = pd.DataFrame({'user': users, 'time': times, 'row': [-1] * nb + list(range(len(rows)))})
    g = d.groupby(['user', 'time'], sort=False).size().rename('size').reset_index()
    g = g.sort_values(['user', 'time'], kind='stable').reset_index(drop=True)
    g['gap'] = g.groupby('user', sort=False)['time'].diff().fillna(3_600_000_000)
    g['session'] = (g['gap'] > 1_800_000).groupby(g['user'], sort=False).cumsum()
    g['pos'] = g.groupby(['user', 'session'], sort=False)['size'].cumsum() - g['size'] + 1
    n10 = np.zeros(len(g), dtype=np.int64)
    for ix in g.groupby('user', sort=False).indices.values():
        t = g.loc[ix, 'time'].to_numpy(); size = g.loc[ix, 'size'].to_numpy()
        cs = np.r_[0, np.cumsum(size)]; left = np.searchsorted(t, t - 600_000, side='left')
        n10[ix] = cs[np.arange(len(ix))] - cs[left]
    g['n10'] = n10
    z = d.loc[d['row'] >= 0].merge(g[['user', 'time', 'pos', 'n10', 'gap']],
                                    on=['user', 'time'], how='left').sort_values('row')
    return z[['pos', 'n10', 'gap']].to_numpy(dtype=np.float32)

def context_features(rows, tabi, di, hi, stats, sessions):
    tab = np.asarray([float(x[tabi]) for x in rows], dtype=np.float32)
    dur = np.asarray([float(x[di]) for x in rows], dtype=np.float32)
    hm = np.asarray([float(x[hi]) for x in rows], dtype=np.float32)
    return np.column_stack((tab, dur, np.log1p(np.maximum(dur, 0)), dur == 0,
                            np.floor(hm / 100), stats, sessions)).astype(np.float32)

def tie_free(users, scores):
    d = pd.DataFrame({'user': users, 'score': scores, 'row': np.arange(len(scores))})
    d = d.sort_values(['user', 'score', 'row'], ascending=[True, False, True], kind='stable')
    rank = d.groupby('user', sort=False).cumcount().to_numpy()
    out = np.empty(len(scores), dtype=np.float64); out[d['row'].to_numpy()] = -rank
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True); ap.add_argument('--out-dir', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--score-extra', default=None)
    ap.add_argument('--k', type=int, default=16); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=300); ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--patience', type=int, default=4)
    a = ap.parse_args()
    smoke = int(os.environ.get('SMOKE_EPOCHS', '0') or 0)
    epochs = min(a.epochs, smoke) if smoke > 0 else a.epochs
    os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()
    rng = np.random.default_rng(a.seed)

    # ---- load ----
    vid2author = dict(read_rows(f'{a.data_dir}/video_features_basic.csv', ['video_id', 'author_id']))
    tr = read_rows(f'{a.data_dir}/train.csv',
                   ['user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'hourmin', 'time_ms'])
    va = read_rows(f'{a.data_dir}/valid.csv',
                   ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'long_view', 'hourmin', 'time_ms'])

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

    Str, totals = historical_stats(tr, ytr, vid2author)
    Sva = fixed_stats(va, 2, totals, vid2author)
    Ctr = context_features(tr, 2, 3, 5, Str, session_features(tr, 0, 6))
    Cva = context_features(va, 3, 4, 6, Sva, session_features(va, 1, 7, tr))
    Xtr = np.column_stack((Xtr, Ctr)).astype(np.float32)
    Xva = np.column_stack((Xva, Cva)).astype(np.float32)

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

    # ---- train LambdaRank and select the best 25-round validation checkpoint ----
    user_code = pd.factorize([x[0] for x in tr], sort=False)[0].astype(np.int32)
    counts = np.bincount(user_code); positives = np.bincount(user_code, weights=ytr)
    mixed = (positives > 0) & (positives < counts)
    rank_order = np.argsort(user_code, kind='stable')
    rank_order = rank_order[mixed[user_code[rank_order]]]
    groups = counts[mixed].astype(np.int32)
    Xrank, yrank = Xtr[rank_order], ytr[rank_order]
    names = FIELDS + ['tab_raw', 'duration_ms', 'log_duration', 'duration_zero', 'hour',
                      'video_count', 'video_rate', 'author_count', 'author_rate',
                      'session_position', 'impressions_10m', 'gap_ms']
    train_set = lgb.Dataset(Xrank, label=yrank, group=groups, feature_name=names,
                            categorical_feature=list(range(len(FIELDS))), free_raw_data=False)
    threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    params = {'objective': 'lambdarank', 'metric': 'None', 'learning_rate': 0.05,
              'num_leaves': 63, 'min_data_in_leaf': 100, 'feature_fraction': 0.8,
              'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambdarank_truncation_level': 10,
              'num_threads': threads, 'seed': a.seed, 'feature_fraction_seed': a.seed,
              'bagging_seed': a.seed, 'data_random_seed': a.seed, 'deterministic': True,
              'force_row_wise': True, 'verbosity': -1}
    best, best_round, history = -1.0, 1, []
    def checkpoint(env):
        nonlocal best, best_round
        ep = env.iteration + 1
        if ep % 25 != 0 and ep != epochs:
            return
        zva = tie_free(uva, env.model.predict(Xva, num_iteration=ep))
        ztr = env.model.predict(Xrank, num_iteration=ep)
        loss = float(np.mean(np.logaddexp(0.0, ztr) - yrank * ztr))
        r = evaluate(uva, yva, zva)
        history.append({'epoch': ep, 'train_loss': loss, 'val_gauc': r['GAUC'],
                        'val_ndcg5': r['nDCG@5'], 'val_primary': r['primary']})
        print(f"epoch {ep:2d} | loss {loss:.4f} | valid GAUC {r['GAUC']:.4f} nDCG@5 {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f}", flush=True)
        if r['primary'] > best + 1e-5:
            best, best_round = r['primary'], ep
    checkpoint.order = 20
    checkpoint.before_iteration = False
    m = lgb.train(params, train_set, num_boost_round=epochs, callbacks=[checkpoint])

    # ---- outputs ----
    sva = tie_free(uva, m.predict(Xva, num_iteration=best_round)); r = evaluate(uva, yva, sva)
    write_predictions(f'{a.out_dir}/predictions.csv', va, sva)
    with open(f'{a.out_dir}/metrics.json', 'w') as fh:
        json.dump({'gauc': r['GAUC'], 'ndcg5': r['nDCG@5'], 'primary': r['primary'],
                   'best_epoch': best_round, 'history': history,
                   'seed': a.seed, 'duration_s': time.time() - t0}, fh, indent=1)
    if a.score_extra:
        ex = read_rows(a.score_extra,
                       ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'hourmin', 'time_ms'])
        Sex = fixed_stats(ex, 2, totals, vid2author)
        Cex = context_features(ex, 3, 4, 5, Sex, session_features(ex, 1, 6, tr))
        Xex = np.column_stack((encode(ex, 1, 2, 3, 4), Cex)).astype(np.float32)
        sex = tie_free([x[1] for x in ex], m.predict(Xex, num_iteration=best_round))
        write_predictions(f'{a.out_dir}/predictions_extra.csv', ex, sex)
    print(f"done: valid primary {r['primary']:.4f} in {time.time() - t0:.0f}s")

if __name__ == '__main__':
    main()
