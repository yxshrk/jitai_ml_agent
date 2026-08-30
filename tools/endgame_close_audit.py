"""Exact-tie sensitivity audit of the designated Pure close (bc07 node_006).

Per-user rank averaging of 3 discrete member rankings over ~4-item slates can
produce exact aggregate ties; tie resolution then silently depends on row order.
Measures (on validation, labels available): tie prevalence and the distribution
of the official primary under random tie resolutions.
Usage: uv run python tools/endgame_close_audit.py \
    logs/run_bigclock_07/node_006/predictions.csv data/real_ws/val.npz
"""
import csv
import sys

import numpy as np

sys.path.insert(0, '.')
from data.official.evaluate import evaluate


def main():
    pred_csv, val_npz = sys.argv[1], sys.argv[2]
    d = np.load(val_npz)
    y, user = d['y'].astype(int), d['user']
    scores = np.empty(len(y))
    with open(pred_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            scores[int(row[0])] = float(row[3])

    base = evaluate(user, y, scores)
    print(f"as-recorded primary: {base['primary']:.6f} "
          f"(gauc {base['GAUC']:.6f} ndcg5 {base['nDCG@5']:.6f})")

    # tie prevalence per user slate
    order = np.argsort(user, kind='stable')
    su, ss = user[order], scores[order]
    cuts = np.r_[0, 1 + np.flatnonzero(su[1:] != su[:-1]), len(su)]
    users_with_tie = rows_in_ties = slates = 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        slates += 1
        vals, counts = np.unique(ss[a:b], return_counts=True)
        tied = counts[counts > 1].sum()
        if tied:
            users_with_tie += 1
            rows_in_ties += int(tied)
    print(f"slates: {slates}; with >=1 exact tie: {users_with_tie} "
          f"({users_with_tie/slates:.1%}); rows inside tie groups: {rows_in_ties} "
          f"({rows_in_ties/len(y):.1%})")

    # primary under random tie resolutions: jitter far below the score quantum
    quantum = np.min(np.diff(np.unique(scores))) if len(np.unique(scores)) > 1 else 1.0
    rng = np.random.default_rng(0)
    prims, gaucs, ndcgs = [], [], []
    for _ in range(200):
        m = evaluate(user, y, scores + rng.uniform(-quantum/1e6, quantum/1e6, len(scores)))
        prims.append(m['primary']); gaucs.append(m['GAUC']); ndcgs.append(m['nDCG@5'])
    for name, arr in (('primary', prims), ('gauc', gaucs), ('ndcg5', ndcgs)):
        a = np.asarray(arr)
        print(f"{name}: mean {a.mean():.6f} sd {a.std():.6f} "
              f"range [{a.min():.6f}, {a.max():.6f}]")
    print(f"as-recorded minus tie-random mean (order dependence): "
          f"{base['primary']-np.mean(prims):+.6f}")


if __name__ == '__main__':
    main()
