"""Adversarial audit of the max_1k_c node_002 score (0.6524) — independent evaluator.

Implements the priority checks from the external review: recompute primary with a
from-scratch tie-aware implementation (NOT data.official.evaluate), test alternative
grouping arrays, alternative label columns, global-vs-grouped AUC, tie rates, tie-order
jitter sensitivity, and row alignment. Pure numpy; no sklearn needed.
Usage: python tools/audit_1k_result.py <run_node_dir> <val_npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def groups(user):
    order = np.argsort(user, kind="stable")
    u = np.asarray(user)[order]
    cuts = np.r_[0, 1 + np.flatnonzero(u[1:] != u[:-1]), len(u)]
    return order, cuts


def ref_gauc(y, s, user, tie_credit=0.5):
    order, cuts = groups(user)
    num = den = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        idx = order[lo:hi]
        yi = y[idx]
        p, n = s[idx][yi == 1], s[idx][yi == 0]
        if not len(p) or not len(n):
            continue
        wins = (p[:, None] > n[None, :]).sum()
        ties = (p[:, None] == n[None, :]).sum()
        auc = (wins + tie_credit * ties) / (len(p) * len(n))
        num += len(p) * auc
        den += len(p)
    return num / den


def ref_ndcg5(y, s, user, rng=None):
    order, cuts = groups(user)
    vals = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        idx = order[lo:hi]
        yi, si = y[idx].astype(float), s[idx].astype(float)
        if rng is not None:  # tie-neutralizing jitter, label-independent
            si = si + rng.uniform(-1e-12, 1e-12, len(si)) * (np.abs(si).max() + 1)
        if yi.sum() == 0:
            vals.append(0.0)
            continue
        k = min(5, len(idx))
        top = np.argsort(-si, kind="stable")[:k]
        dcg = ((2 ** yi[top] - 1) / np.log2(np.arange(2, k + 2))).sum()
        ideal = np.sort(yi)[::-1][:k]
        idcg = ((2 ** ideal - 1) / np.log2(np.arange(2, k + 2))).sum()
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals))


def primary(y, s, user, tie_credit=0.5, rng=None):
    return 0.5 * (ref_gauc(y, s, user, tie_credit) + ref_ndcg5(y, s, user, rng))


def main():
    node_dir, val_path = Path(sys.argv[1]), Path(sys.argv[2])
    vals = []
    with open(node_dir / "predictions.csv") as fh:
        next(fh)
        for line in fh:
            vals.append(float(line.rsplit(",", 1)[1]))
    s = np.asarray(vals, dtype=np.float64); del vals
    val = np.load(val_path, allow_pickle=False)
    y = val["y"].astype(int)
    user = val["user"]
    print(f"rows: csv={len(s)} npz={len(y)} match={len(s) == len(y)}")
    assert len(s) == len(y), "ROW COUNT MISMATCH — stop here"
    print(f"finite scores: {np.isfinite(s).all()}")

    p = primary(y, s, user)
    print(f"[1] independent recompute primary = {p:.6f} "
          f"(gauc {ref_gauc(y, s, user):.6f}, ndcg5 {ref_ndcg5(y, s, user):.6f})")

    # [3] alternative grouping arrays (X columns: user, video, author, tab, dur)
    X = val["X"]
    for i, name in enumerate(["user_field", "video", "author", "tab", "dur_bucket"]):
        print(f"[3] grouped by {name}: {primary(y, s, X[:, i]):.6f}")

    # [5] alternative label columns present in the npz
    for k in val.files:
        if k in ("y", "X", "user", "date", "hourmin"):
            continue
        arr = val[k]
        if arr.shape == y.shape and set(np.unique(arr)[:3]).issubset({0, 1, 0.0, 1.0}):
            print(f"[5] label={k}: {primary(arr.astype(int), s, user):.6f}")

    # [6] global AUC vs GAUC
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
    pos = ranks[y == 1]
    gauc_global = (pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * (len(s) - len(pos)))
    print(f"[6] global AUC {gauc_global:.6f} vs grouped GAUC {ref_gauc(y, s, user):.6f}")

    # [10] optimistic tie credit
    print(f"[10] gauc tie=0.5: {ref_gauc(y, s, user):.6f}  tie=1.0(optimistic): {ref_gauc(y, s, user, 1.0):.6f}")

    # [11] tie-order jitter sensitivity
    js = [primary(y, s, user, rng=np.random.default_rng(r)) for r in range(20)]
    print(f"[11] jitter spread over 20 reps: {max(js) - min(js):.6f}")

    # [17] score quantization
    print(f"[17] unique-score ratio: {len(np.unique(s)) / len(s):.4f}")

    # [21] per-day slices
    dates = val["date"]
    for d in np.unique(dates):
        m = dates == d
        print(f"[21] day {d}: {primary(y[m], s[m], user[m]):.5f} ({m.sum()} rows)")


if __name__ == "__main__":
    main()
