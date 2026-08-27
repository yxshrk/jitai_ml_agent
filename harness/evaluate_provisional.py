"""Pinned provisional ranking metrics, swappable with the official evaluator."""

from __future__ import annotations

import argparse

import numpy as np


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUC as positive/negative pair accuracy, with ties worth one half."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0)))


def evaluate(user_ids, labels, scores) -> dict[str, float]:
    """Return positive-weighted GAUC, mean per-user nDCG@5, and their mean."""
    user_ids = np.asarray(user_ids)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if user_ids.ndim != 1 or labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("user_ids, labels, and scores must be one-dimensional")
    if not (len(user_ids) == len(labels) == len(scores)) or len(labels) == 0:
        raise ValueError("inputs must have equal, non-zero lengths")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")

    auc_total = 0.0
    auc_weight = 0
    ndcgs: list[float] = []
    for user in np.unique(user_ids):
        mask = user_ids == user
        user_labels = labels[mask].astype(int, copy=False)
        user_scores = scores[mask]
        positives = int(user_labels.sum())
        if 0 < positives < len(user_labels):
            auc_total += positives * _auc(user_labels, user_scores)
            auc_weight += positives

        # Stable sorting defines deterministic behavior when prediction scores tie.
        order = np.argsort(-user_scores, kind="stable")[:5]
        discount = np.log2(np.arange(2, len(order) + 2))
        dcg = float(np.sum((np.power(2.0, user_labels[order]) - 1.0) / discount))
        ideal_size = min(positives, 5)
        if ideal_size == 0:
            ndcgs.append(0.0)
        else:
            ideal_discount = np.log2(np.arange(2, ideal_size + 2))
            idcg = float(np.sum(1.0 / ideal_discount))
            ndcgs.append(dcg / idcg)

    gauc = auc_total / auc_weight if auc_weight else 0.0
    ndcg5 = float(np.mean(ndcgs))
    return {"gauc": float(gauc), "ndcg5": ndcg5, "primary": (float(gauc) + ndcg5) / 2.0}


def check(seed: int = 42) -> dict[str, float]:
    """Exercise the GAUC implementation on a sufficiently concentrated fixture."""
    rng = np.random.default_rng(seed)
    user_ids = np.repeat(np.arange(100), 200)
    labels = np.tile(np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)], 100)
    perfect = evaluate(user_ids, labels, labels)
    random = evaluate(user_ids, labels, rng.random(len(labels)))
    if perfect["gauc"] != 1.0:
        raise AssertionError(f"perfect-label GAUC was {perfect['gauc']}, expected 1.0")
    if abs(random["gauc"] - 0.5) > 0.05:
        raise AssertionError(f"random GAUC was {random['gauc']}, expected 0.5 +/- 0.05")
    return {"perfect_gauc": perfect["gauc"], "random_gauc": random["gauc"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.check:
        parser.error("this provisional CLI currently supports only --check")
    result = check(args.seed)
    print(f"check passed: perfect_gauc={result['perfect_gauc']:.6f}, random_gauc={result['random_gauc']:.6f}")


if __name__ == "__main__":
    main()
