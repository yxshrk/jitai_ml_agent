"""Validation-only rank transforms for the overnight breadth campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import polish_stack


def within_group_ranks(keys: list[np.ndarray], scores: np.ndarray) -> np.ndarray:
    """Return stable ordinal ranks in [0, 1] within groups defined by keys."""
    if not keys or any(len(key) != len(scores) for key in keys):
        raise ValueError("rank keys must be nonempty and aligned with scores")
    order = np.lexsort(tuple(reversed(keys)))
    boundaries = np.zeros(len(scores), dtype=bool)
    if len(scores):
        boundaries[0] = True
        for key in keys:
            boundaries[1:] |= key[order][1:] != key[order][:-1]
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(scores)]
    result = np.zeros(len(scores), dtype=np.float64)
    for start, end in zip(starts, ends):
        indices = order[start:end]
        local_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        sorted_scores = scores[indices][local_order]
        tie_starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
        tie_ends = np.r_[tie_starts[1:], len(indices)]
        for tie_start, tie_end in zip(tie_starts, tie_ends):
            ranks[local_order[tie_start:tie_end]] = (tie_start + tie_end - 1) / 2.0
        if len(indices) > 1:
            ranks /= len(indices) - 1
        result[indices] = ranks
    return result


def item_stat(train_items: np.ndarray, labels: np.ndarray, valid_items: np.ndarray,
              variant: str) -> np.ndarray:
    if variant == "exposure":
        weights = np.ones(len(labels), dtype=np.float64)
    elif variant == "long-view":
        weights = labels.astype(np.float64)
    else:
        raise ValueError(f"unknown popularity variant: {variant}")
    unique, inverse = np.unique(train_items, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    positions = np.searchsorted(unique, valid_items)
    found = (positions < len(unique)) & (unique[np.minimum(positions, len(unique) - 1)] == valid_items)
    result = np.zeros(len(valid_items), dtype=np.float64)
    result[found] = totals[positions[found]]
    return result


def popularity(ds, scores: np.ndarray, variant: str, weight: float) -> np.ndarray:
    va, tr = ds["valid"], ds["train"]
    model_ranks = within_group_ranks([va["users"]], scores)
    raw_popularity = item_stat(tr["videos"], tr["y"], va["videos"], variant)
    popularity_ranks = within_group_ranks([va["users"]], raw_popularity)
    return (1.0 - weight) * model_ranks + weight * popularity_ranks


def tab_recombine(ds, scores: np.ndarray) -> np.ndarray:
    va, tr = ds["valid"], ds["train"]
    train_tabs, valid_tabs = tr["X"][:, 3], va["X"][:, 3]
    unique_tabs, inverse = np.unique(train_tabs, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=tr["y"])
    rates = positives / np.maximum(counts, 1)
    tab_order = np.argsort(np.argsort(rates, kind="stable"), kind="stable").astype(np.float64)
    positions = np.searchsorted(unique_tabs, valid_tabs)
    if np.any(positions >= len(unique_tabs)) or np.any(unique_tabs[positions] != valid_tabs):
        raise ValueError("validation contains a tab absent from training")
    local_ranks = within_group_ranks([va["users"], valid_tabs], scores)
    # Lexicographic score: every row in a higher-rate tab precedes every row in
    # a lower-rate tab, while model order is retained inside each (user, tab).
    return tab_order[positions] + local_ranks / 2.0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--method", choices=("popularity", "tab-recombine"), required=True)
    ap.add_argument("--variant", choices=("exposure", "long-view"))
    ap.add_argument("--weight", type=float)
    return ap


def main() -> None:
    args = parser().parse_args()
    ds = polish_stack.load_validation_only(args.data_dir)
    scores = np.load(args.scores)
    if len(scores) != len(ds["valid"]["y"]):
        raise ValueError("score array is not aligned with validation")
    if args.method == "popularity":
        if args.variant is None or args.weight not in (0.1, 0.2):
            raise ValueError("popularity needs --variant and --weight in {0.1, 0.2}")
        transformed = popularity(ds, scores, args.variant, args.weight)
        details = {"variant": args.variant, "weight": args.weight}
    else:
        if args.variant is not None or args.weight is not None:
            raise ValueError("tab-recombine does not accept popularity options")
        transformed = tab_recombine(ds, scores)
        details = {}
    va = ds["valid"]
    metrics = polish_stack.official_metrics(va["users"], va["y"], transformed)
    metrics.update({"method": args.method, "source_scores": str(args.scores), **details})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "scores.npy", transformed)
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)
        handle.write("\n")
    print("final:", json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
