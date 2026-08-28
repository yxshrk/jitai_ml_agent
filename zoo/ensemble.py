"""Within-user rank-average validation predictions from trained model artifacts.

Example:
  uv run python zoo/ensemble.py --data-dir data/real_ws --out-dir /tmp/ens \
    --inputs /tmp/model_a/predictions.csv /tmp/model_b/predictions.csv

The inputs must describe the same validation rows. This script does not train or
touch test data; it writes the standard predictions.csv and metrics.json contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate as official_evaluate

BASELINE_PRIMARY = 0.6016


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42, help="accepted for contract compatibility")
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--weights", nargs="+", type=float)
    return ap


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"row_id", "user_id", "video_id", "score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is empty or lacks columns {sorted(required)}")
    return tuple(np.fromiter((cast(row[name]) for row in rows), dtype=dtype, count=len(rows))
                 for name, cast, dtype in (("row_id", int, np.int64),
                                            ("user_id", int, np.int64),
                                            ("video_id", int, np.int64),
                                            ("score", float, np.float64)))  # type: ignore[return-value]


def within_user_ranks(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Return per-user [0,1] ranks; continuous model scores make ties negligible."""
    result = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        indices = order[start:end]
        score_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[score_order] = np.arange(len(indices), dtype=np.float64)
        if len(indices) > 1:
            ranks /= len(indices) - 1
        result[indices] = ranks
    return result


def rank_average(users: np.ndarray, score_columns: list[np.ndarray],
                 weights: np.ndarray | None = None) -> np.ndarray:
    if not score_columns:
        raise ValueError("at least one score column is required")
    if weights is None:
        weights = np.ones(len(score_columns), dtype=np.float64)
    if len(weights) != len(score_columns) or np.any(weights < 0) or not weights.sum():
        raise ValueError("weights must be nonnegative, nonzero, and match inputs")
    ranks = np.column_stack([within_user_ranks(users, scores) for scores in score_columns])
    return np.average(ranks, axis=1, weights=weights)


def main() -> None:
    args = parser().parse_args()
    loaded = [read_predictions(path) for path in args.inputs]
    row_ids, users, videos, _ = loaded[0]
    for path, (other_rows, other_users, _other_videos, _scores) in zip(args.inputs[1:], loaded[1:]):
        if not np.array_equal(row_ids, other_rows) or not np.array_equal(users, other_users):
            raise ValueError(f"row/user alignment mismatch in {path}")
    weights = None if args.weights is None else np.asarray(args.weights, dtype=np.float64)
    scores = rank_average(users, [item[3] for item in loaded], weights)

    data_dir = ROOT / "data" / "real_ws" if args.data_dir == "real" else Path(args.data_dir)
    with np.load(data_dir / "val.npz") as valid:
        labels = valid["y"].astype(int)
        official_users = valid["user"]
    if len(labels) != len(scores) or not np.array_equal(users, official_users):
        raise ValueError("prediction rows do not align with the requested validation split")
    raw = official_evaluate(users.tolist(), labels.tolist(), scores.tolist())
    metrics = {"gauc": float(raw["GAUC"]), "ndcg5": float(raw["nDCG@5"]),
               "primary": float(raw["primary"]),
               "delta": float(raw["primary"] - BASELINE_PRIMARY),
               "members": [str(path) for path in args.inputs],
               "weights": (weights.tolist() if weights is not None else
                           [1.0] * len(args.inputs))}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        writer.writerows((int(row), int(user), int(video), f"{score:.10f}")
                         for row, user, video, score in zip(row_ids, users, videos, scores))
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True)
        handle.write("\n")
    print("final:", json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
