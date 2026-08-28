"""Reproduce the best sequence cell or its per-user-rank frozen ensemble.

Best sequence model::

    uv run python zoo/seq_best.py --out-dir /tmp/seq-best --seed 42

Validation diversity ensemble (after the command above)::

    uv run python zoo/seq_best.py --out-dir /tmp/seq-ensemble \
      --seq-scores /tmp/seq-best/scores.npy \
      --frozen-predictions logs/run_rehearsal5/node_001/predictions.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import seq_sasrec
from zoo.seq_data import assert_no_leakage, load_or_prepare

BEST_DEFAULTS = {
    "encoder": "transformer",
    "outcome_marks": False,
    "max_history": 50,
    "k": 32,
    "blocks": 1,
    "epochs": 4,
    "batch_size": 4096,
    "predict_batch_size": 8192,
    "lr": 1e-3,
    "step_decay_factor": 0.5,
    "dropout": 0.25,
    "weight_decay": 1e-3,
    "recency_half_life": 7.0,
    "bpr_weight": 0.5,
    "patience_halves": 4,
    "max_runtime": 470,
}


def parser():
    ap = seq_sasrec.parser(__doc__)
    ap.set_defaults(**BEST_DEFAULTS)
    ap.add_argument("--seq-scores", default=None,
                    help="best sequence scores.npy; enables ensemble mode")
    ap.add_argument("--frozen-predictions", default=None,
                    help="frozen scores.npy or row_id/score predictions.csv")
    return ap


def _read_scores(path: str, expected: int) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        result = np.asarray(np.load(source), dtype=np.float64)
    else:
        result = np.empty(expected, dtype=np.float64)
        seen = np.zeros(expected, dtype=bool)
        with source.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row_id = int(row["row_id"])
                if not 0 <= row_id < expected or seen[row_id]:
                    raise ValueError(f"invalid or duplicate row_id {row_id}")
                result[row_id], seen[row_id] = float(row["score"]), True
        if not seen.all():
            raise ValueError(f"{source} does not cover all {expected} validation rows")
    if len(result) != expected:
        raise ValueError(f"{source} has {len(result)} scores; expected {expected}")
    return result


def ensemble(args) -> dict:
    started = time.time()
    data = load_or_prepare(args.data_dir, args.max_history, args.cache)
    assert_no_leakage(data)
    users, labels = data["valid_user"], data["valid_label"]
    sequence = _read_scores(args.seq_scores, len(users))
    frozen = _read_scores(args.frozen_predictions, len(users))
    scores = seq_sasrec.rank_average(users, sequence, frozen)
    metrics = seq_sasrec.official_metrics(users, labels, scores)
    metrics.update({
        "runtime_s": round(time.time() - started, 1),
        "method": "50/50 per-user ordinal-rank average",
        "members": [args.seq_scores, args.frozen_predictions],
        "segments": seq_sasrec.history_segments(
            users, labels, scores, data["valid_history_count"]),
        "leakage_check": "passed",
    })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("ensemble:", json.dumps(metrics, sort_keys=True))
    return metrics


def main() -> None:
    args = parser().parse_args()
    if (args.seq_scores is None) != (args.frozen_predictions is None):
        raise SystemExit("ensemble mode requires both --seq-scores and --frozen-predictions")
    if args.seq_scores is not None:
        ensemble(args)
    else:
        seq_sasrec.run(args)


if __name__ == "__main__":
    main()
