"""Best confirmed polish configuration and five-seed rank-average utility.

Single member::
    uv run python zoo/polish_best.py --out-dir /tmp/member-42 --seed 42

After running seeds 42--46, rank-average them::
    uv run python zoo/polish_best.py --out-dir /tmp/ensemble \
      --rank-average-scores /tmp/member-{42,43,44,45,46}/scores.npy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import polish_stack

BEST_DEFAULTS = {
    "epochs": 8,
    "patience_halves": 5,
    "batch_size": 8192,
    "bpr_weight": 0.4453160212036508,
    "decay_every": 1.0,
    "dropout": 0.18199037655935982,
    "k": 16,
    "lr": 0.0007003874872132884,
    "recency_half_life": 11.424348428709624,
    "step_decay_factor": 0.6103216481366316,
    "weight_decay": 0.00031442255073239905,
}


def parser():
    ap = polish_stack.parser(__doc__)
    ap.set_defaults(**BEST_DEFAULTS)
    ap.add_argument("--rank-average-scores", nargs="+", default=None,
                    help="score .npy files to ensemble instead of training")
    return ap


def rank_average(score_arrays: list[np.ndarray]) -> np.ndarray:
    if len(score_arrays) < 2:
        raise ValueError("rank averaging requires at least two members")
    if len({len(scores) for scores in score_arrays}) != 1:
        raise ValueError("ensemble score arrays have different lengths")
    ranks = []
    for scores in score_arrays:
        order = np.argsort(scores, kind="stable")
        rank = np.empty(len(scores), dtype=np.float64)
        rank[order] = np.arange(len(scores), dtype=np.float64)
        ranks.append(rank / max(1, len(scores) - 1))
    return np.mean(ranks, axis=0)


def main() -> None:
    args = parser().parse_args()
    if args.rank_average_scores is None:
        polish_stack.run(args)
        return
    dataset = polish_stack.load_validation_only(args.data_dir)
    scores = rank_average([np.load(path) for path in args.rank_average_scores])
    valid = dataset["valid"]
    metrics = polish_stack.official_metrics(valid["users"], valid["y"], scores)
    metrics.update({"members": args.rank_average_scores, "method": "global rank average"})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", scores)
    with (out / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, sort_keys=True)
        fh.write("\n")
    print("ensemble:", json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
