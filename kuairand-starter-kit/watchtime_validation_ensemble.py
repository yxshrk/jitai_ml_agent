"""Evaluate a fixed ensemble of validation-only censor-aware watch-time members.

This utility reads saved validation logits only.  It never reads the hidden
test rows or labels, and exists to make the selected ensemble reproducible.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate
from sequence_deepfm import load_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--scores_dir", required=True)
    parser.add_argument("--seeds", default="11,5,6,7")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_name", default="watchtime_validation_ensemble")
    return parser.parse_args()


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct seeds")
    score_paths = [Path(args.scores_dir) / f"seed_{seed}_scores.npy" for seed in seeds]
    missing = [str(path) for path in score_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing member score files: {missing}")
    scores = [np.load(path) for path in score_paths]
    if len({score.shape for score in scores}) != 1:
        raise ValueError("All ensemble members must have identical score shapes")
    rows = load_rows(args.data_dir)["valid"]
    users = [row["user_id"] for row in rows]
    labels = np.asarray([row["label"] for row in rows])
    if len(users) != len(scores[0]):
        raise ValueError("Score vector length does not match the fixed validation split")
    metrics = {name: float(value) for name, value in evaluate(users, labels, np.mean(scores, axis=0)).items()}
    report = {
        "phase": args.experiment_name,
        "selection_split": "validation",
        "test_data_used": False,
        "aggregation": "mean_logit",
        "seeds": seeds,
        "member_score_files": [str(path) for path in score_paths],
        "metrics": metrics,
        "note": "Seeds were selected from validation experiments; confirm the fixed recipe on an earlier chronological holdout before final submission.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
