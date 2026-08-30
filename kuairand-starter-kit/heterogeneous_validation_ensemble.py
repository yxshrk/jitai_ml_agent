"""Validation-only blend of saved contextual-FM and contextual-BPR scores."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate
from temporal_fm import encode, load_rows


BLEND_WEIGHTS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--fm_scores_dir", required=True)
    parser.add_argument("--bpr_scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fixed_bpr_weight",
        type=float,
        default=None,
        help="Evaluate exactly one preselected blend weight for a confirmation run.",
    )
    return parser.parse_args()


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ensemble report: {output}")
    fm_paths = sorted(Path(args.fm_scores_dir).glob("seed_*_scores.npy"))
    if len(fm_paths) < 2:
        raise ValueError("Need at least two saved contextual-FM score vectors")
    fm_scores = [np.load(path) for path in fm_paths]
    bpr_scores = np.load(args.bpr_scores)
    if any(scores.shape != bpr_scores.shape for scores in fm_scores):
        raise ValueError("All score vectors must have identical shape")
    rows = load_rows(args.data_dir)
    encoded, _, _ = encode(rows, ("hour", "weekday", "is_rand"))
    _, valid_labels, valid_users = encoded["valid"]
    if len(valid_labels) != len(bpr_scores):
        raise ValueError("Saved predictions do not match the fixed validation split")
    fm_mean = np.mean(fm_scores, axis=0)
    candidates = []
    weights = (args.fixed_bpr_weight,) if args.fixed_bpr_weight is not None else BLEND_WEIGHTS
    if any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("BPR blend weights must be in [0, 1]")
    for bpr_weight in weights:
        scores = (1.0 - bpr_weight) * fm_mean + bpr_weight * bpr_scores
        metrics = {key: float(value) for key, value in evaluate(valid_users, valid_labels, scores).items()}
        candidates.append({"bpr_weight": bpr_weight, "metrics": metrics})
    best = max(candidates, key=lambda candidate: candidate["metrics"]["primary"])
    report = {
        "phase": "heterogeneous_contextual_validation_ensemble",
        "selection_split": "validation",
        "test_data_used": False,
        "manual_interventions": 0,
        "fm_members": [str(path) for path in fm_paths],
        "bpr_member": str(args.bpr_scores),
        "predeclared_bpr_weights": list(weights),
        "candidates": candidates,
        "selected": best,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(parse_args())
