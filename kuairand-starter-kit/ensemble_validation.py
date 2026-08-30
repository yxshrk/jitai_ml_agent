"""Train and evaluate a validation-only ensemble of temporal/context FMs.

This is a reproducible model-selection utility, not a hidden-test evaluator. It
trains each seed on the fixed training split, selects each checkpoint on the
public validation split, and averages only the resulting validation logits.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from evaluate import evaluate
from temporal_fm import run as run_temporal_fm


def parse_seed_list(text):
    seeds = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not seeds:
        raise ValueError("Provide at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--run_dir", required=True, help="New directory for per-seed JSONL logs and summary.json")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--extra_features", default="hour,weekday,is_rand")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--aggregation",
        choices=("mean_logit", "per_user_rank_average"),
        default="mean_logit",
        help="Combine validation-selected member predictions without accessing test data.",
    )
    return parser.parse_args()


def per_user_rank_average(scores_by_seed, users):
    """Average normalized ranks within each user's evaluation slate.

    The official metric orders impressions only within a user, so this preserves
    each member's scale-free ranking rather than allowing logit calibration to
    dominate an ensemble.  It is solely a validation selection operation.
    """
    groups = defaultdict(list)
    for index, user in enumerate(users):
        groups[user].append(index)
    member_ranks = []
    for scores in scores_by_seed:
        ranks = np.empty(len(scores), dtype=np.float32)
        for indices in groups.values():
            slate_scores = scores[indices]
            order = np.argsort(-slate_scores, kind="mergesort")
            if len(indices) == 1:
                ranks[indices[0]] = 0.5
            else:
                ranks[np.asarray(indices)[order]] = np.linspace(1.0, 0.0, len(indices), dtype=np.float32)
        member_ranks.append(ranks)
    return np.mean(member_ranks, axis=0)


def main(args):
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    seeds = parse_seed_list(args.seeds)
    scores_by_seed = []
    records = []
    valid_users = valid_labels = None

    for seed in seeds:
        seed_args = argparse.Namespace(
            data_dir=args.data_dir,
            extra_features=args.extra_features,
            seed=seed,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            run_log=str(run_dir / f"seed_{seed}.jsonl"),
            validation_scores_out=None,
        )
        record, scores, users, labels = run_temporal_fm(seed_args)
        if valid_users is None:
            valid_users, valid_labels = users, labels
        elif users != valid_users or not np.array_equal(labels, valid_labels):
            raise RuntimeError("Validation split changed between seeds")
        records.append(record)
        scores_by_seed.append(scores)

    if args.aggregation == "mean_logit":
        ensemble_scores = np.mean(scores_by_seed, axis=0)
    else:
        ensemble_scores = per_user_rank_average(scores_by_seed, valid_users)
    ensemble_metrics = {
        name: float(value)
        for name, value in evaluate(valid_users, valid_labels, ensemble_scores).items()
    }
    summary = {
        "selection_split": "validation",
        "test_data_used": False,
        "seeds": seeds,
        "aggregation": args.aggregation,
        "extra_features": [feature for feature in args.extra_features.split(",") if feature],
        "member_best_primary_mean": float(np.mean([record["metrics"]["primary"] for record in records])),
        "ensemble_metrics": ensemble_metrics,
        "members": records,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("\nValidation ensemble")
    print(json.dumps(ensemble_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    started = time.time()
    main(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
