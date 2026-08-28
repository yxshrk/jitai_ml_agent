# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
# ]
# ///
"""Optuna TPE search over the confirmed specialist lever.

This script intentionally uses PEP 723 script-local dependencies so ``uv add``
does not modify the project files. The best trial is confirmed at seeds 42--44.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import optuna
import numpy as np

from final_core import parser, run


def main() -> None:
    ap = parser(__doc__, "specialists")
    ap.add_argument("--trials", type=int, default=25)
    args = ap.parse_args()
    trial_records: list[dict] = []

    def objective(trial) -> float:
        candidate = copy.copy(args)
        candidate.lr = trial.suggest_float("lr", 3e-4, 2e-3, log=True)
        candidate.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        candidate.dropout = trial.suggest_float("dropout", 0.0, 0.4)
        candidate.k = trial.suggest_categorical("k", [8, 12, 16, 24])
        candidate.bpr_weight = trial.suggest_float("bpr_weight", 0.3, 0.8)
        candidate.batch_size = trial.suggest_categorical("batch_size", [4096, 8192, 16384])
        use_recency = trial.suggest_categorical("recency_weighting", [False, True])
        candidate.recency_half_life = (
            trial.suggest_categorical("recency_half_life", [3.0, 7.0, 14.0])
            if use_recency else None
        )
        candidate.epochs = min(args.epochs, 4)
        candidate.patience = min(args.patience, 2)
        candidate.out_dir = str(Path(args.out_dir) / f"trial_{trial.number:02d}")
        result = run(candidate)
        trial_records.append({"trial": trial.number, "primary": result["primary"],
                              "params": dict(trial.params), "runtime_s": result["runtime_s"]})
        return float(result["primary"])

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)
    best_params, best_value = study.best_params, study.best_value
    engine = "optuna.samplers.TPESampler"
    best = copy.copy(args)
    for key, value in best_params.items():
        if key != "recency_weighting":
            setattr(best, key, value)
    if not best_params.get("recency_weighting", False):
        best.recency_half_life = None
    best.patience = max(args.patience, 4)
    confirmations = []
    for seed in (42, 43, 44):
        confirmed = copy.copy(best)
        confirmed.seed = seed
        confirmed.out_dir = str(Path(args.out_dir) / f"confirm_seed_{seed}")
        confirmations.append(run(confirmed))
    primary = np.asarray([item["primary"] for item in confirmations])
    summary = {
        "optuna_best_params": best_params,
        "optuna_best_value": best_value,
        "search_engine": engine,
        "optuna_trials": trial_records,
        "confirmations": confirmations,
        "confirmation_primary_mean": float(primary.mean()),
        "confirmation_primary_population_std": float(primary.std()),
        "delta_vs_0.6016": float(primary.mean() - 0.6016),
        "confirmed_win": bool(primary.mean() - 0.6016 >= 0.002),
    }
    metrics_path = Path(args.out_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True)
        handle.write("\n")
    with (Path(args.out_dir) / "study.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_params": best_params, "best_value": best_value,
                   "search_engine": engine, "trials": trial_records}, handle, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
