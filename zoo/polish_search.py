"""Optuna TPE driver for the frozen-stack polish search.

Run with ``uv run --with optuna python zoo/polish_search.py`` so repository
dependency manifests remain untouched as required by the campaign safety rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import polish_stack


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--work-dir", default="/tmp/polish-campaign")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--study-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience-halves", type=int, default=5)
    ap.add_argument("--max-runtime", type=int, default=350)
    return ap


def trial_namespace(base: argparse.Namespace, trial: optuna.Trial) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=base.data_dir,
        out_dir=str(Path(base.work_dir) / "optuna" / f"trial_{trial.number:03d}"),
        seed=42,
        epochs=base.epochs,
        batch_size=trial.suggest_categorical("batch_size", [4096, 8192, 16384]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        step_decay_factor=trial.suggest_float("step_decay_factor", 0.3, 0.7),
        decay_every=trial.suggest_categorical("decay_every", [0.5, 1.0, 1.5]),
        decay_start_epoch=0.0,
        dropout=trial.suggest_float("dropout", 0.15, 0.4),
        embedding_dropout=0.1,
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 3e-3, log=True),
        k=trial.suggest_categorical("k", [8, 12, 16, 24]),
        recency_half_life=trial.suggest_float("recency_half_life", 4.0, 12.0),
        bpr_weight=trial.suggest_float("bpr_weight", 0.4, 0.6),
        patience_halves=base.patience_halves,
        max_runtime=base.max_runtime,
        subsample=None,
    )


def main() -> None:
    args = parser().parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    dataset = polish_stack.load_validation_only(args.data_dir)
    sampler = optuna.samplers.TPESampler(seed=args.study_seed, n_startup_trials=8,
                                         multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4,
                                         interval_steps=1)
    storage = f"sqlite:///{work / 'optuna.db'}"
    study = optuna.create_study(study_name="frozen-stack-polish", direction="maximize",
                                sampler=sampler, pruner=pruner, storage=storage,
                                load_if_exists=True)

    def objective(trial: optuna.Trial) -> float:
        run_args = trial_namespace(args, trial)

        def checkpoint(step: int, primary: float) -> None:
            trial.report(primary, step)
            if trial.should_prune():
                raise optuna.TrialPruned(f"median prune at half-step {step}")

        metrics = polish_stack.train_and_report(dataset, run_args, checkpoint)
        trial.set_user_attr("best_epoch", metrics["best_epoch"])
        trial.set_user_attr("runtime_s", metrics["runtime_s"])
        return float(metrics["primary"])

    remaining = max(0, args.trials - len(study.trials))
    study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                   show_progress_bar=False)
    rows = [{"number": trial.number, "state": trial.state.name, "value": trial.value,
             "params": trial.params, "user_attrs": trial.user_attrs}
            for trial in study.trials]
    with (work / "optuna_trials.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
        fh.write("\n")
    completed = sorted((trial for trial in study.trials if trial.value is not None),
                       key=lambda trial: trial.value, reverse=True)
    print(json.dumps([{"number": trial.number, "value": trial.value,
                       "params": trial.params} for trial in completed[:3]], indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
