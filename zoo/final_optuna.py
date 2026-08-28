# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "optuna",
# ]
# ///
"""Optuna TPE search over the confirmed specialist lever.

This script intentionally uses PEP 723 script-local dependencies so ``uv add``
does not modify the project files.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

try:
    import optuna
except ModuleNotFoundError:  # offline fallback after uv records the requested dependency
    optuna = None

import math
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
        candidate.epochs = min(args.epochs, 4)
        candidate.patience = min(args.patience, 2)
        candidate.out_dir = str(Path(args.out_dir) / f"trial_{trial.number:02d}")
        result = run(candidate)
        trial_records.append({"trial": trial.number, "primary": result["primary"],
                              "params": dict(trial.params), "runtime_s": result["runtime_s"]})
        return float(result["primary"])

    if optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=args.trials)
        best_params, best_value = study.best_params, study.best_value
        engine = "optuna.samplers.TPESampler"
    else:
        # Network-disabled environments cannot resolve the script dependency. This
        # deterministic fallback preserves TPE's core pattern: random startup,
        # split observations into good/bad densities, then draw candidates near
        # the good set and maximize a Parzen l(x)/g(x) density ratio.
        rng = np.random.default_rng(args.seed)
        observations: list[tuple[float, dict]] = []
        space = {
            "lr": ("log", 3e-4, 2e-3), "weight_decay": ("log", 1e-7, 1e-3),
            "dropout": ("float", 0.0, 0.4), "k": ("cat", [8, 12, 16, 24]),
            "bpr_weight": ("float", 0.3, 0.8),
            "batch_size": ("cat", [4096, 8192, 16384]),
        }

        def density(x, values, width):
            if not values:
                return 1.0
            z = (x - np.asarray(values)) / max(width, 1e-9)
            return float(np.exp(-0.5 * z * z).mean() / max(width, 1e-9)) + 1e-12

        class Trial:
            def __init__(self, number, proposed): self.number, self.params = number, proposed
            def suggest_float(self, name, low, high, log=False): return self.params[name]
            def suggest_categorical(self, name, values): return self.params[name]

        for number in range(args.trials):
            if number < 8:
                proposed = {}
                for name, spec in space.items():
                    if spec[0] == "cat": proposed[name] = rng.choice(spec[1]).item()
                    elif spec[0] == "log": proposed[name] = float(math.exp(rng.uniform(math.log(spec[1]), math.log(spec[2]))))
                    else: proposed[name] = float(rng.uniform(spec[1], spec[2]))
            else:
                ordered = sorted(observations, reverse=True)
                cut = max(2, math.ceil(0.25 * len(ordered)))
                good, bad = [x[1] for x in ordered[:cut]], [x[1] for x in ordered[cut:]]
                proposed = {}
                for name, spec in space.items():
                    if spec[0] == "cat":
                        choices = spec[1]
                        ratios = [(sum(p[name] == c for p in good) + 1) /
                                  (sum(p[name] == c for p in bad) + 1) for c in choices]
                        proposed[name] = choices[int(np.argmax(ratios))]
                    else:
                        lo, hi = spec[1], spec[2]
                        transform = math.log if spec[0] == "log" else float
                        inverse = math.exp if spec[0] == "log" else float
                        gv = [transform(p[name]) for p in good]; bv = [transform(p[name]) for p in bad]
                        tlo, thi = transform(lo), transform(hi); width = (thi - tlo) / math.sqrt(len(observations))
                        candidates = np.clip(rng.normal(rng.choice(gv), width, 32), tlo, thi)
                        ratio = [density(x, gv, width) / density(x, bv, width) for x in candidates]
                        proposed[name] = float(inverse(float(candidates[int(np.argmax(ratio))])))
            value = objective(Trial(number, proposed))
            observations.append((value, proposed))
        best_value, best_params = max(observations, key=lambda item: item[0])
        engine = "offline Parzen l(x)/g(x) fallback (Optuna unavailable after uv add)"
    best = copy.copy(args)
    for key, value in best_params.items():
        setattr(best, key, value)
    best.out_dir = args.out_dir
    best.patience = max(args.patience, 4)
    final = run(best)
    metrics_path = Path(args.out_dir) / "metrics.json"
    final["optuna_best_params"] = best_params
    final["optuna_best_value"] = best_value
    final["search_engine"] = engine
    final["optuna_trials"] = trial_records
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(final, handle, sort_keys=True)
        handle.write("\n")
    with (Path(args.out_dir) / "study.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_params": best_params, "best_value": best_value,
                   "search_engine": engine, "trials": trial_records}, handle, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
