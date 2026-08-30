"""Bounded, validation-only DeepFM configuration search.

The run uses a fixed one-hour wall-clock budget. It records every probe, avoids
test data, and reserves the final part of the budget for fresh-seed checks of
the best short-run configurations.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 2.00


BASE = {
    "history_length": 8, "embedding_dim": 16, "hidden_dim": 64,
    "dropout": 0.10, "learning_rate": 0.001, "weight_decay": 1e-6,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wall_clock_seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--max_probes", type=int, default=48)
    parser.add_argument("--api_guided", action="store_true")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max_api_spend_usd", type=float, default=25.0)
    parser.add_argument("--max_api_calls", type=int, default=8)
    return parser.parse_args()


def load_project_key():
    """Load only the ignored project-local API key at runtime."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ.setdefault("OPENAI_API_KEY", value.strip())
            return


class Director:
    """The LLM chooses among a finite safe menu; it never runs generated code."""

    def __init__(self, args):
        self.args = args
        self.calls = self.input_tokens = self.output_tokens = 0
        self.decisions = []
        self.client = None
        if args.api_guided:
            load_project_key()
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("--api_guided requires OPENAI_API_KEY in .env")
            from openai import OpenAI
            self.client = OpenAI()

    @property
    def estimated_spend(self):
        return (self.input_tokens * INPUT_PRICE_PER_MILLION + self.output_tokens * OUTPUT_PRICE_PER_MILLION) / 1_000_000

    def choose(self, choice_type, available, observed, count):
        """Select IDs through Structured Outputs, falling back deterministically.

        The response schema and local validation deliberately limit the API to
        choosing experiment IDs.  It cannot introduce code, features, data, or
        a configuration outside the locally generated menu.
        """
        fallback = [item["index"] for item in sorted(
            available,
            key=lambda item: item.get("best", {}).get("metrics", {}).get("primary", -np.inf),
            reverse=True,
        )[:count]]
        if not fallback:
            fallback = [item["index"] for item in available[:count]]
        # Conservative upper bound for one call: 10k input + 500 output tokens.
        projected = self.estimated_spend + (10_000 * INPUT_PRICE_PER_MILLION + 500 * OUTPUT_PRICE_PER_MILLION) / 1_000_000
        if self.client is None or self.calls >= self.args.max_api_calls or projected > self.args.max_api_spend_usd:
            return fallback
        menu = {item["index"]: item["config"] for item in available}
        result_rows = [
            {"id": item["index"], "primary": item["best"]["metrics"]["primary"],
             "epoch": item["best"]["epoch"], "train_loss": item["best"]["train_loss"]}
            for item in observed if item.get("ok")
        ]
        schema = {
            "type": "json_schema", "name": "deepfm_candidate_selection", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "candidate_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["candidate_ids"], "additionalProperties": False,
            },
        }
        response = self.client.responses.create(
            model=self.args.model,
            reasoning={"effort": "low"}, max_output_tokens=500,
            text={"format": schema},
            input=(
                "You are a conservative ML experiment director. Choose at most " + str(count) +
                " IDs from the supplied finite DeepFM menu for the next validation-only experiment batch. "
                "Prefer diverse, modest settings and use observed validation results only as noisy evidence. "
                "You must not request test data, feature changes, code changes, or values outside this menu.\n\n" +
                json.dumps({"base": BASE, "available_candidates": menu, "observed_results": result_rows})
            ),
        )
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.calls += 1
        try:
            payload = json.loads(response.output_text)
            selected = [int(value) for value in payload["candidate_ids"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            payload, selected = {"fallback": True}, fallback
        allowed = set(menu)
        selected = list(dict.fromkeys(selected))
        if not selected or len(selected) > count or any(value not in allowed for value in selected):
            selected = fallback
        self.decisions.append({"type": choice_type, "response": payload,
                               "raw_output": response.output_text,
                               "input_tokens": getattr(usage, "input_tokens", None),
                               "output_tokens": getattr(usage, "output_tokens", None)})
        return selected

    def finalists(self, successful):
        return self.choose("finalist_selection", successful, successful, 3)


def candidate_configs(rng, maximum):
    configs, fingerprints = [{"index": 0, "config": BASE}], {tuple(BASE.items())}
    while len(configs) < maximum:
        config = {
            "history_length": int(rng.choice((0, 4, 8, 12, 16))),
            "embedding_dim": int(rng.choice((8, 12, 16, 20, 24))),
            "hidden_dim": int(rng.choice((32, 48, 64, 96, 128))),
            "dropout": float(rng.choice((0.0, 0.05, 0.10, 0.15, 0.20))),
            "learning_rate": float(rng.choice((0.0005, 0.0007, 0.00085, 0.001, 0.00115))),
            "weight_decay": float(rng.choice((0.0, 1e-7, 1e-6, 3e-6, 1e-5))),
        }
        fingerprint = tuple(config.items())
        if fingerprint not in fingerprints:
            configs.append({"index": len(configs), "config": config})
            fingerprints.add(fingerprint)
    return configs


def run_trial(args, run_dir, config, seed, stage, index, epochs, patience, deadline):
    trial_dir = run_dir / f"{stage}_{index:02d}_seed{seed}"
    log_path = trial_dir / "result.json"
    trial_dir.mkdir(parents=True)
    command = [
        sys.executable, "-u", "sequence_deepfm.py", "--data_dir", args.data_dir,
        "--seed", str(seed), "--epochs", str(epochs), "--patience", str(patience),
        "--batch_size", "32768", "--run_log", str(log_path),
    ]
    for key, value in config.items():
        command.extend((f"--{key}", str(value)))
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    timeout = max(1, int(deadline - time.monotonic()))
    started = time.monotonic()
    result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True, env=env, timeout=timeout)
    if result.returncode != 0:
        return {"stage": stage, "index": index, "seed": seed, "config": config, "ok": False,
                "duration_seconds": time.monotonic() - started, "error": result.stderr[-2000:]}
    payload = json.loads(log_path.read_text())
    return {"stage": stage, "index": index, "seed": seed, "config": config, "ok": True,
            "duration_seconds": time.monotonic() - started, "best": payload["best"]}


def main(args):
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    started, deadline = time.monotonic(), time.monotonic() + args.wall_clock_seconds
    rng = np.random.default_rng(args.seed)
    director = Director(args)
    results = []
    # Reserve the final 35% for independent full-length confirmations.
    probe_deadline = started + int(args.wall_clock_seconds * 0.65)
    configs = candidate_configs(rng, args.max_probes)
    remaining = configs.copy()
    # The director sees each completed batch before choosing the next one.
    while remaining and time.monotonic() < probe_deadline:
        candidate_ids = director.choose("probe_batch", remaining, results, min(6, len(remaining)))
        selected = {candidate_id for candidate_id in candidate_ids}
        batch = [item for item in remaining if item["index"] in selected]
        if not batch:
            break
        remaining = [item for item in remaining if item["index"] not in selected]
        for candidate in batch:
            if time.monotonic() >= probe_deadline:
                break
            candidate_id = candidate["index"]
            result = run_trial(args, run_dir, candidate["config"], 100 + candidate_id, "probe", candidate_id, 7, 2, probe_deadline)
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    successful = [item for item in results if item["ok"]]
    by_id = {item["index"]: item for item in successful}
    shortlisted = [by_id[index] for index in director.finalists(successful) if index in by_id]
    confirmations = []
    # Pair each finalist with the baseline on the same fresh seeds, so a noisy
    # probe cannot be mistaken for an improvement.
    confirmation_configs = [("baseline", BASE)] + [(str(candidate["index"]), candidate["config"]) for candidate in shortlisted]
    for index, (label, config) in enumerate(confirmation_configs):
        for seed in (501, 502, 503, 504):
            if time.monotonic() >= deadline:
                break
            result = run_trial(args, run_dir, config, seed, "confirm", index, 12, 3, deadline)
            result["confirmation_label"] = label
            confirmations.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "phase": "one_hour_deepfm_search", "selection_split": "validation", "test_data_used": False,
        "wall_clock_cap_seconds": args.wall_clock_seconds, "elapsed_seconds": time.monotonic() - started,
        "probe_results": results, "confirmation_results": confirmations,
        "api": {"enabled": args.api_guided, "model": args.model, "calls": director.calls,
                "input_tokens": director.input_tokens, "output_tokens": director.output_tokens,
                "estimated_spend_usd": director.estimated_spend, "cap_usd": args.max_api_spend_usd,
                "decisions": director.decisions},
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": str(run_dir / "summary.json"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main(parse_args())
