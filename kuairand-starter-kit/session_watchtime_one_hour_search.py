"""API-directed, one-hour validation-only search around the current leader.

The fixed starting point is the causal session-aware Sequence DeepFM with the
censor-aware watch-time loss.  The API may only choose from predeclared local
configurations; it cannot supply code, data, features, or test-set requests.
The last part of the wall clock is reserved for fresh-seed confirmation.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


# Conservative accounting guard for the requested gpt-5-mini director. The
# call-count cap remains authoritative if pricing changes.
INPUT_PRICE_PER_MILLION = 0.25
OUTPUT_PRICE_PER_MILLION = 2.00
BASE = {
    "history_length": 8,
    "embedding_dim": 16,
    "hidden_dim": 64,
    "dropout": 0.10,
    "learning_rate": 0.001,
    "weight_decay": 1e-6,
    "watchtime_aux_weight": 0.02,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wall_clock_seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--max_probes", type=int, default=24)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max_api_spend_usd", type=float, default=25.0)
    parser.add_argument("--max_api_calls", type=int, default=6)
    return parser.parse_args()


def load_project_key():
    """Load an ignored project-local key without ever logging its value."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENAI_API_KEY" and value.strip():
            os.environ.setdefault("OPENAI_API_KEY", value.strip())
            return


class Director:
    """Choose experiment IDs only through Structured Outputs."""

    def __init__(self, args):
        load_project_key()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY must be present in the ignored project .env file")
        from openai import OpenAI

        self.args = args
        self.client = OpenAI()
        self.calls = self.input_tokens = self.output_tokens = 0
        self.decisions = []

    @property
    def estimated_spend(self):
        return (self.input_tokens * INPUT_PRICE_PER_MILLION + self.output_tokens * OUTPUT_PRICE_PER_MILLION) / 1_000_000

    def choose(self, phase, available, observed, count):
        fallback = [candidate["id"] for candidate in available[:count]]
        projected = self.estimated_spend + (12_000 * INPUT_PRICE_PER_MILLION + 800 * OUTPUT_PRICE_PER_MILLION) / 1_000_000
        if self.calls >= self.args.max_api_calls or projected > self.args.max_api_spend_usd:
            return fallback
        schema = {
            "type": "json_schema",
            "name": "session_watchtime_experiment_selection",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"candidate_ids": {"type": "array", "items": {"type": "integer"}}},
                "required": ["candidate_ids"],
                "additionalProperties": False,
            },
        }
        observed_rows = [
            {
                "id": row["id"], "primary": row["best"]["metrics"]["primary"],
                "gauc": row["best"]["metrics"]["GAUC"], "ndcg5": row["best"]["metrics"]["nDCG@5"],
                "epoch": row["best"]["epoch"],
            }
            for row in observed if row["ok"]
        ]
        response = self.client.responses.create(
            model=self.args.model,
            reasoning={"effort": "low"},
            max_output_tokens=800,
            text={"format": schema},
            input=(
                "You direct a bounded recommender-model experiment. Select at most " + str(count) +
                " candidate IDs for the next validation-only batch. The incumbent uses causal session metadata "
                "and a censor-aware watch-time auxiliary loss. Choose diverse, modest changes and treat scores as noisy. "
                "You may select IDs only; do not propose code, features, data, objective changes, or test-set access.\n\n" +
                json.dumps({"phase": phase, "incumbent": BASE, "available": available, "observed": observed_rows})
            ),
        )
        usage = getattr(response, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.calls += 1
        try:
            payload = json.loads(response.output_text)
            selected = list(dict.fromkeys(int(value) for value in payload["candidate_ids"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            payload, selected = {"fallback": True}, fallback
        allowed = {candidate["id"] for candidate in available}
        if not selected or len(selected) > count or any(value not in allowed for value in selected):
            selected = fallback
        self.decisions.append({
            "phase": phase, "response": payload, "raw_output": response.output_text,
            "input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None),
        })
        return selected


def candidate_menu(rng, maximum):
    """Predeclared, local-only variants near the proven architecture."""
    curated = [
        BASE,
        {**BASE, "watchtime_aux_weight": 0.01},
        {**BASE, "watchtime_aux_weight": 0.03},
        {**BASE, "watchtime_aux_weight": 0.04},
        {**BASE, "dropout": 0.05},
        {**BASE, "dropout": 0.15},
        {**BASE, "learning_rate": 0.00085},
        {**BASE, "learning_rate": 0.00115},
        {**BASE, "weight_decay": 1e-7},
        {**BASE, "weight_decay": 3e-6},
        {**BASE, "history_length": 4},
        {**BASE, "history_length": 12},
        {**BASE, "hidden_dim": 48},
        {**BASE, "hidden_dim": 96},
        {**BASE, "embedding_dim": 12},
        {**BASE, "embedding_dim": 20},
        {**BASE, "watchtime_aux_weight": 0.03, "dropout": 0.05},
        {**BASE, "watchtime_aux_weight": 0.01, "dropout": 0.15},
        {**BASE, "learning_rate": 0.00085, "weight_decay": 3e-6},
        {**BASE, "history_length": 12, "dropout": 0.15},
        {**BASE, "hidden_dim": 48, "dropout": 0.05},
        {**BASE, "embedding_dim": 20, "weight_decay": 3e-6},
    ]
    fingerprints = {tuple(config.items()) for config in curated}
    while len(curated) < maximum:
        candidate = {
            "history_length": int(rng.choice((4, 8, 12))),
            "embedding_dim": int(rng.choice((12, 16, 20))),
            "hidden_dim": int(rng.choice((48, 64, 96))),
            "dropout": float(rng.choice((0.05, 0.10, 0.15))),
            "learning_rate": float(rng.choice((0.00085, 0.001, 0.00115))),
            "weight_decay": float(rng.choice((1e-7, 1e-6, 3e-6))),
            "watchtime_aux_weight": float(rng.choice((0.01, 0.02, 0.03, 0.04))),
        }
        fingerprint = tuple(candidate.items())
        if fingerprint not in fingerprints:
            curated.append(candidate)
            fingerprints.add(fingerprint)
    return [{"id": index, "config": config} for index, config in enumerate(curated[:maximum])]


def run_trial(args, run_dir, candidate, seed, phase, epochs, patience, deadline):
    trial_dir = run_dir / f"{phase}_{candidate['id']:02d}_seed{seed}"
    trial_dir.mkdir(parents=True)
    log_path = trial_dir / "result.json"
    command = [
        sys.executable, "-u", "sequence_deepfm.py", "--data_dir", args.data_dir,
        "--seed", str(seed), "--epochs", str(epochs), "--patience", str(patience),
        "--batch_size", "8192", "--session_features", "--run_log", str(log_path),
    ]
    for key, value in candidate["config"].items():
        command.extend((f"--{key}", str(value)))
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    started = time.monotonic()
    try:
        result = subprocess.run(
            command, cwd=Path(__file__).parent, capture_output=True, text=True, env=environment,
            timeout=max(1, int(deadline - time.monotonic())), check=False,
        )
    except subprocess.TimeoutExpired:
        return {"phase": phase, "id": candidate["id"], "seed": seed, "config": candidate["config"], "ok": False,
                "duration_seconds": time.monotonic() - started, "error": "wall-clock deadline reached"}
    if result.returncode != 0:
        return {"phase": phase, "id": candidate["id"], "seed": seed, "config": candidate["config"], "ok": False,
                "duration_seconds": time.monotonic() - started, "error": result.stderr[-2000:]}
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    return {"phase": phase, "id": candidate["id"], "seed": seed, "config": candidate["config"], "ok": True,
            "duration_seconds": time.monotonic() - started, "best": payload["best"]}


def main(args):
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.wall_clock_seconds
    probe_deadline = started + int(args.wall_clock_seconds * 0.65)
    director = Director(args)
    menu = candidate_menu(np.random.default_rng(args.seed), args.max_probes)
    remaining, probes = menu.copy(), []
    batch_number = 0
    while remaining and time.monotonic() < probe_deadline:
        batch_number += 1
        selected_ids = set(director.choose(f"probe_batch_{batch_number}", remaining, probes, min(6, len(remaining))))
        batch = [candidate for candidate in remaining if candidate["id"] in selected_ids]
        remaining = [candidate for candidate in remaining if candidate["id"] not in selected_ids]
        for candidate in batch:
            if time.monotonic() >= probe_deadline:
                break
            result = run_trial(args, run_dir, candidate, 1000 + candidate["id"], "probe", 8, 2, probe_deadline)
            probes.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    successful = [row for row in probes if row["ok"]]
    by_id = {candidate["id"]: candidate for candidate in menu}
    finalist_ids = director.choose("finalist_selection", successful, successful, 2) if successful else []
    finalists = [by_id[candidate_id] for candidate_id in finalist_ids if candidate_id in by_id]
    confirmations = []
    confirmation_candidates = [{"id": -1, "config": BASE}] + finalists
    for candidate in confirmation_candidates:
        for seed in (2001, 2002, 2003, 2004):
            if time.monotonic() >= deadline:
                break
            result = run_trial(args, run_dir, candidate, seed, "confirm", 12, 3, deadline)
            confirmations.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "phase": "session_watchtime_api_directed_one_hour_search",
        "selection_split": "validation", "test_data_used": False,
        "wall_clock_cap_seconds": args.wall_clock_seconds, "elapsed_seconds": time.monotonic() - started,
        "base_config": BASE, "candidate_menu": menu, "probe_results": probes, "confirmation_results": confirmations,
        "api": {"model": args.model, "calls": director.calls, "input_tokens": director.input_tokens,
                "output_tokens": director.output_tokens, "estimated_spend_usd": director.estimated_spend,
                "cap_usd": args.max_api_spend_usd, "decisions": director.decisions},
    }
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(run_dir / "summary.json"), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main(parse_args())
