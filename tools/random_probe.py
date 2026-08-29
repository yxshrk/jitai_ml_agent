"""Evaluation-only transfer probe built from KuaiRand random exposures.

The project legality ruling is important: ``log_random`` is not the hidden test
set (the hidden test is carved from ``log_standard``).  Even so, this runner
never trains or selects checkpoints on random-log rows.  It uses them only to
compare already-trained candidates after every model has been fit on the
standard-log train split and selected on the standard-log validation split.

Run the complete probe with::

    uv run python tools/random_probe.py

Large derived data and model artifacts are written below ``data/random_probe/``,
which is intentionally gitignored.  The concise, tracked report belongs in
``tools/RANDOM_PROBE.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate
from zoo import polish_stack
from zoo.ensemble import rank_average

ORIGINAL_SEEDS = (46, 74, 93, 91, 60)
WINDOWS = {
    "val_window": (20220422, 20220428),
    "test_window": (20220429, 20220508),
}
FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=ROOT.parent / "KuaiRand-Pure" / "data")
    ap.add_argument("--train-dir", type=Path, default=ROOT / "data" / "real_ws")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "random_probe")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--skip-coral", action="store_true",
                    help="do not attempt to retrieve the optional late-selected seed set")
    return ap


def _raw_values(row: dict[str, str], duration_edges: np.ndarray) -> tuple[str, ...]:
    return (row["user_id"], row["video_id"], row["author_id"], row["tab"],
            str(int(np.searchsorted(duration_edges, float(row["duration_ms"])))))


def _train_encoding(train_csv: Path) -> tuple[list[dict[str, int]], np.ndarray, np.ndarray]:
    """Recreate export_real_ws.py's vocab and duration quantiles from train only."""
    with train_csv.open(newline="", encoding="utf-8") as fh:
        train_rows = list(csv.DictReader(fh))
    durations = np.fromiter((float(row["duration_ms"]) for row in train_rows),
                            dtype=np.float64, count=len(train_rows))
    edges = np.quantile(durations, np.linspace(0, 1, 11)[1:-1])
    vocabs: list[dict[str, int]] = [dict() for _ in FIELDS]
    for row in train_rows:
        for index, value in enumerate(_raw_values(row, edges)):
            vocabs[index].setdefault(value, len(vocabs[index]))
    field_dims = np.asarray([len(vocab) + 1 for vocab in vocabs], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int32)
    return vocabs, edges, offsets


def _video_authors(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["video_id"]: row["author_id"] for row in csv.DictReader(fh)}


def build_probe(raw_dir: Path, train_dir: Path, out_dir: Path) -> dict[str, dict[str, int]]:
    """Encode both random-log windows without exposing them to model training."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocabs, edges, offsets = _train_encoding(train_dir / "train.csv")
    unknown = [len(vocab) for vocab in vocabs]
    field_dims = np.asarray([len(vocab) + 1 for vocab in vocabs], dtype=np.int64)
    with np.load(train_dir / "train.npz", allow_pickle=False) as train_npz:
        if not np.array_equal(field_dims, train_npz["field_dims"]):
            raise ValueError("reconstructed train vocabulary differs from train.npz")
    authors = _video_authors(raw_dir / "video_features_basic_pure.csv")
    buffers: dict[str, dict[str, list[Any]]] = {
        name: {key: [] for key in ("X", "y", "user", "video", "click", "play_time_ms",
                                         "duration_ms", "hourmin", "date")}
        for name in WINDOWS
    }
    random_log = raw_dir / "log_random_4_22_to_5_08_pure.csv"
    with random_log.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            date = int(row["date"])
            window = next((name for name, (low, high) in WINDOWS.items()
                           if low <= date <= high), None)
            if window is None:
                continue
            enriched = dict(row)
            enriched["author_id"] = authors.get(row["video_id"], "UNK")
            encoded = [vocabs[index].get(value, unknown[index]) + int(offsets[index])
                       for index, value in enumerate(_raw_values(enriched, edges))]
            target = buffers[window]
            target["X"].append(encoded)
            target["y"].append(int(row["long_view"]))
            target["user"].append(int(row["user_id"]))
            target["video"].append(int(row["video_id"]))
            target["click"].append(int(row["is_click"]))
            target["play_time_ms"].append(float(row["play_time_ms"]))
            target["duration_ms"].append(float(row["duration_ms"]))
            target["hourmin"].append(int(row["hourmin"]))
            target["date"].append(date)
    summaries: dict[str, dict[str, int]] = {}
    dtypes = {"X": np.int32, "y": np.float32, "user": np.int64, "video": np.int64,
              "click": np.float32, "play_time_ms": np.float32,
              "duration_ms": np.float32, "hourmin": np.int32, "date": np.int32}
    for name, values in buffers.items():
        arrays = {key: np.asarray(value, dtype=dtypes[key]) for key, value in values.items()}
        low, high = WINDOWS[name]
        if not len(arrays["date"]) or arrays["date"].min() < low or arrays["date"].max() > high:
            raise ValueError(f"invalid or empty {name} date range")
        if arrays["X"].shape != (len(arrays["y"]), len(FIELDS)):
            raise ValueError(f"invalid encoded shape for {name}: {arrays['X'].shape}")
        np.savez_compressed(out_dir / f"{name}.npz", **arrays, field_dims=field_dims)
        summaries[name] = {"rows": len(arrays["y"]),
                           "users": int(np.unique(arrays["user"]).size),
                           "min_date": int(arrays["date"].min()),
                           "max_date": int(arrays["date"].max())}
        print(f"built {name}: {json.dumps(summaries[name], sort_keys=True)}", flush=True)
    return summaries


def _parse_late_seeds(text: str) -> tuple[int, ...] | None:
    """Parse the final chosen/member seed list from the coral optimizer log."""
    matches = re.findall(
        r"(?im)^.*(?:chosen|selected|members?).*?seeds?[^\d\n]*([\d][\d,\s\[\]{}()-]*)$", text)
    if not matches:
        return None
    seeds = tuple(dict.fromkeys(int(value) for value in re.findall(r"\d+", matches[-1])))
    return seeds or None


def retrieve_late_seeds() -> tuple[tuple[int, ...] | None, str]:
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "pallav@coral.local", "cat", "~/techjam/ens_late.log"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"SSH retrieval failed: {exc}"
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")
        return None, f"SSH retrieval failed (exit {result.returncode}): {detail}"
    seeds = _parse_late_seeds(result.stdout)
    if seeds is None:
        return None, "ens_late.log was retrieved but its chosen member seeds were unparseable"
    return seeds, "retrieved and parsed ~/techjam/ens_late.log"


def _load_probe(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def _predict(model: torch.nn.Module, features: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.long)
    model.eval()
    with torch.no_grad():
        pieces = [model(tensor[start:start + 200_000])
                  for start in range(0, len(tensor), 200_000)]
    return torch.cat(pieces).numpy() if pieces else np.empty(0, dtype=np.float32)


def _metrics(split: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, float]:
    result = official_evaluate(split["user"].tolist(), split["y"].astype(int).tolist(),
                               scores.tolist())
    return {"gauc": float(result["GAUC"]), "ndcg5": float(result["nDCG@5"]),
            "primary": float(result["primary"])}


def train_seed(seed: int, dataset: dict[str, Any], probes: dict[str, dict[str, np.ndarray]],
               out_dir: Path) -> dict[str, Any]:
    """Run exact polish_stack defaults and capture its validation-selected model."""
    member_dir = out_dir / "models" / f"seed_{seed}"
    args = polish_stack.baseline_args(str(member_dir))
    args.seed = seed
    captured: list[torch.nn.Module] = []
    original_model = polish_stack.DCNLite

    def capture_model(*model_args: Any, **model_kwargs: Any) -> torch.nn.Module:
        model = original_model(*model_args, **model_kwargs)
        captured.append(model)
        return model

    polish_stack.DCNLite = capture_model  # type: ignore[assignment]
    try:
        validation_metrics = polish_stack.train_and_report(dataset, args)
    finally:
        polish_stack.DCNLite = original_model
    if len(captured) != 1:
        raise RuntimeError(f"expected one captured model for seed {seed}, got {len(captured)}")
    model = captured[0]
    torch.save({"seed": seed, "config": validation_metrics["config"],
                "state_dict": model.state_dict()}, member_dir / "checkpoint.pt")
    result: dict[str, Any] = {"candidate": f"seed {seed}", "kind": "single", "seed": seed,
                              "standard_val_primary": validation_metrics["primary"]}
    for window, split in probes.items():
        scores = _predict(model, split["X"])
        np.save(member_dir / f"{window}_scores.npy", scores)
        result[window] = _metrics(split, scores)
    result["delta"] = result["test_window"]["primary"] - result["val_window"]["primary"]
    with (member_dir / "probe_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True)
        fh.write("\n")
    print(f"probe seed {seed}: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def ensemble_result(name: str, seeds: Iterable[int], probes: dict[str, dict[str, np.ndarray]],
                    out_dir: Path) -> dict[str, Any]:
    seed_list = list(seeds)
    result: dict[str, Any] = {"candidate": name, "kind": "ensemble", "seeds": seed_list,
                              "method": "equal per-user rank average"}
    ensemble_dir = out_dir / "ensembles" / name
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    for window, split in probes.items():
        member_scores = [np.load(out_dir / "models" / f"seed_{seed}" /
                                 f"{window}_scores.npy") for seed in seed_list]
        scores = rank_average(split["user"], member_scores)
        np.save(ensemble_dir / f"{window}_scores.npy", scores)
        result[window] = _metrics(split, scores)
    result["delta"] = result["test_window"]["primary"] - result["val_window"]["primary"]
    with (ensemble_dir / "probe_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True)
        fh.write("\n")
    print(f"probe ensemble {name}: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    sizes = build_probe(args.raw_dir, args.train_dir, args.out_dir)
    if args.build_only:
        return {"probe_sizes": sizes}
    if args.skip_coral:
        late_seeds, coral_status = None, "coral lookup skipped by command-line request"
    else:
        late_seeds, coral_status = retrieve_late_seeds()
    ensembles: dict[str, tuple[int, ...]] = {"original": ORIGINAL_SEEDS}
    if late_seeds is not None and late_seeds != ORIGINAL_SEEDS:
        ensembles["late_selected"] = late_seeds
    all_seeds = tuple(dict.fromkeys((*ORIGINAL_SEEDS,
                                    *(late_seeds or ()), 42)))
    dataset = polish_stack.load_validation_only(str(args.train_dir))
    probes = {name: _load_probe(args.out_dir / f"{name}.npz") for name in WINDOWS}
    candidates = [train_seed(seed, dataset, probes, args.out_dir) for seed in all_seeds]
    candidates.extend(ensemble_result(name, seeds, probes, args.out_dir)
                      for name, seeds in ensembles.items())
    payload = {"probe_sizes": sizes, "coral_status": coral_status,
               "late_seeds": list(late_seeds) if late_seeds else None,
               "trained_seeds": list(all_seeds), "runtime_s": round(time.time() - started, 1),
               "candidates": candidates}
    with (args.out_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"complete: {json.dumps(payload, sort_keys=True)}", flush=True)
    return payload


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
