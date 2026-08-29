"""Train seeded copies of a champion script and rank-average their predictions.

This is a validation-only closing move.  Each member obeys the experiment-script
contract, and this wrapper emits the same predictions.csv/metrics.json interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate as official_evaluate

TOTAL_BUDGET_SECONDS = 570


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--member-script", type=Path, default=ROOT / "zoo" / "polish_stack.py")
    ap.add_argument("--member-args", default="", help="shell-style extra arguments for every member")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--member-epochs", type=int, default=8)
    return ap


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = ("row_id", "user_id", "video_id", "score")
    if not rows or tuple(rows[0]) != columns:
        raise ValueError(f"{path} is empty or does not have the contract header")
    arrays = tuple(
        np.fromiter((cast(row[name]) for row in rows), dtype=dtype, count=len(rows))
        for name, cast, dtype in (
            ("row_id", int, np.int64),
            ("user_id", int, np.int64),
            ("video_id", int, np.int64),
            ("score", float, np.float64),
        )
    )
    if not np.array_equal(arrays[0], np.arange(len(rows))):
        raise ValueError(f"{path} row_id values are not in validation-file order")
    return arrays  # type: ignore[return-value]


def _within_user_ranks(users: np.ndarray, scores: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    starts = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        indices = order[start:end]
        score_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float64)
        ranks[score_order] = np.arange(len(indices), dtype=np.float64)
        if len(indices) > 1:
            ranks /= len(indices) - 1
        result[indices] = ranks
    return result


def _validation_labels(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = data_dir / "val.npz"
    if npz_path.exists():
        with np.load(npz_path) as valid:
            return valid["user"].astype(np.int64), valid["y"].astype(np.int64)
    with (data_dir / "val.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.fromiter((int(row["user_id"]) for row in rows), dtype=np.int64, count=len(rows)),
        np.fromiter((int(row["long_view"]) for row in rows), dtype=np.int64, count=len(rows)),
    )


def _metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    raw = official_evaluate(users.tolist(), labels.tolist(), scores.tolist())
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg5": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }


def run(args: argparse.Namespace) -> dict:
    if args.n_members <= 0:
        raise ValueError("n-members must be positive")
    if args.member_epochs <= 0:
        raise ValueError("member-epochs must be positive")

    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    effective_member_epochs = args.member_epochs
    if smoke_epochs is not None:
        try:
            effective_member_epochs = min(effective_member_epochs, int(smoke_epochs))
        except ValueError as exc:
            raise ValueError("SMOKE_EPOCHS must be an integer") from exc
        if effective_member_epochs <= 0:
            raise ValueError("SMOKE_EPOCHS must be positive")

    started = time.monotonic()
    data_dir = _resolved(Path(args.data_dir))
    member_script = _resolved(args.member_script)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.split(args.member_args)
    loaded: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    history: list[dict] = []

    for member_index in range(args.n_members):
        seed = args.seed + member_index
        member_dir = out_dir / f"member_{member_index:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(member_script),
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(member_dir),
            "--seed",
            str(seed),
            *extra_args,
            "--epochs",
            str(effective_member_epochs),
        ]
        remaining = TOTAL_BUDGET_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"seed ensemble exceeded its {TOTAL_BUDGET_SECONDS}s budget")
        try:
            subprocess.run(
                command,
                check=True,
                cwd=ROOT,
                timeout=remaining,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"member {member_index + 1} exceeded the ensemble budget") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no child output").strip()
            raise RuntimeError(f"member {member_index + 1} failed:\n{detail}") from exc
        predictions_path = member_dir / "predictions.csv"
        metrics_path = member_dir / "metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            raise RuntimeError(f"member {member_index + 1} did not produce contract outputs")
        member_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in ("gauc", "ndcg5", "primary"):
            if not isinstance(member_metrics.get(key), (int, float)):
                raise ValueError(f"member {member_index + 1} has invalid metric {key}")
        loaded.append(_read_predictions(predictions_path))
        history.append({
            "stage": "member",
            "member": member_index + 1,
            "seed": seed,
            "val_gauc": float(member_metrics["gauc"]),
            "val_ndcg5": float(member_metrics["ndcg5"]),
            "val_primary": float(member_metrics["primary"]),
        })

    row_ids, users, videos, _ = loaded[0]
    for member_index, (other_rows, other_users, other_videos, _scores) in enumerate(loaded[1:], 2):
        if not (
            np.array_equal(row_ids, other_rows)
            and np.array_equal(users, other_users)
            and np.array_equal(videos, other_videos)
        ):
            raise ValueError(f"member {member_index} prediction rows are not aligned")

    rank_columns = [_within_user_ranks(users, member[3]) for member in loaded]
    ensemble_scores = np.mean(np.column_stack(rank_columns), axis=1)
    official_users, labels = _validation_labels(data_dir)
    if len(labels) != len(ensemble_scores) or not np.array_equal(users, official_users):
        raise ValueError("member predictions do not align with the validation split")
    metrics = _metrics(users, labels, ensemble_scores)
    history.append({
        "stage": "ensemble",
        "members": args.n_members,
        "val_gauc": metrics["gauc"],
        "val_ndcg5": metrics["ndcg5"],
        "val_primary": metrics["primary"],
    })
    result = {
        **metrics,
        "history": history,
        "config": {
            "seed": args.seed,
            "n_members": args.n_members,
            "member_epochs": effective_member_epochs,
            "member_script": str(member_script),
            "member_args": args.member_args,
        },
    }

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        writer.writerows(
            (int(row), int(user), int(video), f"{score:.10f}")
            for row, user, video, score in zip(row_ids, users, videos, ensemble_scores)
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
