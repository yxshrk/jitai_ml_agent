"""Leakage-safe sequence preparation for validation-only recommendation runs.

Each training target receives at most ``max_history`` earlier train impressions.
Rows with the same ``(date, hourmin)`` are treated as simultaneous: the complete
timestamp group is appended only after histories for every row in the group are
recorded.  Thus an outcome mark is available iff its timestamp is *strictly*
earlier than the target timestamp.  Validation histories are immutable snapshots
of the user's complete train-window history; validation labels are never read by
the history builder and can never become validation features.

The cache stores history row indices plus train-row feature sources.  This is
equivalent to materializing four dense history tensors, but is much smaller and
allows outcome marks to be enabled/disabled as a model ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)
PAD = 0
UNK = 1


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path, expected: int) -> dict[str, np.ndarray]:
    columns: dict[str, list[int]] = {name: [] for name in
        ("user", "video_raw", "author_raw", "hourmin", "date")}
    duration: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            columns["user"].append(int(row["user_id"]))
            columns["video_raw"].append(int(row["video_id"]))
            columns["author_raw"].append(int(row["author_id"]))
            columns["hourmin"].append(int(row["hourmin"]))
            columns["date"].append(int(row["date"]))
            duration.append(float(row["duration_ms"]))
    if len(duration) != expected:
        raise ValueError(f"{path} has {len(duration)} rows; expected {expected}")
    result = {key: np.asarray(value, dtype=np.int64) for key, value in columns.items()}
    result["duration_ms"] = np.asarray(duration, dtype=np.float32)
    return result


def _vocab(values: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    unique = np.unique(values)
    mapping = {int(value): offset + 2 for offset, value in enumerate(unique)}
    return mapping, unique


def _encode(values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.fromiter((mapping.get(int(value), UNK) for value in values),
                       dtype=np.int32, count=len(values))


def _stamp(date: np.ndarray, hourmin: np.ndarray) -> np.ndarray:
    return date.astype(np.int64) * 10_000 + hourmin.astype(np.int64)


def build_history_indices(train_users: np.ndarray, train_stamps: np.ndarray,
                          valid_users: np.ndarray, max_history: int
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal train histories and train-only validation snapshots.

    The returned index matrices use ``-1`` for left padding.  Sorting includes
    original row id only as deterministic tie-breaking; outcomes from tied
    timestamps are appended together after all targets at that timestamp.
    """
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    n = len(train_users)
    histories = np.full((n, max_history), -1, dtype=np.int32)
    current: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=max_history))
    full_counts: dict[int, int] = defaultdict(int)
    order = np.lexsort((np.arange(n), train_stamps, train_users))
    start = 0
    while start < n:
        user = int(train_users[order[start]])
        end_user = start + 1
        while end_user < n and int(train_users[order[end_user]]) == user:
            end_user += 1
        cursor = start
        while cursor < end_user:
            stamp = int(train_stamps[order[cursor]])
            end_stamp = cursor + 1
            while end_stamp < end_user and int(train_stamps[order[end_stamp]]) == stamp:
                end_stamp += 1
            prior = list(current[user])
            if prior:
                for row_id in order[cursor:end_stamp]:
                    histories[row_id, -len(prior):] = prior
            group = [int(row_id) for row_id in order[cursor:end_stamp]]
            current[user].extend(group)
            full_counts[user] += len(group)
            cursor = end_stamp
        start = end_user

    valid_histories = np.full((len(valid_users), max_history), -1, dtype=np.int32)
    valid_counts = np.empty(len(valid_users), dtype=np.int32)
    for row_id, user_value in enumerate(valid_users):
        user = int(user_value)
        prior = list(current.get(user, ()))
        if prior:
            valid_histories[row_id, -len(prior):] = prior
        valid_counts[row_id] = full_counts.get(user, 0)
    return histories, valid_histories, valid_counts


def assert_no_leakage(data: dict[str, np.ndarray]) -> None:
    """Raise if any cached history uses a same/future or non-train outcome."""
    histories = data["train_history"]
    train_stamps = data["train_stamp"]
    for start in range(0, len(histories), 100_000):
        block = histories[start:start + 100_000]
        target = train_stamps[start:start + len(block), None]
        used = block >= 0
        safe = np.where(used, block, 0)
        if np.any(used & (train_stamps[safe] >= target)):
            raise AssertionError("train history contains same-or-later timestamp outcome")
    valid_history = data["valid_history"]
    if valid_history.size and (valid_history.min() < -1 or valid_history.max(initial=-1) >= len(train_stamps)):
        raise AssertionError("validation history references a non-train row")
    if np.any((valid_history >= 0) &
              (train_stamps[np.where(valid_history >= 0, valid_history, 0)] >=
               data["valid_stamp"][:, None])):
        raise AssertionError("validation history is not strictly earlier than validation target")


def prepare(data_dir: str | Path = "data/real_ws", max_history: int = 50
            ) -> dict[str, np.ndarray]:
    base = _resolve(data_dir)
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for name, stem, bounds in (("train", "train", TRAIN_DATES),
                               ("valid", "val", VALID_DATES)):
        with np.load(base / f"{stem}.npz", allow_pickle=False) as archive:
            split = {key: np.asarray(archive[key]).copy() for key in archive.files}
        rows = _read_csv(base / f"{stem}.csv", len(split["y"]))
        split.update(rows)
        low, high = bounds
        if len(split["date"]) and (split["date"].min() < low or split["date"].max() > high):
            raise ValueError(f"forbidden date in {name}")
        loaded[name] = split

    tr, va = loaded["train"], loaded["valid"]
    video_map, video_values = _vocab(tr["video_raw"])
    author_map, author_values = _vocab(tr["author_raw"])
    duration_edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    tr_stamp, va_stamp = _stamp(tr["date"], tr["hourmin"]), _stamp(va["date"], va["hourmin"])
    train_history, valid_history, valid_counts = build_history_indices(
        tr["user"], tr_stamp, va["user"], max_history)
    result = {
        "train_history": train_history,
        "valid_history": valid_history,
        "valid_history_count": valid_counts,
        "train_video": _encode(tr["video_raw"], video_map),
        "valid_video": _encode(va["video_raw"], video_map),
        "train_author": _encode(tr["author_raw"], author_map),
        "valid_author": _encode(va["author_raw"], author_map),
        "train_duration": (np.searchsorted(duration_edges, tr["duration_ms"]) + 1).astype(np.int16),
        "valid_duration": (np.searchsorted(duration_edges, va["duration_ms"]) + 1).astype(np.int16),
        "train_outcome": tr["y"].astype(np.int8),
        "valid_label": va["y"].astype(np.int8),
        "train_user": tr["user"].astype(np.int64),
        "valid_user": va["user"].astype(np.int64),
        "train_date": tr["date"].astype(np.int32),
        "valid_date": va["date"].astype(np.int32),
        "train_stamp": tr_stamp,
        "valid_stamp": va_stamp,
        "n_video": np.asarray(len(video_values) + 2, dtype=np.int64),
        "n_author": np.asarray(len(author_values) + 2, dtype=np.int64),
        "n_duration": np.asarray(11, dtype=np.int64),
        "max_history": np.asarray(max_history, dtype=np.int64),
    }
    assert_no_leakage(result)
    return result


def load_or_prepare(data_dir: str | Path = "data/real_ws", max_history: int = 50,
                    cache: str | Path | None = None, rebuild: bool = False
                    ) -> dict[str, np.ndarray]:
    cache_path = Path(cache) if cache is not None else Path("/tmp") / f"mle_seq_cache_h{max_history}.npz"
    if cache_path.exists() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as archive:
            result = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if int(result["max_history"]) != max_history:
            raise ValueError("cache max_history mismatch")
        assert_no_leakage(result)
        return result
    result = prepare(data_dir, max_history)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as fh:
        np.savez_compressed(fh, **result)
    temporary.replace(cache_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/real_ws")
    ap.add_argument("--max-history", type=int, choices=(20, 50), default=50)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    data = load_or_prepare(args.data_dir, args.max_history, args.cache, args.rebuild)
    print(json.dumps({"train_rows": len(data["train_user"]),
                      "valid_rows": len(data["valid_user"]),
                      "max_history": int(data["max_history"]),
                      "n_video": int(data["n_video"]),
                      "n_author": int(data["n_author"]),
                      "leakage_check": "passed"}, sort_keys=True))


if __name__ == "__main__":
    main()
