"""KuaiRand-Pure loader with the OFFICIAL splits/encoding, plus auxiliary targets.

Wraps the verbatim-vendored starter kit (data/official/data.py): the base feature
matrix uses exactly the official 5 fields, official train-vocab + UNK handling, and
official 10-quantile duration buckets, by calling the vendored ``encode`` directly.
On top of that it exposes auxiliary signal columns (click, like, play_time_ms,
duration_ms) as separate TARGET arrays, and raw context columns (hourmin, date, tab)
for feature engineering. Results are cached at data/real/cache.npz (gitignored) so
repeat loads take <5s.

Also loads the synthetic fixture directories (train.csv/val.csv/test.csv) into the
same structure, so zoo scripts accept either a synthetic dir or 'real'.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.data import SPLITS, encode  # official semantics, verbatim

REAL_DATA_DIR = ROOT.parent / "KuaiRand-Pure" / "data"
CACHE_PATH = ROOT / "data" / "real" / "cache.npz"

# Per-split arrays exposed to models. X/y/users are the official encoding;
# the rest are auxiliary targets / raw context, aligned row-for-row.
_KEYS = ("X", "y", "users", "videos", "click", "like", "play_time_ms",
         "duration_ms", "hourmin", "date", "tab")
_SPLIT_NAMES = ("train", "valid", "test")


def _load_real_rows():
    """Mirror of the official load(): same files, order, fields, split windows —
    extended with the auxiliary columns. Official encode() only reads indices 0-6,
    so the extended tuples feed it unchanged."""
    vid2author = {}
    with open(REAL_DATA_DIR / "video_features_basic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    rows = []
    for f in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(REAL_DATA_DIR / f) as fh:
            for r in csv.DictReader(fh):
                rows.append((
                    int(r["date"]), r["user_id"], r["video_id"],
                    vid2author.get(r["video_id"], "UNK"), r["tab"],
                    float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0,
                    int(r["is_click"]), int(r["is_like"]),
                    float(r["play_time_ms"]), int(r["hourmin"]),
                ))
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _build_real_cache() -> dict:
    splits = _load_real_rows()
    enc, total_dim = encode(splits)  # official encoding, verbatim code path
    data = {"field_dims_total": int(total_dim)}
    for name in _SPLIT_NAMES:
        X, y, users = enc[name]
        rws = splits[name]
        data[f"{name}_X"] = X
        data[f"{name}_y"] = y
        data[f"{name}_users"] = np.asarray([int(u) for u in users], dtype=np.int64)
        data[f"{name}_videos"] = np.asarray([int(x[2]) for x in rws], dtype=np.int64)
        data[f"{name}_click"] = np.asarray([x[7] for x in rws], dtype=np.float32)
        data[f"{name}_like"] = np.asarray([x[8] for x in rws], dtype=np.float32)
        data[f"{name}_play_time_ms"] = np.asarray([x[9] for x in rws], dtype=np.float32)
        data[f"{name}_duration_ms"] = np.asarray([x[5] for x in rws], dtype=np.float32)
        data[f"{name}_hourmin"] = np.asarray([x[10] for x in rws], dtype=np.int32)
        data[f"{name}_date"] = np.asarray([x[0] for x in rws], dtype=np.int32)
        data[f"{name}_tab"] = np.asarray([int(x[4]) for x in rws], dtype=np.int32)
    return data


def _load_synthetic(data_dir: Path) -> dict:
    """Load a synthetic fixture dir (train/val/test.csv) into the same structure,
    reusing the official encode() (author_id proxied by video_id — the fixtures
    have no author column)."""
    file_for = {"train": "train.csv", "valid": "val.csv", "test": "test.csv"}
    splits, extras = {}, {}
    for name, fname in file_for.items():
        rows = []
        with open(data_dir / fname) as fh:
            for r in csv.DictReader(fh):
                rows.append((
                    int(r["date"]), r["user_id"], r["video_id"], r["video_id"],
                    r["tab"], float(r["duration_ms"]),
                    1 if r["long_view"] != "0" else 0,
                    int(r.get("click", 0)), int(r.get("like", 0)),
                    float(r.get("play_time_ms", 0)), int(r.get("hourmin", 0)),
                ))
        splits[name] = rows
    enc, total_dim = encode(splits)
    data = {"field_dims_total": int(total_dim)}
    for name in _SPLIT_NAMES:
        X, y, users = enc[name]
        rws = splits[name]
        data[f"{name}_X"] = X
        data[f"{name}_y"] = y
        data[f"{name}_users"] = np.asarray([int(u) for u in users], dtype=np.int64)
        data[f"{name}_videos"] = np.asarray([int(x[2]) for x in rws], dtype=np.int64)
        data[f"{name}_click"] = np.asarray([x[7] for x in rws], dtype=np.float32)
        data[f"{name}_like"] = np.asarray([x[8] for x in rws], dtype=np.float32)
        data[f"{name}_play_time_ms"] = np.asarray([x[9] for x in rws], dtype=np.float32)
        data[f"{name}_duration_ms"] = np.asarray([x[5] for x in rws], dtype=np.float32)
        data[f"{name}_hourmin"] = np.asarray([x[10] for x in rws], dtype=np.int32)
        data[f"{name}_date"] = np.asarray([x[0] for x in rws], dtype=np.int32)
        data[f"{name}_tab"] = np.asarray([int(x[4]) for x in rws], dtype=np.int32)
    return data


def _to_split_dicts(flat: dict, subsample: int | None) -> dict:
    out = {"field_dims_total": int(flat["field_dims_total"])}
    # subsample budget: ~80% train, ~20% valid; test untouched (never trained on).
    limits = {}
    if subsample:
        limits = {"train": max(1, int(subsample * 0.8)),
                  "valid": max(1, subsample - max(1, int(subsample * 0.8)))}
    for name in _SPLIT_NAMES:
        split = {k: np.asarray(flat[f"{name}_{k}"]) for k in _KEYS}
        n = limits.get(name)
        if n:
            split = {k: v[:n] for k, v in split.items()}
        out[name] = split
    return out


def load_dataset(data_dir: str, subsample: int | None = None) -> dict:
    """Load 'real' (cached) or a synthetic fixture directory.

    Returns {'field_dims_total': int, 'train'|'valid'|'test': {key: ndarray}} with
    keys X, y, users, videos, click, like, play_time_ms, duration_ms, hourmin,
    date, tab. The test split is loaded only through this standard interface and
    must never be used for tuning.
    """
    if str(data_dir) == "real":
        if CACHE_PATH.exists():
            with np.load(CACHE_PATH, allow_pickle=False) as z:
                flat = {k: z[k] for k in z.files}
        else:
            flat = _build_real_cache()
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(CACHE_PATH) + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.savez(fh, **flat)
            os.replace(tmp, CACHE_PATH)
        return _to_split_dicts(flat, subsample)
    return _to_split_dicts(_load_synthetic(Path(data_dir)), subsample)


if __name__ == "__main__":
    d = load_dataset("real")
    for name in _SPLIT_NAMES:
        print(name, len(d[name]["y"]), "pos_rate", float(d[name]["y"].mean()))
    print("field_dims_total", d["field_dims_total"])
