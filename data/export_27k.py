"""Stream KuaiRand-27K into train/validation NPZ files.

The vocabulary and duration-decile boundaries are learned from the training
window only.  Dates on or after 2022-04-29 are deliberately never exported.
"""

from __future__ import annotations

import os
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get("KR_RAW", HERE.parent.parent / "KuaiRand-27K" / "data"))
OUT = Path(os.environ.get("KR_OUT", HERE / "real_ws_27k"))
CHUNKSIZE = int(os.environ.get("KR_CHUNKSIZE", "2000000"))

SPLITS = {
    "train": (20220408, 20220421),
    "val": (20220422, 20220428),
}
LOG_FILES = (
    "log_standard_4_08_to_4_21_27k_part1.csv",
    "log_standard_4_08_to_4_21_27k_part2.csv",
    "log_standard_4_22_to_5_08_27k_part1.csv",
    "log_standard_4_22_to_5_08_27k_part2.csv",
)
FEATURE_FILE = "video_features_basic_27k.csv"
LOG_COLS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "is_click",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "tab",
)
LOG_DTYPES = {
    "user_id": np.int32,
    "video_id": np.int32,
    "date": np.int32,
    "hourmin": np.int32,
    "is_click": np.float32,
    "long_view": np.float32,
    "play_time_ms": np.float32,
    "duration_ms": np.float32,
    "tab": np.int32,
}
ARRAY_SPECS = {
    "X": (np.int32, 5),
    "y": (np.float32, None),
    "user": (np.int32, None),
    "click": (np.float32, None),
    "play_time_ms": (np.float32, None),
    "duration_ms": (np.float32, None),
    "hourmin": (np.int32, None),
    "date": (np.int32, None),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def input_paths() -> list[Path]:
    paths = [RAW / name for name in (*LOG_FILES, FEATURE_FILE)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing raw inputs: " + ", ".join(missing))
    return paths


def log_chunks(columns: tuple[str, ...] = LOG_COLS):
    dtypes = {column: LOG_DTYPES[column] for column in columns}
    for filename in LOG_FILES:
        path = RAW / filename
        log(f"reading {filename}")
        yield from pd.read_csv(
            path,
            usecols=list(columns),
            dtype=dtypes,
            chunksize=CHUNKSIZE,
            engine="c",
        )


def split_mask(date: np.ndarray, split: str) -> np.ndarray:
    lo, hi = SPLITS[split]
    return (date >= lo) & (date <= hi)


def duration_quantiles(histogram: Counter[float], count: int) -> np.ndarray:
    """Match NumPy's default linear quantile using a compact value histogram."""
    values = np.asarray(sorted(histogram), dtype=np.float64)
    counts = np.asarray([histogram[value] for value in values], dtype=np.int64)
    cumulative = np.cumsum(counts)
    result = []
    for q in np.linspace(0.0, 1.0, 11)[1:-1]:
        position = (count - 1) * q
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        lower_value = values[np.searchsorted(cumulative, lower, side="right")]
        upper_value = values[np.searchsorted(cumulative, upper, side="right")]
        result.append(lower_value + (position - lower) * (upper_value - lower_value))
    return np.asarray(result, dtype=np.float64)


def scan_logs():
    counts = {name: 0 for name in SPLITS}
    train_users: set[int] = set()
    train_videos: set[int] = set()
    train_tabs: set[int] = set()
    exported_videos: set[int] = set()
    durations: Counter[float] = Counter()

    for chunk in log_chunks(("user_id", "video_id", "date", "duration_ms", "tab")):
        date = chunk["date"].to_numpy(copy=False)
        for split in SPLITS:
            mask = split_mask(date, split)
            selected = chunk.loc[mask]
            counts[split] += len(selected)
            exported_videos.update(map(int, selected["video_id"].unique()))
            if split == "train":
                train_users.update(map(int, selected["user_id"].unique()))
                train_videos.update(map(int, selected["video_id"].unique()))
                train_tabs.update(map(int, selected["tab"].unique()))
                unique, frequency = np.unique(
                    selected["duration_ms"].to_numpy(dtype=np.float64),
                    return_counts=True,
                )
                durations.update(dict(zip(unique.tolist(), frequency.tolist())))

    if not counts["train"] or not counts["val"]:
        raise RuntimeError(f"Empty split found: {counts}")
    edges = duration_quantiles(durations, counts["train"])
    log(f"pass 1 counts: {counts}; duration edges: {edges.tolist()}")
    return counts, train_users, train_videos, train_tabs, exported_videos, edges


def load_needed_authors(exported_videos: set[int], train_videos: set[int]):
    video_to_author: dict[int, int] = {}
    train_authors: set[int] = set()
    log(f"scanning {FEATURE_FILE} for {len(exported_videos):,} needed videos")
    for chunk in pd.read_csv(
        RAW / FEATURE_FILE,
        usecols=["video_id", "author_id"],
        dtype={"video_id": np.int32, "author_id": np.int32},
        chunksize=CHUNKSIZE,
        engine="c",
    ):
        selected = chunk.loc[chunk["video_id"].isin(exported_videos)]
        pairs = zip(selected["video_id"].array, selected["author_id"].array)
        for video, author in pairs:
            video_i, author_i = int(video), int(author)
            video_to_author[video_i] = author_i
            if video_i in train_videos:
                train_authors.add(author_i)
    missing = len(exported_videos) - len(video_to_author)
    log(f"loaded {len(video_to_author):,} author mappings ({missing:,} missing -> UNK)")
    return video_to_author, train_authors


def make_vocab(values: set[int]) -> np.ndarray:
    return np.asarray(sorted(values), dtype=np.int32)


def encode_values(values: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """Encode against a sorted training vocabulary, using len(vocab) as UNK."""
    positions = np.searchsorted(vocabulary, values)
    encoded = np.full(values.shape, len(vocabulary), dtype=np.int32)
    valid = positions < len(vocabulary)
    encoded[valid] = np.where(
        vocabulary[positions[valid]] == values[valid],
        positions[valid],
        len(vocabulary),
    )
    return encoded


def open_arrays(split: str, rows: int):
    arrays = {}
    paths = {}
    for name, (dtype, width) in ARRAY_SPECS.items():
        shape = (rows, width) if width else (rows,)
        path = OUT / f".{split}.{name}.npy"
        paths[name] = path
        arrays[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return arrays, paths


def encode_and_write(counts, video_keys, video_authors, vocabularies, edges, offsets):
    arrays = {}
    paths = {}
    positions = {name: 0 for name in SPLITS}
    for split, rows in counts.items():
        arrays[split], paths[split] = open_arrays(split, rows)

    users, videos, authors, tabs, buckets = vocabularies

    for chunk in log_chunks():
        date = chunk["date"].to_numpy(copy=False)
        for split in SPLITS:
            selected = chunk.loc[split_mask(date, split)]
            n = len(selected)
            if not n:
                continue
            start, stop = positions[split], positions[split] + n
            target = arrays[split]
            raw_user = selected["user_id"].to_numpy(dtype=np.int32, copy=False)
            raw_video = selected["video_id"].to_numpy(dtype=np.int32, copy=False)
            raw_tab = selected["tab"].to_numpy(dtype=np.int32, copy=False)
            raw_duration = selected["duration_ms"].to_numpy(dtype=np.float32, copy=False)
            author_positions = np.searchsorted(video_keys, raw_video)
            raw_author = np.full(n, np.iinfo(np.int32).min, dtype=np.int32)
            author_valid = author_positions < len(video_keys)
            author_valid[author_valid] &= (
                video_keys[author_positions[author_valid]] == raw_video[author_valid]
            )
            raw_author[author_valid] = video_authors[author_positions[author_valid]]
            raw_bucket = np.searchsorted(edges, raw_duration).astype(np.int32)

            columns = (
                encode_values(raw_user, users),
                encode_values(raw_video, videos),
                encode_values(raw_author, authors),
                encode_values(raw_tab, tabs),
                encode_values(raw_bucket, buckets),
            )
            for field, values in enumerate(columns):
                target["X"][start:stop, field] = values + offsets[field]
            target["y"][start:stop] = selected["long_view"].to_numpy(dtype=np.float32)
            target["user"][start:stop] = raw_user
            target["click"][start:stop] = selected["is_click"].to_numpy(dtype=np.float32)
            target["play_time_ms"][start:stop] = selected["play_time_ms"].to_numpy(dtype=np.float32)
            target["duration_ms"][start:stop] = raw_duration
            target["hourmin"][start:stop] = selected["hourmin"].to_numpy(dtype=np.int32)
            target["date"][start:stop] = selected["date"].to_numpy(dtype=np.int32)
            positions[split] = stop

    if positions != counts:
        raise RuntimeError(f"Pass counts changed: expected {counts}, wrote {positions}")
    for split_arrays in arrays.values():
        for array in split_arrays.values():
            array.flush()
    return arrays, paths


def package(arrays, paths, field_dims):
    for split in SPLITS:
        destination = OUT / f"{split}.npz"
        temporary = OUT / f".{split}.npz.tmp"
        log(f"compressing {split} arrays into {destination.name}")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays[split], field_dims=field_dims)
        os.replace(temporary, destination)
        arrays[split].clear()
        for path in paths[split].values():
            path.unlink()
        log(f"finished {destination.name} ({destination.stat().st_size / 2**30:.2f} GiB)")


def main() -> None:
    started = time.monotonic()
    input_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    forbidden = [OUT / "test.npz", OUT / "train.csv", OUT / "val.csv", OUT / "test.csv"]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError("Refusing to run with forbidden output files present: " + ", ".join(existing))

    counts, train_users, train_videos, train_tabs, exported_videos, edges = scan_logs()
    video_to_author, train_authors = load_needed_authors(exported_videos, train_videos)
    video_keys = np.fromiter(video_to_author, dtype=np.int32, count=len(video_to_author))
    video_authors = np.fromiter(
        video_to_author.values(), dtype=np.int32, count=len(video_to_author)
    )
    order = np.argsort(video_keys)
    video_keys, video_authors = video_keys[order], video_authors[order]
    del video_to_author, order
    bucket_values = set(range(10))
    vocabularies = tuple(
        map(make_vocab, (train_users, train_videos, train_authors, train_tabs, bucket_values))
    )
    field_dims = np.asarray([len(vocab) + 1 for vocab in vocabularies], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int32)
    temporary_bytes = sum(counts.values()) * 48
    free_bytes = shutil.disk_usage(OUT).free
    log(
        f"field_dims={field_dims.tolist()}; temporary arrays need "
        f"{temporary_bytes / 2**30:.2f} GiB; {free_bytes / 2**30:.2f} GiB free"
    )
    if temporary_bytes + 2 * 2**30 > free_bytes:
        raise RuntimeError("Insufficient free disk space for temporary typed arrays")

    arrays, paths = encode_and_write(
        counts, video_keys, video_authors, vocabularies, edges, offsets
    )
    package(arrays, paths, field_dims)
    elapsed = time.monotonic() - started
    log(f"export complete in {elapsed:.1f} seconds; rows={counts}")


if __name__ == "__main__":
    main()
