"""Export label-free KuaiRand test-window features in official row order.

This pipeline reads test FEATURES only to produce predictions (the required
submission artifact); labels are never loaded; it is intended to run only at
submission time.
"""

from __future__ import annotations

import argparse
import csv
from array import array
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MIN_DATE, TRAIN_MAX_DATE = 20220408, 20220421
TEST_MIN_DATE, TEST_MAX_DATE = 20220429, 20220508
FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket")
FEATURE_COLUMNS = ("user_id", "video_id", "date", "tab", "duration_ms")
EXPECTED_PURE_TEST_ROWS = 170_588


def dataset_paths(dataset: str) -> tuple[Path, Path, str]:
    suffix = "pure" if dataset == "pure" else "1k"
    raw = ROOT.parent / ("KuaiRand-Pure" if dataset == "pure" else "KuaiRand-1K") / "data"
    out = ROOT / "data" / ("test_features" if dataset == "pure" else "test_features_1k") / "test.npz"
    return raw, out, suffix


def _require_files(raw_dir: Path, suffix: str) -> tuple[Path, Path, Path]:
    paths = (
        raw_dir / f"video_features_basic_{suffix}.csv",
        raw_dir / f"log_standard_4_08_to_4_21_{suffix}.csv",
        raw_dir / f"log_standard_4_22_to_5_08_{suffix}.csv",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required KuaiRand raw file(s): " + ", ".join(missing))
    return paths


def _selected_rows(path: Path) -> Iterator[tuple[str, str, str, str, str]]:
    """Yield only feature cells; test label cells are deliberately never indexed."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        try:
            indices = tuple(header.index(name) for name in FEATURE_COLUMNS)
        except ValueError as exc:
            raise ValueError(f"{path} lacks a required feature column") from exc
        for row in reader:
            if row:
                yield tuple(row[index] for index in indices)  # type: ignore[misc]


def _authors(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        video_index, author_index = header.index("video_id"), header.index("author_id")
        for row in reader:
            if row:
                result[row[video_index]] = row[author_index]
    return result


def _raw_values(row: tuple[str, str, str, str, str], authors: dict[str, str],
                edges: np.ndarray) -> tuple[str, str, str, str, str]:
    user, video, _date, tab, duration = row
    return (user, video, authors.get(video, "UNK"), tab,
            str(int(np.searchsorted(edges, float(duration)))))


def export(raw_dir: Path, out_path: Path, suffix: str) -> int:
    video_path, first_log, second_log = _require_files(raw_dir, suffix)
    authors = _authors(video_path)
    vocabs: list[dict[str, int]] = [dict() for _ in FIELDS]
    durations = array("d")

    # Match export_real_ws.write_npz exactly: train-window encounter-order
    # vocabularies and float64 NumPy duration deciles.
    for log_path in (first_log, second_log):
        for row in _selected_rows(log_path):
            date = int(row[2])
            if TRAIN_MIN_DATE <= date <= TRAIN_MAX_DATE:
                durations.append(float(row[4]))
                raw = (row[0], row[1], authors.get(row[1], "UNK"), row[3], "")
                for index, value in enumerate(raw[:4]):
                    vocabs[index].setdefault(value, len(vocabs[index]))
    if not durations:
        raise ValueError(f"no train-window rows found in {raw_dir}")
    duration_values = np.frombuffer(durations, dtype=np.float64)
    edges = np.quantile(duration_values, np.linspace(0, 1, 11)[1:-1])
    for duration in duration_values:
        value = str(int(np.searchsorted(edges, duration)))
        vocabs[4].setdefault(value, len(vocabs[4]))

    field_dims = np.asarray([len(vocab) + 1 for vocab in vocabs], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int32)
    unknown = [len(vocab) for vocab in vocabs]

    # Count first so the 1K variant can allocate compact arrays instead of
    # retaining millions of Python row objects.
    count = sum(
        TEST_MIN_DATE <= int(row[2]) <= TEST_MAX_DATE
        for path in (first_log, second_log)
        for row in _selected_rows(path)
    )
    X = np.empty((count, len(FIELDS)), dtype=np.int32)
    user_ids = np.empty(count, dtype=np.int64)
    video_ids = np.empty(count, dtype=np.int64)
    dates = np.empty(count, dtype=np.int32)
    position = 0
    # Official data.load() appends these two files in this order, then filters;
    # filtering while streaming therefore preserves the identical row order.
    for log_path in (first_log, second_log):
        for row in _selected_rows(log_path):
            date = int(row[2])
            if not TEST_MIN_DATE <= date <= TEST_MAX_DATE:
                continue
            values = _raw_values(row, authors, edges)
            for index, value in enumerate(values):
                X[position, index] = vocabs[index].get(value, unknown[index]) + offsets[index]
            user_ids[position] = int(row[0])
            video_ids[position] = int(row[1])
            dates[position] = date
            position += 1
    if position != count:
        raise AssertionError(f"test row count changed while exporting: {position} != {count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, user_id=user_ids, video_id=video_ids, date=dates)
    with np.load(out_path, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "user_id", "video_id", "date"}:
            raise AssertionError(f"label-free archive contract violated: {archive.files}")
    if suffix == "pure" and count != EXPECTED_PURE_TEST_ROWS:
        raise AssertionError(f"Pure test rows {count:,} != expected {EXPECTED_PURE_TEST_ROWS:,}")
    return count


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("pure", "1k"), default="pure")
    ap.add_argument("--raw-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main() -> None:
    args = parser().parse_args()
    default_raw, default_out, suffix = dataset_paths(args.dataset)
    count = export(args.raw_dir or default_raw, args.out or default_out, suffix)
    print(f"exported {args.dataset} test features: {count:,} rows -> {args.out or default_out}")


if __name__ == "__main__":
    main()
