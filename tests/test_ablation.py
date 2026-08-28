from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from zoo.ablate_fields import LEVEL_ADDITIONS, field_names, load_validation_only


ROOT = Path(__file__).resolve().parents[1]


def test_field_levels_are_exactly_cumulative() -> None:
    expected_counts = (5, 7, 10, 15, 21, 24)
    previous: tuple[str, ...] = ()
    for level, expected_count in enumerate(expected_counts):
        current = field_names(level)
        assert len(current) == expected_count
        assert current[:len(previous)] == previous
        assert current[-len(LEVEL_ADDITIONS[level]):] == LEVEL_ADDITIONS[level]
        assert len(set(current)) == len(current)
        previous = current


def _write_split(base: Path, stem: str, dates: np.ndarray) -> None:
    rows = len(dates)
    x = np.column_stack((np.arange(rows) % 3, np.arange(rows) % 4,
                         np.arange(rows) % 2, np.arange(rows) % 2,
                         np.arange(rows) % 3)).astype(np.int64)
    y = (np.arange(rows) % 2).astype(np.float32)
    np.savez(base / f"{stem}.npz", X=x, y=y, user=x[:, 0], click=y,
             play_time_ms=np.full(rows, 1000, dtype=np.float32),
             duration_ms=np.full(rows, 20_000, dtype=np.float32),
             hourmin=np.full(rows, 1200, dtype=np.int32), date=dates,
             field_dims=np.asarray([4, 5, 3, 2, 4], dtype=np.int64))
    with (base / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("video_id", "author_id"))
        writer.writeheader()
        for index in range(rows):
            writer.writerow({"video_id": index % 4, "author_id": index % 2})


def test_loader_rejects_test_window_dates(tmp_path: Path) -> None:
    _write_split(tmp_path, "train", np.asarray([20220408, 20220421]))
    _write_split(tmp_path, "val", np.asarray([20220428, 20220429]))
    with pytest.raises(ValueError, match="forbidden date"):
        load_validation_only(str(tmp_path))


def test_level0_end_to_end_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    command = [
        "uv", "run", "python", "zoo/ablate_fields.py",
        "--data-dir", "data/real_ws", "--out-dir", str(out_dir),
        "--field-level", "0", "--seed", "42", "--subsample", "4000",
        "--epochs", "1", "--patience-halves", "2", "--batch-size", "512",
        "--max-runtime", "60",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=90, check=False)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["field_level"] == 0
    assert metrics["n_fields"] == 5
    assert metrics["field_names"] == list(field_names(0))
    assert len(metrics["history"]) == 2
    assert {"gauc", "ndcg5", "primary"} <= metrics.keys()
    header = (out_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "row_id,user_id,video_id,score"
