"""Safety and contract tests for the validation-only audit campaign."""

from __future__ import annotations

import csv
import json
from argparse import Namespace

import numpy as np
import pytest

from zoo.audit_campaign import _aligned_times, load_validation_only, run


def _fixture(path, forbidden_val_date: bool = False):
    path.mkdir()
    dims = np.asarray([4, 5, 4, 3, 4], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(dims[:-1])]
    rng = np.random.default_rng(7)
    for stem, rows, date in (("train", 36, 20220420),
                              ("val", 18, 20220429 if forbidden_val_date else 20220424)):
        users = np.repeat(np.arange(6), rows // 6)
        raw = np.column_stack([users % dims[0], rng.integers(0, dims[1], rows),
                               rng.integers(0, dims[2], rows), rng.integers(0, dims[3], rows),
                               rng.integers(0, dims[4], rows)])
        X = (raw + offsets).astype(np.int32)
        y = ((np.arange(rows) + users) % 3 == 0).astype(np.float32)
        np.savez(path / f"{stem}.npz", X=X, y=y, user=users.astype(np.int64),
                 click=y, play_time_ms=np.full(rows, 1000, np.float32),
                 duration_ms=np.full(rows, 20_000, np.float32),
                 hourmin=np.full(rows, 1200, np.int32),
                 date=np.full(rows, date, np.int32), field_dims=dims)
        with (path / f"{stem}.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(("user_id", "video_id"))
            writer.writerows((int(u), int(i % 9)) for i, u in enumerate(users))


def test_control_contract_history_and_determinism(tmp_path):
    data = tmp_path / "data"
    _fixture(data)

    def execute(out):
        return run(Namespace(data_dir=str(data), out_dir=str(out), seed=42, epochs=1,
                             batch_size=12, lr=1e-3, k=4, patience_halves=4,
                             subsample=None, lambda_weight=0.0, duration_heads=False,
                             tab_bias=False,
                             metadata_crosses=False, session_features=False,
                             raw_data_dir=None))

    first = execute(tmp_path / "first")
    second = execute(tmp_path / "second")
    assert len(first["history"]) == 2
    assert first["history"] == second["history"]
    assert first["primary"] == second["primary"]
    saved = json.loads((tmp_path / "first" / "metrics.json").read_text())
    assert saved["history"] == first["history"]
    with (tmp_path / "first" / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 18
    assert set(rows[0]) == {"row_id", "user_id", "video_id", "score"}


def test_loader_rejects_test_window_date(tmp_path):
    data = tmp_path / "data"
    _fixture(data, forbidden_val_date=True)
    with pytest.raises(ValueError, match="forbidden date"):
        load_validation_only(str(data))


def test_duplicate_session_join_uses_occurrence_order(tmp_path):
    export = tmp_path / "val.csv"
    raw = tmp_path / "raw.csv"
    headers = ("user_id", "video_id", "date", "hourmin")
    with export.open("w", newline="") as fh:
        writer = csv.writer(fh); writer.writerow(headers)
        writer.writerows(((1, 9, 20220422, 1200), (1, 9, 20220422, 1200)))
    with raw.open("w", newline="") as fh:
        writer = csv.writer(fh); writer.writerow((*headers, "time_ms"))
        writer.writerows(((1, 9, 20220422, 1200, 100), (1, 9, 20220422, 1200, 200)))
    assert _aligned_times(export, raw, 2, 20220428).tolist() == [100, 200]
