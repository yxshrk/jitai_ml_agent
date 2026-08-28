import csv
from pathlib import Path

import numpy as np
import pytest

from zoo.hist_campaign import (_group_loo_rates, add_affinity_features,
                               load_workspace)


def _write_split(root: Path, name: str, dates, ys):
    n = len(ys)
    X = np.column_stack((np.arange(n) % 2, np.arange(n) % 3 + 3,
                         np.arange(n) % 2 + 6, np.zeros(n), np.zeros(n))).astype(np.int32)
    np.savez(root / f"{name}.npz", X=X, y=np.asarray(ys, np.float32),
             user=np.arange(n, dtype=np.int64) % 2, click=np.zeros(n, np.float32),
             play_time_ms=np.zeros(n, np.float32), duration_ms=np.arange(n) * 1000 + 1000,
             hourmin=np.full(n, 1200, np.int32), date=np.asarray(dates, np.int32),
             field_dims=np.asarray([3, 4, 3, 2, 2]))
    with (root / ("train.csv" if name == "train" else "val.csv")).open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(("video_id",)); w.writerows([[100 + i] for i in range(n)])


def test_loader_rejects_out_of_window_dates(tmp_path):
    _write_split(tmp_path, "train", [20220421], [1])
    _write_split(tmp_path, "val", [20220429], [0])
    with pytest.raises(ValueError, match="validation data outside"):
        load_workspace(str(tmp_path))


def test_affinity_train_value_is_leave_one_out():
    u = np.array([0, 0, 0])
    key = np.array([7, 7, 8])
    y = np.array([1.0, 0.0, 0.0])
    rt, rv = _group_loo_rates(u, key, np.array([0]), np.array([7]), y, smoothing=10)
    # Changing a row's own label cannot directly remain in its pair numerator.
    # For the positive first row, pair LOO positive count is zero.
    prior0 = (y.sum() - y[0]) / 2
    assert rt[0] == pytest.approx((0 + 10 * prior0) / 11)
    assert rv[0] == pytest.approx((1 + 10 * y.mean()) / 12)


def test_affinity_adds_three_within_user_fields():
    tr = {"X": np.array([[0, 2, 4, 0, 8], [0, 3, 5, 1, 8]]),
          "users": np.array([0, 0]), "y": np.array([1.0, 0.0]),
          "duration_ms": np.array([1000.0, 2000.0]), "tab": np.array([0, 1])}
    va = {"X": np.array([[0, 2, 4, 0, 8], [0, 3, 5, 1, 8]]),
          "users": np.array([0, 0]), "y": np.array([0.0, 1.0]),
          "duration_ms": np.array([1000.0, 2000.0]), "tab": np.array([0, 1])}
    ds = {"train": tr, "valid": va, "field_dims_total": 10}
    add_affinity_features(ds)
    assert ds["train"]["X"].shape[1] == 8
    assert ds["valid"]["X"].shape[1] == 8
    assert ds["field_dims_total"] > 10
