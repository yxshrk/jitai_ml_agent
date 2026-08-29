"""Cheap contract tests for the label-free final test pipeline (no training)."""

from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from data.export_test_features import EXPECTED_PURE_TEST_ROWS, _selected_rows
from evidence.submission import SubmissionError, check
from tools import predict_test

ROOT = Path(__file__).resolve().parents[1]
PURE_RAW = ROOT.parent / "KuaiRand-Pure" / "data"


def test_real_pure_export_matches_official_test_filter_order_if_present() -> None:
    """Exercise the starter-kit data.load ordering, selecting features only."""
    archive_path = ROOT / "data" / "test_features" / "test.npz"
    raw_paths = (
        PURE_RAW / "log_standard_4_08_to_4_21_pure.csv",
        PURE_RAW / "log_standard_4_22_to_5_08_pure.csv",
    )
    if not archive_path.is_file() or not all(path.is_file() for path in raw_paths):
        return
    expected = [
        (int(row[0]), int(row[1]), int(row[2]))
        for path in raw_paths
        for row in _selected_rows(path)
        if 20220429 <= int(row[2]) <= 20220508
    ]
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == {"X", "user_id", "video_id", "date"}
        actual = list(zip(archive["user_id"], archive["video_id"], archive["date"]))
    assert len(actual) == EXPECTED_PURE_TEST_ROWS
    assert actual == expected


def test_train_loader_opens_train_npz_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        data_dir / "train.npz",
        X=np.zeros((2, 5), dtype=np.int32),
        y=np.asarray([0, 1], dtype=np.float32),
        user=np.asarray([1, 2], dtype=np.int64),
        date=np.asarray([20220408, 20220421], dtype=np.int32),
        field_dims=np.ones(5, dtype=np.int64),
    )
    (data_dir / "val.npz").write_bytes(b"must never be opened")
    train = predict_test.load_train_only(data_dir)
    assert train["X"].shape == (2, 5)
    assert train["y"].tolist() == [0.0, 1.0]


def test_npz_submission_check_and_member_modes(tmp_path: Path) -> None:
    features = tmp_path / "test.npz"
    np.savez(
        features,
        X=np.zeros((2, 5), dtype=np.int32),
        user_id=np.asarray([7, 7]),
        video_id=np.asarray([8, 9]),
        date=np.asarray([20220429, 20220508]),
    )
    submission = tmp_path / "submission.csv"
    with submission.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "user_id", "video_id", "score"))
        writer.writerow((0, 7, 8, 0.25))
        writer.writerow((1, 7, 9, 0.75))
    assert check(submission, features) == [0.25, 0.75]
    assert predict_test.parse_members("seed46,74,93") == [46, 74, 93]
    args = Namespace(dataset="pure", members="46,74", single_member=True,
                     member_args="", data_dir=None, test_features=None,
                     out=Path("submission.csv"))
    assert predict_test.parse_members(args.members)[:1] == [46]


def test_submission_checker_rejects_test_labels(tmp_path: Path) -> None:
    features = tmp_path / "bad.npz"
    np.savez(features, user_id=np.asarray([1]), video_id=np.asarray([2]),
             y=np.asarray([1]))
    submission = tmp_path / "submission.csv"
    submission.write_text("row_id,user_id,video_id,score\n0,1,2,0.5\n")
    with pytest.raises(SubmissionError, match="label-free"):
        check(submission, features)


def test_one_k_tuned_defaults() -> None:
    args = predict_test._member_config("1k", "")
    assert (args.lr, args.dropout, args.weight_decay, args.k, args.epochs) == (
        0.00168, 0.21, 0.000037, 24, 6,
    )
