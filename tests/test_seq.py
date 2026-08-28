from __future__ import annotations

import numpy as np
import pytest

from zoo.seq_data import assert_no_leakage, build_history_indices
from zoo.seq_sasrec import SequenceRanker, per_user_ranks, rank_average
from zoo.seq_best import BEST_DEFAULTS, parser as best_parser


def tiny_data():
    # User 1 has tied rows 0/1: neither may see the other's outcome.
    users = np.array([1, 1, 1, 2], dtype=np.int64)
    stamps = np.array([100, 100, 200, 150], dtype=np.int64)
    valid_users = np.array([1, 2, 3], dtype=np.int64)
    train_history, valid_history, counts = build_history_indices(users, stamps, valid_users, 2)
    return {
        "train_history": train_history,
        "valid_history": valid_history,
        "valid_history_count": counts,
        "train_stamp": stamps,
        "valid_stamp": np.array([300, 300, 300], dtype=np.int64),
    }


def test_same_timestamp_outcomes_are_excluded():
    data = tiny_data()
    assert data["train_history"][0].tolist() == [-1, -1]
    assert data["train_history"][1].tolist() == [-1, -1]
    assert data["train_history"][2].tolist() == [0, 1]
    assert_no_leakage(data)


def test_future_outcome_is_detected():
    data = tiny_data()
    data["train_history"][0, -1] = 2
    with pytest.raises(AssertionError, match="same-or-later"):
        assert_no_leakage(data)


def test_validation_history_is_train_only_and_never_rolls_forward():
    data = tiny_data()
    # Both validation rows for a user would receive the same train snapshot;
    # no validation row index can be represented in this train-index namespace.
    assert data["valid_history"][0].tolist() == [1, 2]
    assert data["valid_history_count"].tolist() == [3, 1, 0]
    assert np.all(data["valid_history"] < len(data["train_stamp"]))
    bad = tiny_data()
    bad["valid_history"][0, -1] = len(bad["train_stamp"])
    with pytest.raises(AssertionError, match="non-train"):
        assert_no_leakage(bad)


def test_history_shapes_padding_and_truncation():
    users = np.ones(5, dtype=np.int64)
    stamps = np.arange(5, dtype=np.int64)
    train, valid, counts = build_history_indices(users, stamps, np.array([1]), 2)
    assert train.shape == (5, 2)
    assert valid.shape == (1, 2)
    assert train[4].tolist() == [2, 3]
    assert valid[0].tolist() == [3, 4]
    assert counts.tolist() == [5]


def test_sasrec_shape_and_backward_contract():
    import torch

    model = SequenceRanker(20, 10, 11, max_history=4, k=16, blocks=1,
                           dropout=0.2, outcome_marks=True)
    history = torch.tensor([[2, 3, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
    author = torch.where(history > 0, torch.tensor(2), torch.tensor(0))
    duration = torch.where(history > 0, torch.tensor(1), torch.tensor(0))
    outcome = torch.where(history > 0, torch.tensor(2), torch.tensor(0))
    score = model(history, author, duration, outcome,
                  torch.tensor([5, 6, 7]), torch.tensor([2, 3, 4]), torch.tensor([1, 2, 3]))
    assert score.shape == (3,)
    assert torch.isfinite(score).all()
    score.sum().backward()
    assert model.video.weight.grad is not None


def test_per_user_rank_average_contract():
    users = np.array([1, 1, 1, 2, 2])
    left = np.array([0.3, 0.1, 0.2, 8.0, 9.0])
    right = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    assert per_user_ranks(users, left).tolist() == [1.0, 0.0, 0.5, 0.0, 1.0]
    combined = rank_average(users, left, right)
    assert np.allclose(combined, [0.5, 0.5, 0.5, 0.5, 0.5])


def test_best_cli_wires_the_measured_configuration():
    args = best_parser().parse_args(["--out-dir", "/tmp/seq-test-contract"])
    for key, expected in BEST_DEFAULTS.items():
        assert getattr(args, key) == expected
