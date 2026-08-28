"""Sanity checks for the frozen-stack polish runner."""

from pathlib import Path

import numpy as np

from zoo import polish_stack


def test_baseline_cli_defaults() -> None:
    args = polish_stack.baseline_args()
    assert (args.lr, args.step_decay_factor, args.decay_every) == (1e-3, 0.5, 1.0)
    assert (args.dropout, args.embedding_dropout, args.weight_decay) == (0.2, 0.1, 1e-5)
    assert (args.k, args.batch_size, args.recency_half_life, args.bpr_weight) == (16, 8192, 7.0, 0.5)


def test_cli_parses_search_knobs() -> None:
    args = polish_stack.parser().parse_args([
        "--out-dir", "/tmp/test-polish", "--lr", "0.0003", "--step-decay-factor", "0.7",
        "--decay-every", "1.5", "--dropout", "0.4", "--weight-decay", "0.003",
        "--k", "24", "--batch-size", "16384", "--recency-half-life", "12",
        "--bpr-weight", "0.6",
    ])
    assert args.k == 24 and args.batch_size == 16384 and args.decay_every == 1.5
    assert np.isclose(args.lr, 3e-4) and np.isclose(args.bpr_weight, 0.6)


def test_recency_weights_are_normalized_and_monotonic() -> None:
    weights = polish_stack.recency_weights(np.array([20220408, 20220415, 20220421]), 7.0)
    assert np.isclose(weights.mean(), 1.0)
    assert np.all(np.diff(weights) > 0)


def test_real_data_contract_if_present() -> None:
    data = Path(__file__).resolve().parents[1] / "data" / "real_ws"
    if not (data / "train.npz").exists():
        return
    ds = polish_stack.load_validation_only(str(data), subsample=32)
    assert ds["train"]["X"].shape == (32, 5)
    assert tuple(ds["field_names"]) == polish_stack.FROZEN_FIELDS
