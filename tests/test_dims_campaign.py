"""Contract and unit coverage for the under-swept-dimensions campaign."""

from __future__ import annotations

import csv
import json
from argparse import Namespace

import numpy as np
import pytest

from zoo.dims_campaign import PairSampler, load_data, parse_aux, run


def _fixture(path, missing_like: bool = False):
    path.mkdir()
    dims = np.asarray([6, 8, 5, 3, 4], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(dims[:-1])]
    rng = np.random.default_rng(7)
    for stem, rows, date in (("train", 48, 20220420), ("val", 24, 20220424)):
        users = np.repeat(np.arange(6), rows // 6)
        raw = np.column_stack((users, rng.integers(0, dims[1], rows),
                               rng.integers(0, dims[2], rows), rng.integers(0, dims[3], rows),
                               rng.integers(0, dims[4], rows)))
        x = (raw + offsets).astype(np.int32)
        y = ((np.arange(rows) + users) % 3 == 0).astype(np.float32)
        np.savez(path / f"{stem}.npz", X=x, y=y, user=users.astype(np.int64),
                 click=y, play_time_ms=np.where(y, 20_000, 1_000).astype(np.float32),
                 duration_ms=np.full(rows, 20_000, np.float32),
                 hourmin=np.full(rows, 1200, np.int32), date=np.full(rows, date, np.int32),
                 field_dims=dims)
        headers = ["video_id"] + ([] if missing_like else ["like"])
        with (path / f"{stem}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers); writer.writeheader()
            for i in range(rows):
                writer.writerow({"video_id": 100 + i, **({} if missing_like else {"like": int(y[i])})})


def _args(data, out, **updates):
    values = dict(data_dir=str(data), out_dir=str(out), seed=42, epochs=1, batch_size=12,
                  patience_halves=4, max_runtime=30, subsample=None,
                  negatives_per_positive=1, negative_sampling="uniform",
                  aux_tasks="click,effective_view", aux_weights="0.2,0.2",
                  optimizer="adam", lr=1e-3, embedding_lr=0.05)
    values.update(updates)
    return Namespace(**values)


def test_contract_history_predictions_and_determinism(tmp_path):
    data = tmp_path / "data"
    _fixture(data)
    first = run(_args(data, tmp_path / "a"))
    second = run(_args(data, tmp_path / "b"))
    assert len(first["history"]) == 2
    assert first["history"] == second["history"]
    assert json.loads((tmp_path / "a" / "metrics.json").read_text())["history"] == first["history"]
    with (tmp_path / "a" / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    assert set(rows[0]) == {"row_id", "user_id", "video_id", "score"}


def test_aux_parser_and_missing_exported_signal(tmp_path):
    assert parse_aux("play_fraction", "0.3") == (("play_fraction",), (0.3,))
    data = tmp_path / "data"
    _fixture(data, missing_like=True)
    with pytest.raises(ValueError, match="missing requested auxiliary column.*like"):
        load_data(str(data), ("like",))


def test_pair_sampler_counts_and_hard_candidates():
    users = np.array([0, 0, 0, 0, 1, 1])
    labels = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    items = np.array([0, 1, 2, 3, 0, 1])
    sampler = PairSampler(users, labels, items)
    scores = np.array([0.0, 0.1, 0.9, 0.8, 0.0, 0.2])
    pos, neg = sampler.sample(np.random.default_rng(2), 3, "hard", scores)
    assert len(pos) == 6
    # User 0's top half contains item rows 2 and 3; user 1 has only row 5.
    assert set(neg[pos == 0]).issubset({2, 3})
    assert set(neg[pos == 4]) == {5}


@pytest.mark.parametrize("optimizer", ("adagrad", "sparse-adam", "sgd", "split-adagrad-adam"))
def test_optimizer_variants_smoke(tmp_path, optimizer):
    data = tmp_path / f"data-{optimizer}"
    _fixture(data)
    result = run(_args(data, tmp_path / f"out-{optimizer}", optimizer=optimizer,
                       aux_tasks="none", aux_weights="none", lr=0.01))
    assert result["config"]["optimizer"] == optimizer
    assert result["history"]
