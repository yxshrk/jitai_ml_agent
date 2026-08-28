from __future__ import annotations

import numpy as np
import torch

from zoo.sweep_train import DCNLite, PairSampler, official_metrics


def test_dcn_output_heads_and_shape() -> None:
    model = DCNLite(100, fields=10, k=8, hidden=16, dropout=0.2,
                    embedding_dropout=0.3, auxiliary=True)
    output = model(torch.randint(0, 100, (7, 10)))
    assert set(output) == {"main", "click", "effective_view"}
    assert all(value.shape == (7,) for value in output.values())


def test_pair_sampler_keeps_pairs_within_user_and_opposite_label() -> None:
    users = np.array([1, 1, 1, 2, 2, 3])
    labels = np.array([1, 0, 0, 0, 1, 1])
    sampler = PairSampler(users, labels)
    positive, negative = sampler.sample(np.random.default_rng(42))
    assert np.all(users[positive] == users[negative])
    assert np.all(labels[positive] == 1)
    assert np.all(labels[negative] == 0)


def test_official_metrics_wrapper() -> None:
    users = np.array([1, 1, 2, 2])
    labels = np.array([1, 0, 1, 0])
    result = official_metrics(users, labels, np.array([1.0, 0.0, 0.8, 0.1]))
    assert result == {"gauc": 1.0, "ndcg5": 1.0, "primary": 1.0}
