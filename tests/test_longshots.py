"""Sanity tests for the final long-shot campaign helpers."""

import numpy as np
import pytest

from zoo.ls_campaign import (
    asymmetric_smoothed_targets,
    capped_day_weights,
    discrete_survival_targets,
    empirical_bayes_lambda,
    masking_probability,
)


def test_masking_probability_is_frequency_dependent():
    probability = masking_probability(np.array([1, 3, 99]), p0=0.2, alpha=0.5)
    np.testing.assert_allclose(probability, 0.2 / np.sqrt(np.array([2, 4, 100])))
    assert probability[0] > probability[1] > probability[2]


def test_capped_day_weights_select_worst_of_last_four():
    dates = np.repeat(np.arange(1, 7), 2)
    losses = np.array([9, 9, 1, 1, 2, 2, 8, 8, 3, 3, 7, 7], dtype=float)
    weights, days = capped_day_weights(dates, losses, last_days=4, worst_k=1, cap=3)
    assert days == (4,)
    np.testing.assert_array_equal(weights[dates == 4], 3)
    np.testing.assert_array_equal(weights[dates != 4], 1)


def test_asymmetric_smoothing_has_separate_short_schedule():
    labels = np.array([1, 0, 1, 0, 1], dtype=np.float32)
    play = np.array([18_000, 18_000, 9_000, 9_000, 1_000], dtype=np.float32)
    duration = np.array([30_000, 30_000, 10_000, 10_000, 30_000], dtype=np.float32)
    target = asymmetric_smoothed_targets(
        labels, play, duration, long_near=(0.2, 0.1),
        short_near=(0.05, 0.02), far=(0.01, 0.005), width=0.2)
    np.testing.assert_allclose(target, [0.8, 0.1, 0.95, 0.02, 0.99])


def test_empirical_bayes_shrinkage_formula():
    np.testing.assert_allclose(empirical_bayes_lambda(np.array([0, 20, 100]), 20),
                               [0, 0.5, 5 / 6])
    with pytest.raises(ValueError):
        empirical_bayes_lambda(3, 0)


def test_discrete_survival_targets_are_censoring_correct():
    event, risk = discrete_survival_targets(
        np.array([1_000, 9_000, 18_000]), np.array([20_000, 20_000, 20_000]), bins=4)
    np.testing.assert_array_equal(event, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(risk, [[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
