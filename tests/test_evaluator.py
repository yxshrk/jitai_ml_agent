import subprocess
import sys

import numpy as np

from harness.evaluate_provisional import evaluate


def test_hand_computed_metrics() -> None:
    # User A: labels [1,0,1], ranked labels are [1,0,1]. One of the two
    # positive/negative pairs is correct, so AUC=1/2; DCG=1+1/log2(4).
    # User B has zero positives: excluded from GAUC, included as nDCG=0.
    users = np.array(["A", "A", "A", "B", "B"])
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.95, 0.9, 0.8, 0.7, 0.1])
    result = evaluate(users, labels, scores)
    user_a_ndcg = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    expected_ndcg = user_a_ndcg / 2
    assert result["gauc"] == 0.5
    np.testing.assert_allclose(result["ndcg5"], expected_ndcg)
    np.testing.assert_allclose(result["primary"], (0.5 + expected_ndcg) / 2)


def test_gauc_uses_positive_weights_and_half_credit_for_ties() -> None:
    # A has AUC 1 with weight 1. B has AUC 3/4 with weight 2.
    users = np.array(["A", "A", "B", "B", "B", "B"])
    labels = np.array([1, 0, 1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.9, 0.4, 0.8, 0.1])
    result = evaluate(users, labels, scores)
    expected_b_ndcg = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    np.testing.assert_allclose(result["gauc"], (1 + 2 * 0.75) / 3)
    np.testing.assert_allclose(result["ndcg5"], (1 + expected_b_ndcg) / 2)

    tied = evaluate([0, 0], [1, 0], [0.5, 0.5])
    assert tied["gauc"] == 0.5


def test_check_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "harness/evaluate_provisional.py", "--check"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "check passed" in completed.stdout
