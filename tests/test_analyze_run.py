from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.analyze_run import (
    analyze,
    kendall_tau_b,
    load_validation_data,
    rank_average,
)


def _write_predictions(path: Path, users: np.ndarray, scores: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, score) in enumerate(zip(users, scores)):
            writer.writerow([row_id, int(user), row_id, score])


def _synthetic_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run_synthetic"
    data_dir.mkdir()

    # Three validation users, each with [positive, negative].  Their train-history
    # depths are 1, 2, 3, yielding hand-computable terciles at 5/3 and 7/3.
    train_users = np.array([0, 1, 1, 2, 2, 2], dtype=np.int64)
    val_users = np.repeat(np.arange(3, dtype=np.int64), 2)
    labels = np.tile(np.array([1, 0], dtype=np.float32), 3)
    duration = np.array([10000, 12000, 13000, 17000, 20000, 22000], dtype=np.float32)
    # Encoded tab field has offset 6; raw tabs are 0, 1, 0 by user.
    raw_tabs = np.array([0, 0, 1, 1, 0, 0], dtype=np.int32)
    val_x = np.zeros((6, 4), dtype=np.int32)
    val_x[:, 3] = 6 + raw_tabs
    field_dims = np.array([3, 2, 1, 2], dtype=np.int64)
    np.savez(data_dir / "train.npz", user=train_users)
    np.savez(data_dir / "val.npz", user=val_users, y=labels, duration_ms=duration,
             X=val_x, field_dims=field_dims)

    baseline = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]
    # Reverse only user 0. Per-user taus are [-1, +1, +1], mean = 1/3;
    # one of three top-1 items changes.
    child = [0.1, 0.9, 0.8, 0.2, 0.7, 0.3]
    _write_predictions(run_dir / "calib_seed42" / "predictions.csv", val_users, baseline)
    _write_predictions(run_dir / "node_001" / "predictions.csv", val_users, child)
    for path in [run_dir / "calib_seed42" / "metrics.json",
                 run_dir / "node_001" / "metrics.json"]:
        path.write_text(json.dumps({"gauc": 0, "ndcg5": 0, "primary": 0}))
    records = [
        {"n": 0, "node_id": "node_000", "parent": "baseline", "accepted": True,
         "metrics": {"gauc": 1, "ndcg5": 1, "primary": 1}},
        {"n": 1, "node_id": "node_001", "parent": "node_000", "accepted": False,
         "metrics": {"gauc": 2 / 3, "ndcg5": 0, "primary": 0}},
    ]
    run_dir.mkdir(exist_ok=True)
    (run_dir / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run_dir, data_dir


def _segment(analysis, node: str, segment: str) -> dict:
    return next(row for row in analysis.segment_rows
                if row["node"] == node and row["segment"] == segment)


def test_segment_metrics_match_hand_computation(tmp_path: Path) -> None:
    run_dir, data_dir = _synthetic_fixture(tmp_path)
    analysis = analyze(run_dir, load_validation_data(data_dir))

    baseline_short = _segment(analysis, "node_000", "duration<=18000ms")
    child_short = _segment(analysis, "node_001", "duration<=18000ms")
    discount_second = 1 / math.log2(3)

    assert baseline_short["metrics"] == {"gauc": 1.0, "ndcg5": 1.0, "primary": 1.0}
    assert child_short["metrics"]["gauc"] == 0.5
    assert child_short["metrics"]["ndcg5"] == (discount_second + 1.0) / 2
    expected_primary = (0.5 + (discount_second + 1.0) / 2) / 2
    assert child_short["metrics"]["primary"] == expected_primary
    assert child_short["delta"] == expected_primary - 1.0

    # Tab 1 consists solely of user 1, whose ranking remains perfect.
    assert _segment(analysis, "node_001", "tab=1")["metrics"]["primary"] == 1.0
    # History terciles put user 0 in low, user 1 in mid, and user 2 in high.
    assert _segment(analysis, "node_001", "history:low<=1.66667")["metrics"]["gauc"] == 0.0


def test_kendall_and_top1_change_are_hand_computable(tmp_path: Path) -> None:
    run_dir, data_dir = _synthetic_fixture(tmp_path)
    analysis = analyze(run_dir, load_validation_data(data_dir))
    change = analysis.ranking_changes[0]

    assert kendall_tau_b([0.9, 0.1], [0.1, 0.9]) == -1.0
    assert kendall_tau_b([1.0, 1.0], [2.0, 2.0]) == 1.0
    assert kendall_tau_b([1.0, 1.0], [2.0, 1.0]) == 0.0
    assert change.tau == 1 / 3
    assert change.top1_changed_pct == 100 / 3
    assert change.high_change is True


def test_rank_average_and_ensemble_metrics_match_hand_computation(tmp_path: Path) -> None:
    run_dir, data_dir = _synthetic_fixture(tmp_path)
    validation = load_validation_data(data_dir)
    analysis = analyze(run_dir, validation)

    baseline = next(node.scores for node in analysis.nodes if node.node_id == "node_000")
    child = next(node.scores for node in analysis.nodes if node.node_id == "node_001")
    combined = rank_average(validation.users, [baseline, child])
    # User 0 is tied after averaging opposite ranks; the other users remain ordered.
    np.testing.assert_array_equal(combined, [-1.5, -1.5, -1.0, -2.0, -1.0, -2.0])

    best_plus_rejected = next(
        result for result in analysis.ensembles
        if result.name == "best + high-change rejected nodes"
    )
    # GAUC: mean(.5, 1, 1) = 5/6. Stable tie order leaves nDCG perfect.
    assert best_plus_rejected.metrics["gauc"] == 5 / 6
    assert best_plus_rejected.metrics["ndcg5"] == 1.0
    assert best_plus_rejected.metrics["primary"] == pytest.approx(11 / 12)
    assert best_plus_rejected.delta == pytest.approx(-1 / 12)
