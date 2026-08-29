"""End-to-end contract coverage for the train-and-rank seed ensemble."""

from __future__ import annotations

import csv
import json
import subprocess
import sys

from conftest import ROOT


def _run(out_dir) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "zoo" / "ensemble_node.py"),
            "--data-dir",
            str(ROOT / "data" / "synthetic"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "17",
            "--n-members",
            "3",
            "--member-epochs",
            "1",
            "--member-script",
            "zoo/fm_torch.py",
            "--member-args",
            "--batch-size 256",
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_ensemble_node_contract_history_and_determinism(tmp_path) -> None:
    first_out, second_out = tmp_path / "first", tmp_path / "second"
    _run(first_out)
    _run(second_out)

    assert (first_out / "predictions.csv").read_bytes() == (
        second_out / "predictions.csv"
    ).read_bytes()
    assert (first_out / "metrics.json").read_bytes() == (second_out / "metrics.json").read_bytes()

    metrics = json.loads((first_out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) >= {"gauc", "ndcg5", "primary", "history"}
    assert all(0.0 <= metrics[key] <= 1.0 for key in ("gauc", "ndcg5", "primary"))
    assert len(metrics["history"]) == 4
    assert [entry["stage"] for entry in metrics["history"]] == [
        "member",
        "member",
        "member",
        "ensemble",
    ]
    assert [entry["seed"] for entry in metrics["history"][:-1]] == [17, 18, 19]
    assert metrics["history"][-1]["val_primary"] == metrics["primary"]
    assert all("val_primary" in entry for entry in metrics["history"])

    with (first_out / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert list(rows[0]) == ["row_id", "user_id", "video_id", "score"]
    assert [int(row["row_id"]) for row in rows] == list(range(len(rows)))
    assert all((first_out / f"member_{index:02d}" / "predictions.csv").exists()
               for index in range(3))
