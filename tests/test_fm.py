import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from data.synthetic.generate import generate
from harness.evaluate_provisional import evaluate


def test_fm_interface_is_deterministic_and_beats_random(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    generate(data_dir, seed=42)

    command = [
        sys.executable, "zoo/fm_torch.py", "--data-dir", str(data_dir),
        "--out-dir", str(first_out), "--seed", "42",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    command[command.index(str(first_out))] = str(second_out)
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert (first_out / "predictions.csv").read_bytes() == (second_out / "predictions.csv").read_bytes()
    assert (first_out / "metrics.json").read_bytes() == (second_out / "metrics.json").read_bytes()
    metrics = json.loads((first_out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"gauc", "ndcg5", "primary"}

    with (first_out / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    with (data_dir / "val.csv").open(newline="", encoding="utf-8") as handle:
        validation = list(csv.DictReader(handle))
    assert list(predictions[0]) == ["row_id", "user_id", "video_id", "score"]
    assert len(predictions) == len(validation)
    assert [int(row["row_id"]) for row in predictions] == list(range(len(predictions)))

    users = np.asarray([int(row["user_id"]) for row in validation])
    labels = np.asarray([int(row["long_view"]) for row in validation])
    random_scores = np.random.default_rng(42).random(len(labels))
    random_metrics = evaluate(users, labels, random_scores)
    delta = metrics["primary"] - random_metrics["primary"]
    print(f"FM primary={metrics['primary']:.6f}, random primary={random_metrics['primary']:.6f}, delta={delta:.6f}")
    assert delta > 0.05
