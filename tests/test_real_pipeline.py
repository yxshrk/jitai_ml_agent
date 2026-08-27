"""Real-data pipeline tests. All tests skip when ../KuaiRand-Pure is absent."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT.parent / "KuaiRand-Pure" / "data"

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="../KuaiRand-Pure dataset not present"
)

OFFICIAL_COUNTS = {"train": 1_141_112, "valid": 124_909, "test": 170_588}


@pytest.fixture(scope="module")
def dataset():
    from data.real_loader import load_dataset

    return load_dataset("real")


def test_split_sizes_match_official(dataset):
    for name, expected in OFFICIAL_COUNTS.items():
        assert len(dataset[name]["y"]) == expected, name
        assert dataset[name]["X"].shape[0] == expected, name


def test_aux_targets_exposed_and_consistent(dataset):
    tr = dataset["train"]
    for key in ("click", "like", "play_time_ms", "duration_ms"):
        assert len(tr[key]) == OFFICIAL_COUNTS["train"]
    # long_view definition sanity: watched >= min(duration, 18s) implies label 1
    # for the vast majority of rows (label comes straight from the log column).
    ev = tr["play_time_ms"] >= np.minimum(tr["duration_ms"], 18_000)
    agree = float((ev == (tr["y"] == 1)).mean())
    assert agree > 0.95


def test_vendored_evaluate_reproduces_fm_baseline_score():
    """The vendored official evaluate.py must reproduce the FM baseline's score.

    Prediction file is produced by `uv run python data/make_fm_baseline.py`
    (starter-kit FM, seed 0). Its recorded metrics must be reproduced to 4dp,
    and the primary must sit within seed noise of the published 0.6016.
    """
    path = ROOT / "data" / "real" / "fm_baseline_valid.npz"
    if not path.exists():
        pytest.skip("run `uv run python data/make_fm_baseline.py` first")
    from data.official.evaluate import evaluate

    z = np.load(path)
    r = evaluate(list(z["users"]), list(z["labels"]), list(z["scores"]))
    assert abs(r["GAUC"] - float(z["gauc"])) < 5e-5
    assert abs(r["nDCG@5"] - float(z["ndcg5"])) < 5e-5
    assert abs(r["primary"] - float(z["primary"])) < 5e-5
    assert abs(r["primary"] - 0.6016) < 0.003  # published FM valid primary, seed noise


@pytest.mark.parametrize("script", ["fm_bpr", "fm_feats", "dcn_lite", "mtl"])
def test_zoo_script_end_to_end_subsample(script, tmp_path):
    out = tmp_path / script
    proc = subprocess.run(
        [sys.executable, str(ROOT / "zoo" / f"{script}.py"), "--data-dir", "real",
         "--out-dir", str(out), "--seed", "0", "--subsample", "50000",
         "--epochs", "2", "--patience", "1"],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    metrics = json.loads((out / "metrics.json").read_text())
    for key in ("gauc", "ndcg5", "primary"):
        assert 0.0 <= metrics[key] <= 1.0
    with (out / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and set(rows[0]) == {"row_id", "user_id", "video_id", "score"}
    assert len(rows) == 10_000  # valid share of the 50k subsample
