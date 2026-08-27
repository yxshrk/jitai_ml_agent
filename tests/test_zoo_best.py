"""zoo/best.py and zoo/dcn_feats.py obey the CONTRACTS.md section-3 interface."""

import csv
import json
import subprocess
import sys

from conftest import ROOT

DATA_DIR = ROOT / "data" / "synthetic"


def _run(script: str, out_dir, extra=()):
    subprocess.run(
        [sys.executable, str(ROOT / "zoo" / script), "--data-dir", str(DATA_DIR),
         "--out-dir", str(out_dir), "--seed", "42", "--epochs", "3", *extra],
        check=True, cwd=ROOT, capture_output=True)


def _check_outputs(out_dir):
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert set(metrics) >= {"gauc", "ndcg5", "primary"}
    assert 0.0 <= metrics["primary"] <= 1.0
    with (out_dir / "predictions.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and set(rows[0]) == {"row_id", "user_id", "video_id", "score"}
    return metrics


def test_best_contract_and_determinism(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _run("best.py", a)
    _run("best.py", b)
    ma, mb = _check_outputs(a), _check_outputs(b)
    keys = ("gauc", "ndcg5", "primary")  # runtime_s legitimately varies
    assert {k: ma[k] for k in keys} == {k: mb[k] for k in keys}


def test_dcn_feats_flags(tmp_path):
    out = tmp_path / "o"
    _run("dcn_feats.py", out, extra=["--cross-layers", "1", "--hidden", "32",
                                     "--aux-weight", "0.1", "--item-agg"])
    _check_outputs(out)
