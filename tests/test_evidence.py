"""Tests for the evidence pack: render.py (report generation from journal.jsonl)
and submission.py (submission build + validation)."""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evidence"))

import render  # noqa: E402
import submission  # noqa: E402
from submission import SubmissionError  # noqa: E402


# ---------------------------------------------------------------- fixture

def _rec(n, node, parent, action, hyp, primary, best, accepted, *,
         error=None, recovery=None, intervention=False,
         tin=1000, tout=500, dur=100.0, roles=None, method_selection=None):
    return {
        "n": n,
        "hypothesis": hyp,
        "node_id": node, "parent": parent, "action": action,
        "code_path": f"logs/run_test/nodes/{n:03d}.py",
        "change_summary": f"change for iter {n}",
        "metrics": {"gauc": primary + 0.01, "ndcg5": primary - 0.01, "primary": primary},
        "val_best_so_far": best,
        "accepted": accepted,
        "duration_s": dur,
        "tokens_in": tin, "tokens_out": tout,
        "error": error, "recovery": recovery,
        "intervention": intervention,
        "method_selection": method_selection,
        **({"tokens_by_role": roles} if roles else {}),
    }


@pytest.fixture()
def run_dir(tmp_path):
    """8 iterations: 4 accepted, 2 rejected, 2 errored; 1 intervention.

    Hand-computed aggregates:
      accepted=4 rejected=2 errors=2 interventions=1
      tokens_in = 8*1000 = 8000 ; tokens_out = 8*500 = 4000
      wall = 8*100 = 800 s
      best accepted primary = 0.6150 (iter 8, node_008)
      per-role: proposer in=4000 out=3000 ; parser in=4000 out=1000
    """
    roles_a = {"proposer": {"in": 500, "out": 375}, "parser": {"in": 500, "out": 125}}
    recs = [
        _rec(1, "node_001", "baseline", "draft", "BPR pairwise loss aligns with GAUC",
             0.6050, 0.6050, True, roles=roles_a, method_selection={
                 "diagnosis": "metric-mismatch",
                 "chosen_method_id": "bpr-hybrid",
                 "citation": "Rendle et al., BPR",
                 "why": "Within-user pairs align directly with GAUC.",
                 "rejected": [{
                     "method_id": "item-aggregates",
                     "reason": "Measured dead at 0.6038 primary.",
                 }],
             }),
        _rec(2, "node_002", "node_001", "improve", "finer duration buckets",
             0.6045, 0.6050, False, roles=roles_a),
        _rec(3, "node_003", "node_001", "improve", "early stopping on val GAUC",
             0.6090, 0.6090, True, roles=roles_a),
        _rec(4, "node_004", "node_003", "improve", "DCNv2-lite head",
             0.0, 0.6090, False, error="RuntimeError: shape mismatch",
             recovery="patched", roles=roles_a),
        _rec(5, "node_005", "node_003", "debug", "fix cross-layer dims",
             0.6110, 0.6110, True, roles=roles_a),
        _rec(6, "node_006", "node_005", "improve", "multi-task aux heads",
             0.0, 0.6110, False, error="CUDA OOM", recovery="reverted",
             intervention=True, roles=roles_a),
        _rec(7, "node_007", "node_005", "improve", "item aggregate rates",
             0.6105, 0.6110, False, roles=roles_a),
        _rec(8, "node_008", "node_005", "improve", "seed ensemble of best config",
             0.6150, 0.6150, True, roles=roles_a),
    ]
    d = tmp_path / "run_test"
    d.mkdir()
    with open(d / "journal.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return d


# ---------------------------------------------------------------- render.py

def test_render_outputs_and_aggregates(run_dir):
    report = render.render(run_dir)
    assert (report / "trajectory.png").is_file()
    assert (report / "trajectory.png").stat().st_size > 1000
    assert (report / "results.md").is_file()
    assert (report / "RUNLOG.md").is_file()

    results = (report / "results.md").read_text()
    # final best metrics (iter 8: primary 0.6150, gauc 0.6250, ndcg5 0.6050)
    assert "0.6150" in results
    assert "0.6250" in results
    assert "0.6050" in results
    assert "+0.0134" in results          # 0.6150 - 0.6016
    assert "node_008" in results
    # counts
    assert "| iterations used | 8 / 50 (official cap) |" in results
    assert "| accepted | 4 |" in results
    assert "| rejected | 2 |" in results
    assert "| errors | 2 |" in results
    assert "| human interventions | 1 |" in results
    # tokens and wall clock
    assert "8,000" in results
    assert "4,000" in results
    assert "800 s" in results
    # per-role split: 8 iters * per-iter role tokens
    assert "| proposer | 4,000 | 3,000 |" in results
    assert "| parser | 4,000 | 1,000 |" in results

    runlog = (report / "RUNLOG.md").read_text()
    for n in range(1, 9):
        assert f"| {n} |" in runlog
    assert "BPR pairwise loss" in runlog
    assert "RuntimeError: shape mismatch -> patched" in runlog
    assert "CUDA OOM -> reverted" in runlog
    assert "HUMAN INTERVENTION" in runlog
    assert "## Diagnose → select evidence" in runlog
    assert "### Iteration 1: bpr-hybrid" in runlog
    assert "Diagnosis: metric-mismatch" in runlog
    assert "Rendle et al., BPR" in runlog
    assert "Rejected alternative: item-aggregates" in runlog
    assert "Measured dead at 0.6038 primary" in runlog


def test_render_missing_journal(tmp_path):
    with pytest.raises(FileNotFoundError):
        render.render(tmp_path)


# ---------------------------------------------------------------- submission.py

@pytest.fixture()
def split_file(tmp_path):
    p = tmp_path / "val.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "video_id", "tab", "long_view"])
        for i in range(5):
            w.writerow([f"u{i}", f"v{i}", 1, 0])
    return p


def _write_submission(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)


def _good_rows():
    return [submission.HEADER] + [[i, f"u{i}", f"v{i}", 0.1 * i] for i in range(5)]


def test_build_and_check_ok(tmp_path, split_file):
    preds = tmp_path / "predictions.csv"
    _write_submission(preds, _good_rows())
    out = tmp_path / "submission.csv"
    assert submission.build(preds, split_file, out) == 5
    scores = submission.check(out, split_file)
    assert scores == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_check_bad_header(tmp_path, split_file):
    rows = _good_rows()
    rows[0] = ["rowid", "user", "video", "score"]
    p = tmp_path / "s.csv"
    _write_submission(p, rows)
    with pytest.raises(SubmissionError, match="header"):
        submission.check(p, split_file)


def test_check_row_id_gap(tmp_path, split_file):
    rows = _good_rows()
    rows[3][0] = 5  # gap: 0,1,5,...
    p = tmp_path / "s.csv"
    _write_submission(p, rows)
    with pytest.raises(SubmissionError, match="row_id"):
        submission.check(p, split_file)


def test_check_nan_score(tmp_path, split_file):
    rows = _good_rows()
    rows[2][3] = "nan"
    p = tmp_path / "s.csv"
    _write_submission(p, rows)
    with pytest.raises(SubmissionError, match="NaN"):
        submission.check(p, split_file)


def test_check_wrong_count(tmp_path, split_file):
    rows = _good_rows()[:-1]  # 4 rows instead of 5
    p = tmp_path / "s.csv"
    _write_submission(p, rows)
    with pytest.raises(SubmissionError, match="count mismatch|rows"):
        submission.check(p, split_file)


def test_check_misalignment(tmp_path, split_file):
    rows = _good_rows()
    rows[2][1] = "u99"
    p = tmp_path / "s.csv"
    _write_submission(p, rows)
    with pytest.raises(SubmissionError, match="alignment"):
        submission.check(p, split_file)
