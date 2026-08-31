"""Resume-from-run: continue a prior run from just before iteration N with the
exact state that run had, under current agent code."""

import json
from pathlib import Path

import pytest

from agent.fake_brain import FakeBrain
from harness.cli import build_parser
from harness.loop import ROOT, Loop, LoopConfig

DATA_DIR = ROOT / "data" / "synthetic"


def _loop(tmp_path: Path, name: str, **overrides) -> Loop:
    config = LoopConfig(data_dir=DATA_DIR, run_dir=tmp_path / name,
                        cross_run_path=tmp_path / "CROSS_RUN.md",
                        sigma=overrides.pop("sigma", 0.003), **overrides)
    return Loop(config, FakeBrain("", root=str(ROOT)))


def _state_after(loop: Loop, upto: int) -> dict:
    """Replay-independent view of a loop's state through iteration `upto`,
    recomputed from its journal with the official convergence rule."""
    records = [json.loads(l) for l in (loop.run_dir / "journal.jsonl").read_text().splitlines()]
    by_n = {r["n"]: r for r in records if "n" in r}
    best = by_n[0]["metrics"]["primary"]
    streak = 0
    for k in range(1, upto + 1):
        before = best
        if by_n[k]["accepted"]:
            best = by_n[k]["metrics"]["primary"]
        streak = 0 if best - before > loop.config.epsilon else streak + 1
    return {"best": best, "streak": streak}


def test_resume_reconstructs_state_and_continues(tmp_path):
    original = _loop(tmp_path, "orig", max_iters=3)
    original.run()
    orig_records = (original.run_dir / "journal.jsonl").read_text().splitlines()
    assert len(orig_records) == 4  # baseline + 3 iterations

    resumed = _loop(tmp_path, "resumed", max_iters=3,
                    resume_from=original.run_dir, resume_at=2)
    # reconstruct only (no LLM): call the resume hook directly, then inspect
    resumed.prepare_workspace()
    n = resumed.resume_from_run(original.run_dir, 2)
    assert n == 1
    expected = _state_after(original, 1)
    assert resumed.champion.primary == pytest.approx(expected["best"])
    assert resumed.no_improve_streak == expected["streak"]
    assert set(resumed.nodes) == {"node_000", "node_001"}
    assert resumed.sigma == pytest.approx(original.sigma)
    assert (resumed.run_dir / "nodes" / "001.py").read_text() == \
        (original.run_dir / "nodes" / "001.py").read_text()
    assert (resumed.run_dir / "node_001").is_dir()
    assert not (resumed.run_dir / "nodes" / "002.py").exists()

    lines = (resumed.run_dir / "journal.jsonl").read_text().splitlines()
    assert lines[:2] == orig_records[:2]  # verbatim history
    marker = json.loads(lines[2])
    assert marker["resumed_at"] == 2 and marker["intervention"] is False

    # now let the real run() drive the continuation from scratch (fresh loop)
    resumed2 = _loop(tmp_path, "resumed2", max_iters=3,
                     resume_from=original.run_dir, resume_at=2)
    summary = resumed2.run()
    assert summary["resumed_from"] == str(original.run_dir)
    assert summary["resumed_at"] == 2
    assert "not a designation candidate" in summary["lineage"]
    assert summary["iterations"] == 3
    out = [json.loads(l) for l in (resumed2.run_dir / "journal.jsonl").read_text().splitlines()]
    iteration_ns = [r["n"] for r in out if "n" in r]
    assert iteration_ns == [0, 1, 2, 3]  # history 0-1, marker, then new 2-3
    assert (resumed2.run_dir / "nodes" / "003.py").exists()


def test_resume_refuses_bad_targets(tmp_path):
    original = _loop(tmp_path, "orig", max_iters=2)
    original.run()
    bad = _loop(tmp_path, "bad")
    bad.prepare_workspace()
    with pytest.raises(ValueError):
        bad.resume_from_run(original.run_dir, 0)
    with pytest.raises(ValueError):
        bad.resume_from_run(original.run_dir, 99)
    with pytest.raises(FileNotFoundError):
        bad.resume_from_run(tmp_path / "nowhere", 1)


def test_cli_accepts_resume_flags():
    args = build_parser().parse_args([
        "run", "--data-dir", "x", "--resume-from", "logs/run_a", "--at", "3"])
    assert args.resume_from == Path("logs/run_a") and args.at == 3
