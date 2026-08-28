"""Loop tests: dry run journal conformance, acceptance rule, convergence,
timeout+fixer recovery, leakage guard."""

import json
from pathlib import Path

import pytest

from agent.fake_brain import FakeBrain, canned_script
from agent import prompts
from harness.loop import ROOT, LeakageError, Loop, LoopConfig, Node, RunResult

DATA_DIR = ROOT / "data" / "synthetic"

RECORD_KEYS = {
    "n", "hypothesis", "node_id", "parent", "action", "code_path", "change_summary",
    "diff", "history", "metrics", "val_best_so_far", "accepted", "duration_s", "tokens_in",
    "tokens_out", "error", "recovery", "intervention", "usd_total", "method_selection",
}

CRASHING_SCRIPT = "import sys\nraise RuntimeError('boom')\n"
SLOW_SCRIPT = "import time\ntime.sleep(30)\n"
LEAKY_SCRIPT = (
    "import argparse\n"
    "open('data/test/test.csv')\n"
)


def make_loop(tmp_path: Path, brain, **overrides) -> Loop:
    config = LoopConfig(data_dir=DATA_DIR, run_dir=tmp_path / "run", sigma=overrides.pop("sigma", 0.003),
                        **overrides)
    return Loop(config, brain)


def fake_brain(scripts=None, fixes=None) -> FakeBrain:
    return FakeBrain("", scripts=scripts, fixes=fixes, root=str(ROOT))


def test_dry_run_journal_conforms_to_contract(tmp_path):
    loop = make_loop(tmp_path, fake_brain(), max_iters=3)
    summary = loop.run()
    assert summary["iterations"] == 3
    lines = (loop.run_dir / "journal.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["action"] == "reproduce_baseline"
    lines = lines[1:]  # iteration records
    assert len(lines) == 3
    for i, line in enumerate(lines, start=1):
        record = json.loads(line)
        assert set(record) == RECORD_KEYS
        assert record["n"] == i
        assert record["node_id"] == f"node_{i:03d}"
        assert record["action"] in ("draft", "debug", "improve")
        assert isinstance(record["accepted"], bool)
        assert set(record["metrics"]) == {"gauc", "ndcg5", "primary"}
        assert isinstance(record["intervention"], bool) and not record["intervention"]
        assert record["method_selection"]["chosen_method_id"] == "regularization-schedule"
    # best node artifacts exist
    assert (loop.run_dir / "nodes" / "001.py").exists()
    assert summary["best_metrics"]["primary"] > 0


def _seed_champion(loop: Loop, primary: float = 0.70) -> None:
    node = Node("node_000", "baseline", "draft", "baseline", loop.nodes_dir / "000.py")
    loop.nodes_dir.mkdir(parents=True, exist_ok=True)
    node.code_path.write_text("# baseline")
    node.metrics = {"gauc": primary, "ndcg5": primary, "primary": primary}
    node.primary = primary
    node.status = "accepted"
    loop.nodes["node_000"] = node
    loop.champion = node
    loop.sigma = 0.003


def test_acceptance_accepts_above_two_sigma(tmp_path):
    loop = make_loop(tmp_path, fake_brain())
    _seed_champion(loop)
    node = Node("node_001", "node_000", "improve", "h", loop.nodes_dir / "001.py")
    accepted, note = loop.acceptance(node, {"gauc": 0, "ndcg5": 0, "primary": 0.71})  # delta 0.01 > 0.006
    assert accepted and note is None


def test_acceptance_rejects_nonpositive_delta(tmp_path):
    loop = make_loop(tmp_path, fake_brain())
    _seed_champion(loop)
    node = Node("node_001", "node_000", "improve", "h", loop.nodes_dir / "001.py")
    accepted, _ = loop.acceptance(node, {"gauc": 0, "ndcg5": 0, "primary": 0.699})
    assert not accepted


def test_acceptance_grey_zone_confirm(tmp_path, monkeypatch):
    loop = make_loop(tmp_path, fake_brain())
    _seed_champion(loop)
    node = Node("node_001", "node_000", "improve", "h", loop.nodes_dir / "001.py")
    node.code_path.write_text("# candidate")

    # confirm run agrees -> mean delta 0.004 >= sigma -> accept
    monkeypatch.setattr(loop, "run_script", lambda *a, **k: RunResult(
        True, metrics={"gauc": 0, "ndcg5": 0, "primary": 0.704}))
    accepted, note = loop.acceptance(node, {"gauc": 0, "ndcg5": 0, "primary": 0.704})
    assert accepted and "confirm passed" in note

    # confirm run collapses -> mean delta below sigma -> reject
    monkeypatch.setattr(loop, "run_script", lambda *a, **k: RunResult(
        True, metrics={"gauc": 0, "ndcg5": 0, "primary": 0.698}))
    accepted, note = loop.acceptance(node, {"gauc": 0, "ndcg5": 0, "primary": 0.704})
    assert not accepted and "confirm failed" in note


def test_sigma_floor_applies(tmp_path):
    loop = make_loop(tmp_path, fake_brain(), sigma=0.0001)
    _seed_champion(loop)
    loop.sigma = 0.0001  # 2*sigma < 0.002 -> floor 0.002 governs
    node = Node("node_001", "node_000", "improve", "h", loop.nodes_dir / "001.py")
    accepted, _ = loop.acceptance(node, {"gauc": 0, "ndcg5": 0, "primary": 0.703})
    assert accepted  # 0.003 >= floor 0.002


def test_convergence_detection(tmp_path):
    # OFFICIAL rule: converged after N=3 consecutive completed iterations whose
    # best-so-far improvement is <= epsilon; any >epsilon improvement resets.
    loop = make_loop(tmp_path, fake_brain())
    loop.no_improve_streak = 3
    assert loop.converged()
    loop.no_improve_streak = 2
    assert not loop.converged()
    loop.no_improve_streak = 0
    assert not loop.converged()


def test_crash_then_fixer_patches(tmp_path):
    good = canned_script("fixed", 'r["video_id"]', root=str(ROOT))
    brain = fake_brain(
        scripts=[{"hypothesis": "crashes first", "code": CRASHING_SCRIPT}],
        fixes=[good],
    )
    loop = make_loop(tmp_path, brain, max_iters=1)
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert record["recovery"] == "patched"
    assert record["metrics"]["primary"] > 0
    assert loop.nodes["node_001"].status in ("accepted", "rejected")


def test_timeout_marks_reverted_without_fix(tmp_path):
    brain = fake_brain(scripts=[{"hypothesis": "too slow", "code": SLOW_SCRIPT}], fixes=[])
    loop = make_loop(tmp_path, brain, max_iters=1, timeout_s=2)
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert record["recovery"] == "reverted"
    assert "timeout" in record["error"]
    assert not record["accepted"]


def test_leakage_guard_rejects_test_path(tmp_path):
    brain = fake_brain(scripts=[{"hypothesis": "tries to peek", "code": LEAKY_SCRIPT}])
    loop = make_loop(tmp_path, brain, max_iters=1)
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert not record["accepted"]
    assert "leakage guard" in record["error"]


def test_workspace_excludes_test_split(tmp_path):
    loop = make_loop(tmp_path, fake_brain())
    loop.prepare_workspace()
    names = {p.name for p in loop.workspace.iterdir()}
    assert names == {"train.csv", "val.csv"}


def test_workspace_leakage_assertion_fires(tmp_path):
    loop = make_loop(tmp_path, fake_brain())
    loop.workspace.mkdir(parents=True)
    (loop.workspace / "test.csv").write_text("x")
    with pytest.raises(LeakageError):
        loop.prepare_workspace()


def test_debug_policy_after_persistent_failure(tmp_path):
    # iteration 1 fails (no fix); iteration 2 must be a debug of node_001
    good = canned_script("debug-fix", 'r["video_id"]', root=str(ROOT))
    brain = fake_brain(
        scripts=[
            {"hypothesis": "crashes", "code": CRASHING_SCRIPT},
            {"hypothesis": "debugged version", "code": good},
        ],
        fixes=[],
    )
    loop = make_loop(tmp_path, brain, max_iters=2)
    loop.run()
    records = [json.loads(l) for l in (loop.run_dir / "journal.jsonl").read_text().splitlines()]
    records = [r for r in records if r.get("action") != "reproduce_baseline"]
    assert records[0]["recovery"] == "reverted"
    assert records[1]["action"] == "debug"
    assert records[1]["parent"] == "node_001"
    assert records[0]["method_selection"] is not None
    assert records[1]["method_selection"] is None
    assert len(brain.selection_streak_states) == 1


def test_streak_state_reaches_selector_and_proposer_prompts():
    streak = {"no_improve_streak": 2, "n_converge": 3, "iters_left": 1}
    selector = prompts.selector_user_prompt(
        "### card: Card\n- status: untried", ["node_001 rejected"], [], streak
    )
    proposer = prompts.proposer_user_prompt(
        ["node_001 rejected"], "improve", "node_000", "# parent",
        method_selection={
            "diagnosis": "overfit",
            "chosen_method_id": "card",
            "why": "highest expected gain",
        },
        selected_method_card="### card: Card\n- status: untried",
        streak_state=streak,
    )
    for prompt in (selector, proposer):
        assert "'no_improve_streak': 2" in prompt
        assert "'n_converge': 3" in prompt
        assert "'iters_left': 1" in prompt
        assert prompts.CONVERGENCE_PRESSURE in prompt
    assert "## Selected method (implement THIS)" in proposer
