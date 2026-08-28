"""Loop tests: dry run journal conformance, acceptance rule, convergence,
timeout+fixer recovery, leakage guard."""

import json
from pathlib import Path

import pytest

from agent.fake_brain import FakeBrain, canned_script
from agent import prompts
from agent.brain import parse_method_card_metadata
from harness.cli import build_parser
from harness.loop import ROOT, LeakageError, Loop, LoopConfig, Node, RunResult

DATA_DIR = ROOT / "data" / "synthetic"

RECORD_KEYS = {
    "n", "hypothesis", "node_id", "parent", "action", "code_path", "change_summary",
    "diff", "history", "metrics", "val_best_so_far", "accepted", "duration_s", "tokens_in",
    "tokens_out", "error", "recovery", "intervention", "usd_total", "method_selection",
    "context_mode", "expected_delta", "expected_delta_basis", "realized_delta",
    "verdict_note", "failure_stage", "fixer_eligible",
}

CRASHING_SCRIPT = "import sys\nraise RuntimeError('boom')\n"
SLOW_SCRIPT = '''\
import argparse, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--data-dir")
p.add_argument("--out-dir", type=Path, required=True)
p.add_argument("--seed")
a = p.parse_args()
if os.environ.get("SMOKE_EPOCHS"):
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "predictions.csv").write_text("row_id,user_id,video_id,score\\n")
    (a.out_dir / "metrics.json").write_text(json.dumps({"gauc": .5, "ndcg5": .5, "primary": .5}))
else:
    time.sleep(30)
'''
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


def test_cli_accepts_full_context_mode():
    args = build_parser().parse_args([
        "run", "--data-dir", str(DATA_DIR), "--context-mode", "full", "--dry-run",
    ])
    assert args.context_mode == "full"


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
        assert record["method_selection"]["chosen_method_id"] in loop.brain.method_cards
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
    assert records[1]["method_selection"] == records[0]["method_selection"]
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


def test_full_context_prompt_contains_prior_node_history(tmp_path):
    loop = make_loop(tmp_path, fake_brain(), context_mode="full")
    _seed_champion(loop)
    prior = Node("node_001", "node_000", "improve", "distinctive hypothesis", loop.nodes_dir / "001.py")
    prior.status = "rejected"
    prior.change_summary = "distinctive change summary"
    prior.metrics = {
        "gauc": 0.71, "ndcg5": 0.69, "primary": 0.70,
        "history": [{"epoch": 17, "val_primary": "DISTINCTIVE_HISTORY_LINE"}],
    }
    loop.nodes[prior.node_id] = prior
    prompt = prompts.proposer_user_prompt(
        loop.journal_lines, "improve", "node_000", "# parent",
        context_mode="full", full_context=loop.full_proposer_context(),
    )
    assert "distinctive hypothesis" in prompt
    assert "distinctive change summary" in prompt
    assert "DISTINCTIVE_HISTORY_LINE" in prompt
    assert "outcome: rejected" in prompt


def test_expected_and_realized_delta_recorded(tmp_path):
    brain = fake_brain(scripts=[{
        "hypothesis": "calibrated proposal",
        "expected_delta": 0.0125,
        "code": canned_script("calibrated", 'r["video_id"]', root=str(ROOT)),
    }])
    loop = make_loop(tmp_path, brain, max_iters=1)
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert record["expected_delta"] == pytest.approx(0.0125)
    assert "node_000" in record["expected_delta_basis"]
    assert record["realized_delta"] == pytest.approx(
        record["metrics"]["primary"]
        - json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[0])["metrics"]["primary"]
    )
    assert record["context_mode"] == "compact"


def test_realized_delta_is_null_on_error(tmp_path):
    brain = fake_brain(scripts=[{
        "hypothesis": "predicted but crashed",
        "expected_delta": 0.02,
        "code": CRASHING_SCRIPT,
    }])
    loop = make_loop(tmp_path, brain, max_iters=1)
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert record["expected_delta"] == pytest.approx(0.02)
    assert record["realized_delta"] is None
    assert record["error"]


def test_improve_iteration_records_method_selection(tmp_path):
    brain = fake_brain()
    loop = make_loop(tmp_path, brain, max_iters=1, draft_tiers=())
    loop.run()
    record = json.loads((loop.run_dir / "journal.jsonl").read_text().splitlines()[1])
    assert record["action"] == "improve"
    assert record["method_selection"]["diagnosis"] == "overfit"
    expected = {"no_improve_streak": 0, "n_converge": 3, "iters_left": 0}
    assert brain.selection_streak_states == [expected]
    assert brain.proposal_streak_states == [expected]


def test_reflector_fires_at_three_stagnant_iterations(tmp_path):
    brain = fake_brain(scripts=[{"hypothesis": "leaky", "code": LEAKY_SCRIPT}])
    loop = make_loop(
        tmp_path, brain, max_iters=3, n_converge=10, reflect_every=99,
    )
    loop.run()
    reflector = brain.meter.per_role["fake/fake/reflector"]
    assert reflector["calls"] == 1


class StubbornBrain(FakeBrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector_requests = []

    def select_method(self, journal_lines, parent_history, streak_state,
                      excluded_families=None, enforce_family_exclusion=False):
        self.selector_requests.append({
            "excluded_families": list(excluded_families or []),
            "strict": enforce_family_exclusion,
        })
        self.meter.add("fake/fake/selector", 80, 40)
        return {
            "diagnosis": "overfit",
            "chosen_method_id": "regularization-schedule",
            "citation": "stubborn fake",
            "why": "deliberately violates diversity",
            "rejected": [],
        }


class BprBrain(FakeBrain):
    def select_method(self, journal_lines, parent_history, streak_state,
                      excluded_families=None, enforce_family_exclusion=False):
        self.meter.add("fake/fake/selector", 80, 40)
        return {
            "diagnosis": "metric-mismatch",
            "chosen_method_id": "bpr-hybrid",
            "citation": "measured 0.6048",
            "why": "exercise reference routing",
            "rejected": [],
        }


def test_below_reference_routes_to_debug_then_rejects_if_still_low(tmp_path, monkeypatch):
    scripts = [
        {"hypothesis": "first implementation", "code": "# candidate"},
        {"hypothesis": "debug implementation", "code": "# debugged"},
    ]
    brain = BprBrain("", scripts=scripts, root=str(ROOT))
    loop = make_loop(tmp_path, brain, max_iters=2)
    _seed_champion(loop, primary=0.6000)
    low = RunResult(True, metrics={"gauc": .6020, "ndcg5": .6020, "primary": .6020})
    monkeypatch.setattr(loop, "run_experiment", lambda *args, **kwargs: (low, "full"))

    loop.iterate(1)
    suspect = loop.nodes["node_001"]
    assert suspect.status == "suspect_implementation"
    assert suspect.verdict_note == (
        "below card reference (0.6020 vs 0.6048) — implementation suspected"
    )
    assert loop.next_move()[:2] == ("debug", suspect)
    record = json.loads(loop.journal_path.read_text().splitlines()[0])
    assert record["verdict_note"] == suspect.verdict_note

    loop.iterate(2)
    assert loop.nodes["node_002"].status == "rejected"


def test_draft_diversity_retries_then_overrides_stubborn_selector(tmp_path, monkeypatch):
    brain = StubbornBrain("", scripts=[{"hypothesis": "diverse draft", "code": "# code"}],
                          root=str(ROOT))
    loop = make_loop(tmp_path, brain, max_iters=2)
    _seed_champion(loop)
    prior = Node("node_001", "node_000", "draft", "prior overfit draft", loop.nodes_dir / "001.py")
    prior.code_path.write_text("# prior")
    prior.status = "rejected"
    prior.method_selection = {"chosen_method_id": "regularization-schedule"}
    loop.nodes[prior.node_id] = prior
    result = RunResult(True, metrics={"gauc": .69, "ndcg5": .69, "primary": .69})
    monkeypatch.setattr(loop, "run_experiment", lambda *args, **kwargs: (result, "full"))

    loop.iterate(2)

    assert brain.selector_requests == [
        {"excluded_families": ["overfit"], "strict": False},
        {"excluded_families": ["overfit"], "strict": True},
    ]
    selection = loop.nodes["node_002"].method_selection
    assert selection["harness_override"] is True
    assert selection["chosen_method_id"] == "duration-regime-heads"
    assert not ({"overfit"} & set(parse_method_card_metadata(
        brain.method_cards[selection["chosen_method_id"]]
    )["treats"]))


def test_expected_delta_basis_and_minimal_mutation_are_prompt_contracts():
    prompt = prompts.proposer_user_prompt([], "improve", "node_000", "# parent")
    assert "smallest coherent change" in prompt
    assert "unnecessary rewrites are defects" in prompt
    assert "expected_delta_basis" in prompts.TASK_BRIEF
    assert "specific measured card value or journal line" in prompts.TASK_BRIEF


def test_every_method_card_declares_parseable_reference_primary():
    brain = fake_brain()
    assert len(brain.method_cards) == 20
    for card in brain.method_cards.values():
        assert "- reference_primary:" in card
        metadata = parse_method_card_metadata(card)
        assert metadata["reference_primary"] is None or isinstance(
            metadata["reference_primary"], float
        )


def test_smoke_failure_is_fixer_eligible_and_skips_full_run(tmp_path):
    smoke_failure = '''\
import os, sys
if os.environ.get("SMOKE_EPOCHS") == "1":
    raise RuntimeError("smoke-only failure")
sys.exit(0)
'''
    loop = make_loop(
        tmp_path,
        fake_brain(scripts=[{"hypothesis": "fails smoke", "code": smoke_failure}]),
        max_iters=1,
    )
    loop.run()
    record = json.loads(loop.journal_path.read_text().splitlines()[1])
    assert record["failure_stage"] == "smoke"
    assert record["fixer_eligible"] is True
    assert record["recovery"] == "reverted"
    assert "smoke-only failure" in record["error"]
    assert not (loop.run_dir / "node_001").exists()
    assert "SMOKE_EPOCHS" in prompts.TASK_BRIEF
    assert "cap" in prompts.TASK_BRIEF
