"""Budget: pricing, rounding, unknown-model fallback, hard-cap refusal, loop stop."""

import json

import pytest

from agent.budget import Budget, BudgetExhausted, cost_usd, rates_for
from agent.fake_brain import FakeBrain
from harness.loop import ROOT, Loop, LoopConfig

DATA_DIR = ROOT / "data" / "synthetic"


def test_rates_exact_prefix_and_fallback():
    assert rates_for("gpt-5.6-sol") == (15.0, 60.0)
    assert rates_for("gpt-5.5-turbo-x") == (12.0, 48.0)      # prefix pattern
    assert rates_for("gpt-5.4-mini") == (1.0, 4.0)            # exact beats gpt-5.4*
    assert rates_for("claude-haiku-4-5-20251001") == (0.8, 4.0)
    assert rates_for("mystery-model-9000") == (15.0, 75.0)    # most expensive row


def test_cost_rounds_up():
    # 1 in + 1 out token of the cheapest model still costs a whole 0.0001
    assert cost_usd("gpt-5.4-nano", 1, 1) == 0.0001
    assert cost_usd("gpt-5.6-sol", 1_000_000, 1_000_000) == 75.0


def test_ledger_persists_and_precheck_refuses(tmp_path):
    ledger = tmp_path / "spend.json"
    budget = Budget(cap_usd=1.0, ledger_path=ledger)
    budget.record("openai", "gpt-5.6-sol", 10_000, 10_000)  # $0.75
    assert Budget(cap_usd=1.0, ledger_path=ledger).total() == 0.75  # cross-instance
    # worst case 10k in + 10k out = $0.75 -> 1.50 > cap 1.0 -> refused
    with pytest.raises(BudgetExhausted):
        budget.precheck("gpt-5.6-sol", 10_000, 10_000)
    budget.precheck("gpt-5.4-nano", 10_000, 10_000)  # cheap call still allowed
    entries = json.loads(ledger.read_text())["entries"]
    assert entries[0]["model"] == "gpt-5.6-sol" and entries[0]["usd"] == 0.75


class ExhaustedBrain(FakeBrain):
    def propose(self, *args, **kwargs):
        raise BudgetExhausted("cap hit")


def test_loop_stops_gracefully_on_budget_exhausted(tmp_path):
    config = LoopConfig(data_dir=DATA_DIR, run_dir=tmp_path / "run", sigma=0.003, max_iters=5)
    loop = Loop(config, ExhaustedBrain("", root=str(ROOT)))
    summary = loop.run()
    assert summary["stop_reason"] == "budget_exhausted"
    records = [json.loads(l) for l in (loop.run_dir / "journal.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert "budget_exhausted" in records[0]["error"]
    assert "usd_total" in records[0]


def test_max_usd_soft_ceiling(tmp_path):
    brain = FakeBrain("", root=str(ROOT))
    brain.usd_run = 99.0
    config = LoopConfig(data_dir=DATA_DIR, run_dir=tmp_path / "run", sigma=0.003,
                        max_iters=5, max_usd=10.0)
    loop = Loop(config, brain)
    summary = loop.run()
    assert summary["stop_reason"] == "max_usd"
    assert summary["iterations"] == 0
