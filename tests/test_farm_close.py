"""Typed-plan, blend, and loop coverage for the farm-close node."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from agent import prompts
from agent.brain import extract_json_spec
from agent.fake_brain import FakeBrain
from harness import farm_close
from harness.farm_close import (
    Candidate,
    FarmClosePlanError,
    MemberResult,
    PredictionVector,
    assert_member_distinctness,
    blend_rank_average,
    run_plan,
    select_full_candidate,
    select_probe_portfolio,
    validate_plan,
)
from harness.loop import Loop, LoopConfig, Node, RunResult

from conftest import ROOT


DATA_DIR = ROOT / "data" / "synthetic"


def smoke_member_code(key_expression: str, smooth: float) -> str:
    return f'''\
import argparse, csv, json
from pathlib import Path
from data.official.evaluate import evaluate

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, required=True)
p.add_argument("--out-dir", type=Path, required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
with (a.data_dir / "train.csv").open(newline="") as handle:
    train = list(csv.DictReader(handle))
with (a.data_dir / "val.csv").open(newline="") as handle:
    valid = list(csv.DictReader(handle))

def key(row):
    return {key_expression}

prior = sum(int(row["long_view"]) for row in train) / len(train)
counts = {{}}
for row in train:
    n, total = counts.get(key(row), (0, 0))
    counts[key(row)] = (n + 1, total + int(row["long_view"]))
scores = []
for row in valid:
    n, total = counts.get(key(row), (0, 0))
    scores.append((total + {smooth!r} * prior) / (n + {smooth!r}))
raw = evaluate(
    [int(row["user_id"]) for row in valid],
    [int(row["long_view"]) for row in valid],
    scores,
)
metrics = {{
    "gauc": raw["GAUC"], "ndcg5": raw["nDCG@5"], "primary": raw["primary"],
    "history": [{{"epoch": 1, "train_loss": None,
                  "val_gauc": raw["GAUC"], "val_primary": raw["primary"]}}],
}}
a.out_dir.mkdir(parents=True, exist_ok=True)
with (a.out_dir / "predictions.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["row_id", "user_id", "video_id", "score"])
    for index, (row, score) in enumerate(zip(valid, scores)):
        writer.writerow([index, row["user_id"], row["video_id"], score])
with (a.out_dir / "metrics.json").open("w") as handle:
    json.dump(metrics, handle)
'''


def valid_plan() -> dict:
    return {
        "probe_epochs": 2,
        "admission_primary": 0.604,
        "members": [
            {
                "family": f"family-{index}",
                "code": f"# generated family {index}\n",
                "config": {"epochs": 5 + index, "batch_size": 128},
                "seed": 40 + index,
            }
            for index in range(4)
        ],
        "blend": {"method": "rank_average", "scope": "per_user"},
    }


def test_plan_validation_normalizes_and_rejects_schema_violations() -> None:
    plan = validate_plan(valid_plan())
    assert len(plan["members"]) == 4
    assert plan["probe_epochs"] == 2
    assert plan["admission_primary"] == pytest.approx(0.604)
    assert plan["full_member_limit"] == 3
    assert plan["min_probe_blend_gain"] == 0.0
    assert plan["blend"] == {
        "weights": "equal",
        "aggregations": [{"method": "rank_average", "scope": "per_user"}],
    }

    too_small = valid_plan()
    too_small["members"] = too_small["members"][:3]
    with pytest.raises(FarmClosePlanError, match="4 to 6"):
        validate_plan(too_small)

    duplicate_seed = valid_plan()
    duplicate_seed["members"][1]["seed"] = duplicate_seed["members"][0]["seed"]
    with pytest.raises(FarmClosePlanError, match="seeds must be distinct"):
        validate_plan(duplicate_seed)

    two_sources = valid_plan()
    two_sources["members"][0]["script_source"] = "zoo/fm_torch.py"
    with pytest.raises(FarmClosePlanError, match="exactly one"):
        validate_plan(two_sources)

    low_gate = valid_plan()
    low_gate["admission_primary"] = 0.6039
    with pytest.raises(FarmClosePlanError, match="0.6040"):
        validate_plan(low_gate)

    two_rules = valid_plan()
    two_rules["blend"] = {
        "weights": "equal",
        "aggregations": [
            {"method": "rank_average", "scope": "per_user"},
            {"method": "rank_average", "scope": "global"},
        ],
    }
    assert len(validate_plan(two_rules)["blend"]["aggregations"]) == 2

    weighted = valid_plan()
    weighted["blend"]["weights"] = "optimized"
    with pytest.raises(FarmClosePlanError, match="weights must be 'equal'"):
        validate_plan(weighted)

    too_many_rules = valid_plan()
    too_many_rules["blend"] = {
        "weights": "equal",
        "aggregations": [
            {"method": "rank_average", "scope": "per_user"},
            {"method": "rank_average", "scope": "global"},
            {"method": "rank_average", "scope": "per_user"},
        ],
    }
    with pytest.raises(FarmClosePlanError, match="one or two rules"):
        validate_plan(too_many_rules)


def test_proposer_contract_and_parser_use_typed_plan_instead_of_code() -> None:
    prompt = prompts.proposer_user_prompt(
        [],
        "improve",
        "node_003",
        "# parent",
        method_selection={
            "diagnosis": "flat-signal",
            "chosen_method_id": "diverse-family-farm-close",
            "why": "close now",
        },
        selected_method_card="### diverse-family-farm-close: Farm",
        timeout_s=7200,
    )
    assert "Do NOT write an orchestration" in prompt
    assert "exactly 4-6 members" in prompt
    assert "60-120 minutes" in prompt
    assert '"execution_kind":"farm_close"' in prompt

    farm = {
        "execution_kind": "farm_close",
        "farm_close_plan": valid_plan(),
        "expected_delta": 0.0035,
    }
    assert extract_json_spec(json.dumps(farm)) == farm
    alias = {
        "execution_kind": "farm_close",
        "ensemble_plan": valid_plan(),
        "expected_delta": 0.0035,
    }
    assert extract_json_spec(json.dumps(alias)) == {
        "execution_kind": "farm_close",
        "expected_delta": 0.0035,
        "farm_close_plan": alias["ensemble_plan"],
    }

    script = {"execution_kind": "script", "code": "print('ok')"}
    assert extract_json_spec(json.dumps(script)) == script
    assert extract_json_spec('{"code":"print(1)"}') == {
        "execution_kind": "script",
        "code": "print(1)",
    }
    with pytest.raises(ValueError, match="must not carry code"):
        extract_json_spec(json.dumps({**farm, "code": "print(1)"}))
    with pytest.raises(ValueError, match="must not carry a farm-close plan"):
        extract_json_spec(json.dumps({**script, "ensemble_plan": valid_plan()}))
    with pytest.raises(ValueError, match="missing execution_kind"):
        extract_json_spec(json.dumps({"farm_close_plan": valid_plan()}))


def test_rank_average_math_for_per_user_and_global_scopes() -> None:
    users = np.array([1, 1, 1, 2, 2])
    left = np.array([0.0, 1.0, 2.0, 9.0, 1.0])
    right = np.array([2.0, 1.0, 0.0, 0.0, 9.0])

    np.testing.assert_allclose(
        blend_rank_average(users, [left, right], "per_user"),
        np.full(5, 0.5),
    )
    np.testing.assert_allclose(
        blend_rank_average(users, [left, right], "global"),
        [0.375, 0.375, 0.375, 0.625, 0.75],
    )


def test_member_distinctness_asserts_on_duplicate_and_parent_vectors() -> None:
    first = np.array([0.1, 0.2, 0.3])
    second = np.array([0.3, 0.1, 0.2])
    assert_member_distinctness([first, second], ["first", "second"])

    with pytest.raises(AssertionError, match="first and duplicate"):
        assert_member_distinctness([first, first.copy()], ["first", "duplicate"])
    with pytest.raises(AssertionError, match="allclose to parent"):
        assert_member_distinctness([first, second], ["first", "second"], against=first)


def _member(index: int, primary: float, scores: list[float]) -> MemberResult:
    predictions = PredictionVector(
        row_ids=np.arange(4, dtype=np.int64),
        users=np.array([1, 1, 2, 2], dtype=np.int64),
        videos=np.array([10, 11, 12, 13], dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float64),
    )
    return MemberResult(
        index=index,
        family=f"family-{index}",
        seed=40 + index,
        config={"epochs": 5 + index},
        out_dir=Path(f"member-{index}"),
        metrics={
            "gauc": primary,
            "ndcg5": primary,
            "primary": primary,
            "history": [{"epoch": 1, "train_loss": 0.5, "val_primary": primary}],
        },
        predictions=predictions,
    )


def _candidate(
    kind: str,
    positions: tuple[int, ...],
    member_ids: tuple[str, ...],
    primary: float,
    aggregation_order: int = 0,
) -> Candidate:
    return Candidate(
        kind=kind,
        member_positions=positions,
        member_ids=member_ids,
        aggregation=(
            {"method": "rank_average", "scope": "per_user"}
            if kind == "blend" else None
        ),
        aggregation_order=aggregation_order if kind == "blend" else -1,
        metrics={"gauc": primary, "ndcg5": primary, "primary": primary},
        scores=np.arange(4, dtype=np.float64),
        source_phase="probe" if kind != "incumbent" else "incumbent",
    )


def test_anchor_constrained_selection_and_strict_singleton_gain_path() -> None:
    members = [
        _member(0, 0.8, [0.1, 0.9, 0.2, 0.8]),
        _member(1, 0.8, [0.2, 0.8, 0.3, 0.7]),
        _member(2, 0.7, [0.3, 0.7, 0.1, 0.9]),
    ]
    candidates = [
        _candidate("singleton", (0,), ("family-0",), 0.8),
        _candidate("singleton", (1,), ("family-1",), 0.8),
        _candidate("singleton", (2,), ("family-2",), 0.7),
        _candidate("blend", (1, 2), ("family-1", "family-2"), 0.9),
        _candidate("blend", (0, 2), ("family-0", "family-2"), 0.82),
        _candidate(
            "blend", (0, 1, 2),
            ("family-0", "family-1", "family-2"), 0.82,
        ),
    ]

    anchor, unconstrained, constrained, selected, gain = select_probe_portfolio(
        candidates,
        members,
        full_member_limit=3,
        min_probe_blend_gain=0.0,
    )
    assert anchor.member_ids == ("family-0",)
    assert unconstrained.member_ids == ("family-1", "family-2")
    assert constrained.member_ids == ("family-0", "family-2")
    assert selected is constrained
    assert gain == pytest.approx(0.02)

    no_gain_candidates = [
        _candidate(
            candidate.kind,
            candidate.member_positions,
            candidate.member_ids,
            0.8 if candidate.member_ids == ("family-0", "family-2")
            else candidate.metrics["primary"],
            candidate.aggregation_order,
        )
        for candidate in candidates
    ]
    no_gain_candidates[-1].metrics["primary"] = 0.8
    *_, singleton_selected, zero_gain = select_probe_portfolio(
        no_gain_candidates,
        members,
        full_member_limit=3,
        min_probe_blend_gain=0.0,
    )
    assert singleton_selected.member_ids == ("family-0",)
    assert zero_gain == pytest.approx(0.0)

    incumbent = _candidate("incumbent", (), ("incumbent",), 0.82)
    full_singleton = _candidate("singleton", (0,), ("family-0",), 0.82)
    assert select_full_candidate([full_singleton, incumbent]) is incumbent


def _persist_member_predictions(member: MemberResult, out_dir: Path) -> None:
    member.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        writer.writerows(zip(
            member.predictions.row_ids,
            member.predictions.users,
            member.predictions.videos,
            member.predictions.scores,
        ))


def test_graceful_degradation_to_best_single_member(tmp_path, monkeypatch) -> None:
    probe = [
        _member(0, 0.650, [0.1, 0.9, 0.2, 0.8]),
        _member(1, 0.603, [0.2, 0.8, 0.1, 0.9]),
        _member(2, 0.590, [0.3, 0.7, 0.4, 0.6]),
        _member(3, 0.580, [0.4, 0.6, 0.3, 0.7]),
    ]
    full = [_member(0, 0.651, [0.11, 0.91, 0.21, 0.81])]
    _persist_member_predictions(full[0], tmp_path / "out" / "full" / "family-0")
    calls = []

    monkeypatch.setattr(
        farm_close,
        "_resolve_member_scripts",
        lambda plan, out_dir: [Path(f"member-{index}.py") for index in range(4)],
    )
    monkeypatch.setattr(
        farm_close,
        "_validation_labels",
        lambda data_dir: (
            np.array([1, 1, 2, 2], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
    )

    def fake_phase(phase, *args, **kwargs):
        calls.append(phase)
        return (probe, []) if phase == "probe" else (full, [])

    monkeypatch.setattr(farm_close, "_run_phase", fake_phase)
    result = run_plan(
        valid_plan(),
        tmp_path / "data",
        tmp_path / "out",
        timeout_s=60,
        execution_seed=42,
        base_seed=42,
    )

    assert calls == ["probe", "full"]
    assert result["farm_close"]["degraded_to_single"] is True
    assert result["farm_close"]["final_kind"] == "single_member"
    assert result["farm_close"]["winning_families"] == ["family-0"]
    expected = farm_close._metrics(
        np.array([1, 1, 2, 2]),
        np.array([0, 1, 0, 1]),
        full[0].predictions.scores,
    )
    assert result["primary"] == pytest.approx(expected["primary"])
    assert result["history"][-1]["stage"] == "full_best_single"
    assert result["fallback_to_singleton"] is True
    assert result["fallback_to_incumbent"] is False
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "recipe.json").exists()
    assert (tmp_path / "out" / "plan.requested.json").exists()
    assert (tmp_path / "out" / "plan.resolved.json").exists()
    assert (tmp_path / "out" / "probe_candidates.json").exists()
    assert (tmp_path / "out" / "full_candidates.json").exists()
    probe_audit = json.loads((tmp_path / "out" / "probe_candidates.json").read_text())
    assert probe_audit["planned_candidate_count"] == 15
    assert probe_audit["evaluated_candidate_count"] == 15
    assert all("member_ids" in row and "primary" in row for row in probe_audit["candidates"])
    with (tmp_path / "out" / "predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4


def test_frozen_recipe_recomputes_metrics_from_saved_full_vectors(
    tmp_path, monkeypatch
) -> None:
    probe = [
        _member(index, 0.7 - index * 0.01, scores)
        for index, scores in enumerate([
            [0.1, 0.9, 0.2, 0.8],
            [0.2, 0.8, 0.1, 0.9],
            [0.3, 0.7, 0.4, 0.6],
            [0.4, 0.6, 0.3, 0.7],
        ])
    ]
    selected_scores = np.array([0.1, 0.9, 0.2, 0.8])
    saved_scores = np.array([0.9, 0.1, 0.8, 0.2])
    full = [_member(0, 0.99, saved_scores.tolist())]
    _persist_member_predictions(full[0], tmp_path / "out" / "full" / "family-0")
    full[0].predictions.scores = selected_scores

    monkeypatch.setattr(
        farm_close,
        "_resolve_member_scripts",
        lambda plan, out_dir: [Path(f"member-{index}.py") for index in range(4)],
    )
    monkeypatch.setattr(
        farm_close,
        "_validation_labels",
        lambda data_dir: (
            np.array([1, 1, 2, 2], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        farm_close,
        "_run_phase",
        lambda phase, *args, **kwargs: ((probe, []) if phase == "probe" else (full, [])),
    )

    result = run_plan(
        valid_plan(),
        tmp_path / "data",
        tmp_path / "out",
        timeout_s=60,
        execution_seed=42,
        base_seed=42,
    )

    expected = farm_close._metrics(
        np.array([1, 1, 2, 2]),
        np.array([0, 1, 0, 1]),
        saved_scores,
    )
    assert result["primary"] == pytest.approx(expected["primary"])
    assert result["farm_close"]["selected_table_metrics"]["primary"] != pytest.approx(
        result["primary"]
    )
    recipe = json.loads((tmp_path / "out" / "recipe.json").read_text())
    assert recipe["source_phase"] == "full"
    assert recipe["members"][0]["phase"] == "full"
    assert recipe["members"][0]["seed"] == 40
    np.testing.assert_allclose(
        farm_close.read_predictions(tmp_path / "out" / "predictions.csv").scores,
        saved_scores,
    )


def test_full_stage_can_select_and_record_incumbent(tmp_path, monkeypatch) -> None:
    probe = [
        _member(index, 0.7, scores)
        for index, scores in enumerate([
            [0.1, 0.9, 0.2, 0.8],
            [0.2, 0.8, 0.1, 0.9],
            [0.3, 0.7, 0.4, 0.6],
            [0.4, 0.6, 0.3, 0.7],
        ])
    ]
    full = [_member(0, 0.1, [0.9, 0.1, 0.8, 0.2])]
    _persist_member_predictions(full[0], tmp_path / "out" / "full" / "family-0")
    parent_path = tmp_path / "parent.csv"
    parent_path.write_text(
        "row_id,user_id,video_id,score\n"
        "0,1,10,0.1\n1,1,11,0.9\n2,2,12,0.2\n3,2,13,0.8\n"
    )
    monkeypatch.setattr(
        farm_close,
        "_resolve_member_scripts",
        lambda plan, out_dir: [Path(f"member-{index}.py") for index in range(4)],
    )
    monkeypatch.setattr(
        farm_close,
        "_validation_labels",
        lambda data_dir: (
            np.array([1, 1, 2, 2], dtype=np.int64),
            np.array([0, 1, 0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        farm_close,
        "_run_phase",
        lambda phase, *args, **kwargs: ((probe, []) if phase == "probe" else (full, [])),
    )

    result = run_plan(
        valid_plan(),
        tmp_path / "data",
        tmp_path / "out",
        timeout_s=60,
        execution_seed=42,
        base_seed=42,
        parent_predictions=parent_path,
    )

    assert result["fallback_to_incumbent"] is True
    assert result["fallback_to_singleton"] is False
    assert result["farm_close"]["final_kind"] == "incumbent"
    recipe = json.loads((tmp_path / "out" / "recipe.json").read_text())
    assert recipe["source_phase"] == "incumbent"
    full_audit = json.loads((tmp_path / "out" / "full_candidates.json").read_text())
    assert full_audit["winner"]["kind"] == "incumbent"


def test_farm_close_end_to_end_on_synthetic_data(tmp_path, monkeypatch) -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    plan = {
        "probe_epochs": 1,
        "members": [
            {
                "family": family,
                "code": smoke_member_code(key_expression, smooth),
                "config": {},
                "seed": 70 + index,
            }
            for index, (family, key_expression, smooth) in enumerate(
                [
                    ("smoke-video", 'row["video_id"]', 10.0),
                    ("smoke-video-tab", '(row["video_id"], row["tab"])', 20.0),
                    ("smoke-duration", 'int(row["duration_ms"]) // 10000', 10.0),
                    (
                        "smoke-video-hour",
                        '(row["video_id"], int(row["hourmin"]) // 600)',
                        20.0,
                    ),
                ]
            )
        ],
        "blend": {"method": "rank_average", "scope": "per_user"},
    }

    result = run_plan(
        plan,
        DATA_DIR,
        tmp_path / "farm",
        timeout_s=120,
        execution_seed=42,
        base_seed=42,
    )

    detail = result["farm_close"]
    assert len(detail["probe_members"]) == 4
    assert not detail["probe_failures"]
    assert len(detail["probe_combinations"]) == 11
    assert 1 <= len(detail["full_members"]) <= 3
    assert detail["winning_families"]
    assert result["history"][-1]["val_primary"] == result["primary"]
    assert (tmp_path / "farm" / "predictions.csv").exists()
    assert (tmp_path / "farm" / "metrics.json").exists()
    progress = [
        json.loads(line)
        for line in (tmp_path / "farm" / "progress.log").read_text().splitlines()
    ]
    completed_probe = [
        row for row in progress
        if row["phase"] == "probe" and row["status"] == "completed"
    ]
    assert len(completed_probe) == 4
    assert all("config" in row and "primary" in row for row in completed_probe)


class FarmPlanBrain(FakeBrain):
    def __init__(self) -> None:
        super().__init__("", scripts=[], root=str(ROOT))
        self.repairs = 0

    def select_method(self, *args, **kwargs):
        self.meter.add("fake/fake/selector", 20, 10)
        return {
            "diagnosis": "flat-signal",
            "chosen_method_id": "diverse-family-farm-close",
            "citation": "farm-close card",
            "why": "close with cross-family complementarity",
            "rejected": [],
        }

    def propose(self, *args, **kwargs):
        self.meter.add("fake/fake/proposer", 20, 10)
        invalid = valid_plan()
        invalid["members"][1]["seed"] = invalid["members"][0]["seed"]
        return {
            "execution_kind": "farm_close",
            "hypothesis": "A cross-family rank blend should improve the parent.",
            "expected_delta": 0.01,
            "expected_delta_basis": "The farm-close card reports 0.605863 primary.",
            "ensemble_plan": invalid,
            "timeout_s": 7200,
        }

    def repair_farm_close_plan(self, spec, error):
        assert "seeds must be distinct" in error
        self.repairs += 1
        repaired = dict(spec)
        repaired["ensemble_plan"] = valid_plan()
        self.meter.add("fake/fake/proposer", 20, 10)
        return repaired


def test_loop_repairs_plan_and_counts_farm_close_as_one_iteration(tmp_path, monkeypatch) -> None:
    brain = FarmPlanBrain()
    loop = Loop(
        LoopConfig(
            data_dir=DATA_DIR,
            run_dir=tmp_path / "run",
            cross_run_path=tmp_path / "CROSS_RUN.md",
            sigma=0.003,
            timeout_s=7200,
        ),
        brain,
    )
    loop.nodes_dir.mkdir(parents=True)
    parent = Node("node_000", "baseline", "draft", "baseline", loop.nodes_dir / "000.py")
    parent.code_path.write_text("# baseline\n")
    parent.metrics = {"gauc": 0.70, "ndcg5": 0.70, "primary": 0.70, "history": []}
    parent.primary = 0.70
    parent.status = "accepted"
    loop.nodes[parent.node_id] = parent
    loop.champion = parent
    loop.sigma = 0.003
    monkeypatch.setattr(
        loop,
        "run_experiment",
        lambda node, timeout: (
            RunResult(
                True,
                metrics={
                    "gauc": 0.71,
                    "ndcg5": 0.71,
                    "primary": 0.71,
                    "history": [{"epoch": 1, "val_primary": 0.71}],
                },
            ),
            "full",
        ),
    )

    loop.iterate(1)

    node = loop.nodes["node_001"]
    assert brain.repairs == 1
    assert node.status == "accepted"
    assert loop.champion is node
    assert node.farm_close_plan == validate_plan(valid_plan())
    assert node.farm_close_plan_path.exists()
    assert "from harness.farm_close import run_plan" in node.code_path.read_text()
    records = [json.loads(line) for line in loop.journal_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["n"] == 1
    assert records[0]["farm_close_plan"] == node.farm_close_plan
    assert records[0]["recovery"] == "plan-repaired"


class IncumbentFallbackBrain(FarmPlanBrain):
    def __init__(self) -> None:
        super().__init__()
        self.fix_calls = 0

    def propose(self, *args, **kwargs):
        self.meter.add("fake/fake/proposer", 20, 10)
        # Deliberately omit the discriminator to exercise the method-card fallback.
        return {
            "hypothesis": "Retain the incumbent when all full candidates lose.",
            "expected_delta": 0.0,
            "expected_delta_basis": "The incumbent is the declared full-stage comparator.",
            "farm_close_plan": valid_plan(),
        }

    def fix(self, code, error):
        self.fix_calls += 1
        return code


def test_incumbent_fallback_is_recorded_and_skips_noop_fixer(
    tmp_path, monkeypatch
) -> None:
    brain = IncumbentFallbackBrain()
    loop = Loop(
        LoopConfig(
            data_dir=DATA_DIR,
            run_dir=tmp_path / "run",
            cross_run_path=tmp_path / "CROSS_RUN.md",
            sigma=0.003,
            timeout_s=7200,
        ),
        brain,
    )
    loop.nodes_dir.mkdir(parents=True)
    parent = Node("node_000", "baseline", "draft", "baseline", loop.nodes_dir / "000.py")
    parent.code_path.write_text("# baseline\n")
    parent.metrics = {"gauc": 0.70, "ndcg5": 0.70, "primary": 0.70, "history": []}
    parent.primary = 0.70
    parent.status = "accepted"
    loop.nodes[parent.node_id] = parent
    loop.champion = parent
    loop.sigma = 0.003
    parent_dir = loop.run_dir / parent.node_id
    parent_dir.mkdir()
    parent_predictions = parent_dir / "predictions.csv"
    parent_predictions.write_text(
        "row_id,user_id,video_id,score\n"
        "0,1,10,0.1\n1,1,11,0.9\n2,2,12,0.2\n3,2,13,0.8\n"
    )
    run_calls = []

    def fake_run(node, timeout):
        run_calls.append(node.execution_kind)
        node_dir = loop.run_dir / node.node_id
        node_dir.mkdir(exist_ok=True)
        (node_dir / "predictions.csv").write_bytes(parent_predictions.read_bytes())
        return RunResult(
            True,
            metrics={
                "gauc": 0.70,
                "ndcg5": 0.70,
                "primary": 0.70,
                "history": [],
                "fallback_to_singleton": False,
                "fallback_to_incumbent": True,
                "farm_close": {"fallback_to_incumbent": True},
            },
        ), "full"

    monkeypatch.setattr(loop, "run_experiment", fake_run)
    loop.iterate(1)

    node = loop.nodes["node_001"]
    assert run_calls == ["farm_close"]
    assert node.status == "rejected"
    assert node.failure_stage is None
    assert node.fixer_eligible is False
    assert brain.fix_calls == 0
    assert node.metrics["fallback_to_incumbent"] is True


def test_cli_exposes_direct_farm_close_subcommand() -> None:
    from harness.cli import build_parser

    args = build_parser().parse_args([
        "farm-close",
        "--plan",
        "plan.json",
        "--data-dir",
        "data/synthetic",
        "--out-dir",
        "out",
        "--parent-predictions",
        "parent/predictions.csv",
    ])
    assert args.command == "farm-close"
    assert args.timeout_s == 7200.0
