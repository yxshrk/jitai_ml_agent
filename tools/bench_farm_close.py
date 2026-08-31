"""Farm-close assembler bench over REAL cached member vectors (tier-2 eval).

Runs the deterministic selection/blend machinery of harness/farm_close.py over
the four measured cross-family champion prediction sets (the same vectors the
heterogeneous blend audit used) and asserts the decisions. No training happens
here and no LLM is called: this validates the assembler's math and policy, not
the agent. All numbers are CACHED-ARTIFACT VALIDATION, not run results.

Usage: uv run python tools/bench_farm_close.py
Exit code 0 = all checks pass.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.farm_close import (  # noqa: E402
    Candidate,
    MemberResult,
    _candidate_results,
    _declared_candidate_count,
    _metrics,
    _validation_labels,
    blend_rank_average,
    read_predictions,
    select_full_candidate,
    select_probe_portfolio,
)

# The four measured families (see evidence/blend_audit.md); primaries are the
# audited full-fidelity numbers those files scored on official validation.
MEMBERS = [
    ("champ", "package-seed-ensemble", "logs/run_bigclock_07/node_006/predictions.csv", 0.605575),
    ("tkern", "temporal-pair-kernel", "logs/run_novel_l1/node_004/predictions.csv", 0.605146),
    ("dsamp", "decayed-positive-sampling", "logs/run_qb_b/node_001/predictions.csv", 0.604657),
    ("gbce", "gauge-fixed-bce", "logs/run_novel_r1/node_003/predictions.csv", 0.604479),
]
AUDIT_EQUAL_BLEND = 0.605639  # predeclared 4-way midrank blend, blend_audit.md

checks: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    checks.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def build_members(users: np.ndarray, labels: np.ndarray) -> list[MemberResult]:
    members = []
    for index, (mid, family, rel, _expected) in enumerate(MEMBERS):
        predictions = read_predictions(ROOT / rel)
        members.append(MemberResult(
            index=index,
            family=family,
            seed=42 + index,
            config={"member": mid},
            out_dir=ROOT / "logs",
            metrics=_metrics(users, labels, predictions.scores),
            predictions=predictions,
        ))
    return members


def main() -> int:
    users, labels = _validation_labels(ROOT / "data" / "real_ws")
    members = build_members(users, labels)

    # 1. Singleton primaries reproduce the audited numbers.
    for member, (mid, _f, _p, expected) in zip(members, MEMBERS):
        got = member.metrics["primary"]
        check(f"singleton:{mid}", abs(got - expected) < 2e-4,
              f"primary {got:.6f} vs audited {expected:.6f}")

    aggregations = [{"method": "rank_average", "scope": "global"}]
    candidates = _candidate_results(members, users, labels, aggregations, "probe")

    # 2. Enumeration is complete and matches the declared count.
    declared = _declared_candidate_count(len(members), len(aggregations))
    check("enumeration", len(candidates) == declared,
          f"{len(candidates)} candidates vs declared {declared}")

    # 3. Equal 4-way blend lands in the audited neighborhood (ordinal vs
    #    midrank ranking differ slightly; the audit's midrank scored 0.605639).
    four = next(c for c in candidates
                if c.kind == "blend" and len(c.member_ids) == 4)
    check("four-way-blend", abs(four.metrics["primary"] - AUDIT_EQUAL_BLEND) < 8e-4,
          f"primary {four.metrics['primary']:.6f} vs audit midrank {AUDIT_EQUAL_BLEND:.6f}")

    # 4. Anchor policy: best standalone (champ) is the anchor; the selected
    #    portfolio contains it with 2-3 distinct families.
    anchor, unconstrained, constrained, selected, gain = select_probe_portfolio(
        candidates, members, full_member_limit=3, min_probe_blend_gain=0.0)
    check("anchor", anchor.member_ids == ("package-seed-ensemble",),
          f"anchor {anchor.member_ids} primary {anchor.metrics['primary']:.6f}")
    if selected.kind == "blend":
        ok = "package-seed-ensemble" in selected.member_ids and 2 <= len(selected.member_ids) <= 3
        check("portfolio", ok and gain is not None and gain > 0,
              f"selected {selected.member_ids} gain {gain:+.6f}")
    else:
        check("portfolio", gain is None or gain <= 0,
              f"singleton path taken, gain {gain}")
    if unconstrained is not None:
        print(f"       unconstrained probe winner: {unconstrained.member_ids} "
              f"{unconstrained.metrics['primary']:.6f} (logged, not auto-promoted)")

    # 5. Singleton path when the gain bar is unreachable.
    _a, _u, _c, forced, forced_gain = select_probe_portfolio(
        candidates, members, full_member_limit=3, min_probe_blend_gain=1.0)
    check("singleton-path", forced.kind == "singleton" and forced.member_ids == ("package-seed-ensemble",),
          f"gain bar 1.0 -> {forced.kind} {forced.member_ids}")

    # 6. Determinism under member permutation: same selected member-id set.
    baseline_ids = set(selected.member_ids)
    rng = random.Random(7)
    stable = True
    for _ in range(4):
        order = list(range(len(members)))
        rng.shuffle(order)
        permuted = [members[i] for i in order]
        for new_index, member in enumerate(permuted):
            member.index = new_index
        cands_p = _candidate_results(permuted, users, labels, aggregations, "probe")
        _a2, _u2, _c2, sel_p, _g2 = select_probe_portfolio(
            cands_p, permuted, full_member_limit=3, min_probe_blend_gain=0.0)
        if set(sel_p.member_ids) != baseline_ids:
            stable = False
            break
    for original_index, member in enumerate(members):
        member.index = original_index
    check("determinism", stable, f"selected set stable under permutation: {sorted(baseline_ids)}")

    # 7. Full-stage table: incumbent (champ vector) beats or ties weaker fulls,
    #    and exact ties retain the incumbent.
    weak = [m for m in members if m.family != "package-seed-ensemble"]
    full_cands = _candidate_results(weak, users, labels, aggregations, "full")
    incumbent = Candidate(
        kind="incumbent", member_positions=(), member_ids=("incumbent",),
        aggregation=None, aggregation_order=-1,
        metrics=_metrics(users, labels, members[0].predictions.scores),
        scores=members[0].predictions.scores, source_phase="incumbent")
    winner = select_full_candidate(full_cands + [incumbent])
    check("full-selection", winner.metrics["primary"] >= max(
        c.metrics["primary"] for c in full_cands),
        f"winner {winner.kind} {winner.member_ids} {winner.metrics['primary']:.6f}")
    tie_clone = Candidate(
        kind="blend", member_positions=(0, 1), member_ids=("a", "b"),
        aggregation={"method": "rank_average", "scope": "global"}, aggregation_order=0,
        metrics=dict(incumbent.metrics), scores=incumbent.scores, source_phase="full")
    tie_winner = select_full_candidate([tie_clone, incumbent])
    check("tie-retains-incumbent", tie_winner.kind == "incumbent",
          f"exact tie -> {tie_winner.kind}")

    # 8. Blend math cross-check: independent rank-average of champ+tkern equals
    #    the assembler's candidate for the same pair.
    pair = next(c for c in candidates if set(c.member_ids) == {"package-seed-ensemble", "temporal-pair-kernel"})
    manual = blend_rank_average(
        users, [members[0].predictions.scores, members[1].predictions.scores], "global")
    check("blend-math", float(np.max(np.abs(manual - pair.scores))) < 1e-12,
          "independent recomputation matches candidate scores")

    report = {
        "label": "cached-artifact validation (assembler bench, no training, no LLM)",
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
        "all_passed": all(p for _n, p, _d in checks),
    }
    out = ROOT / "logs" / "bench_farm_close.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{sum(p for _n,p,_d in checks)}/{len(checks)} checks passed -> {out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
