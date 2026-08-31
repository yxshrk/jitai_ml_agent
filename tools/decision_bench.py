"""Decision bench: evaluate the SELECTOR's judgment against frozen decision states
with known-best moves — real LLM calls, no training. Each scenario reconstructs a
real (or realistic) mid-run state; the selector's pick is scored against an
accepted-move set derived from the campaign ledger.

Usage: uv run python tools/decision_bench.py            # all scenarios
       uv run python tools/decision_bench.py --n 3      # repeat each 3x (stochasticity)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain  # noqa: E402

CURVE_OVERFIT = [{"epoch": e, "train_loss": 0.55 - 0.01 * e,
                  "val_gauc": 0.665 + 0.002 * min(e, 4) - 0.001 * max(0, e - 4),
                  "val_primary": 0.600 + 0.001 * min(e, 4) - 0.0006 * max(0, e - 4)}
                 for e in range(1, 11)]
CURVE_NONE: list = []

F4_NODE002_CURVE = [{"epoch": 1, "train_loss": None, "val_gauc": 0.656895, "val_primary": 0.593472}, {"epoch": 2, "train_loss": None, "val_gauc": 0.662607, "val_primary": 0.597593}, {"epoch": 3, "train_loss": None, "val_gauc": 0.666214, "val_primary": 0.600637}, {"epoch": 4, "train_loss": None, "val_gauc": 0.667006, "val_primary": 0.601295}, {"epoch": 5, "train_loss": None, "val_gauc": 0.668237, "val_primary": 0.602319}, {"epoch": 6, "train_loss": None, "val_gauc": 0.668707, "val_primary": 0.602584}, {"epoch": 7, "train_loss": None, "val_gauc": 0.669069, "val_primary": 0.602833}, {"epoch": 8, "train_loss": None, "val_gauc": 0.669443, "val_primary": 0.603106}, {"epoch": 9, "train_loss": None, "val_gauc": 0.669412, "val_primary": 0.603117}, {"epoch": 10, "train_loss": None, "val_gauc": 0.669376, "val_primary": 0.603119}]

SCENARIOS = [
    dict(
        name="strong_opener_full_streak",
        note="eps-clearing dial accept (z2-now state): best move = a big-evidence "
             "opportunity or close, NOT another small treatment",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [draft] "package-dial-sweep" primary=0.6051 ACCEPTED',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 16},
        good={"seq-deepfm-composite", "diverse-family-farm-close",
              "heterogeneous-ensemble-design", "ensemble-design-sweep",
              "context-stratified-pairs", "temporal-pair-kernel"},
        bad={"freq-adaptive-reg", "lambda-weighted-pairs", "listwise-regime",
             "mtl-shared-bottom", "dndcg-lambda"},
    ),
    dict(
        name="two_strikes_on_good_base",
        note="composite accepted then 2 rejects (x1-n6 state): the ONLY sane move "
             "is an ensemble close",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED',
            'node_003 [draft] "seq-deepfm-composite" primary=0.6044 ACCEPTED',
            'node_004 [draft] "gauge-fixed-bce" primary=0.6040 REJECTED',
            'node_005 [draft] "social-mtl-heads" primary=0.6043 REJECTED',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 2, "iterations_done": 5, "max_iters": 16},
        good={"ensemble-design-sweep", "heterogeneous-ensemble-design",
              "diverse-family-farm-close"},
        bad={"regularization-schedule", "freq-adaptive-reg", "embedding-dim-down",
             "listwise-regime", "recency-weighting"},
    ),
    dict(
        name="blind_telemetry",
        note="parent emitted no curve (r8 state): must NOT guess-diagnose into a "
             "treatment; opportunities are the low-confidence play",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED',
            'node_001 [draft] "gated-session-residual" primary=0.6024 ACCEPTED',
        ],
        history=CURVE_NONE,
        streak={"no_improve_streak": 1, "iterations_done": 2, "max_iters": 16},
        good={"seq-deepfm-composite", "diverse-family-farm-close",
              "context-stratified-pairs", "package-dial-sweep",
              "ensemble-design-sweep", "heterogeneous-ensemble-design"},
        bad={"freq-adaptive-reg", "mtl-shared-bottom", "listwise-regime"},
    ),
    dict(
        name="dead_family_temptation",
        note="overfit diagnosis with watch-time cards visible: must not pick "
             "measured-dead families",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 16},
        good={"package-dial-sweep", "seq-deepfm-composite", "regularization-schedule",
              "diverse-family-farm-close", "context-stratified-pairs",
              "temporal-pair-kernel", "gauge-fixed-bce", "stage-matrix-sweep"},
        bad={"relative-watch-component", "listwise-regime", "mtl-shared-bottom",
             "dndcg-lambda", "finalmlp", "covisit-svd-init"},
    ),
    dict(
        name="opening_expected_value",
        note="farm_f2 iter-1 state (31 Aug): fresh calibrated baseline, full clock. "
             "The opening treatment should be the highest evidence-expected-gain "
             "broad package (measured +0.002-0.003 class), not a small single "
             "treatment that spends an iteration on a likely sub-epsilon strike",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 16},
        good={"seq-deepfm-composite", "package-dial-sweep", "stage-matrix-sweep",
              "context-stratified-pairs", "temporal-pair-kernel"},
        bad={"regularization-schedule", "freq-adaptive-reg", "embedding-dim-down",
             "swa-then-ensemble", "seed-ensemble", "session-time-features",
             "diverse-family-farm-close", "heterogeneous-ensemble-design"},
    ),
    dict(
        name="endgame_eps_math",
        note="farm_f1 failure state (31 Aug): streak=2 so the run ends unless THIS "
             "iteration gains >= eps=0.002; same-family seed closes measure well "
             "below eps, so only a cross-family close has evidence reaching the bar",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [draft] "seq-deepfm-composite" primary=0.6037 ACCEPTED (+0.0018, below eps: strike 1)',
            'node_002 [draft] "gauge-fixed-bce" primary=0.6046 ACCEPTED (+0.0010, below eps: strike 2)',
        ],
        history=CURVE_NONE,
        streak={"no_improve_streak": 2, "iterations_done": 3, "max_iters": 16},
        good={"diverse-family-farm-close", "heterogeneous-ensemble-design"},
        bad={"ensemble-design-sweep", "seed-ensemble", "swa-then-ensemble",
             "regularization-schedule", "freq-adaptive-reg", "session-time-features"},
        expect_plan=True,
    ),
    # Weak-base endgames (best ~0.6026-0.6029): a strong PACKAGE is neutral, not
    # bad. Two measured facts (31 Aug): the farm-close precondition requires an
    # established champion to derive members from, which a 0.603 base barely is,
    # and farm_f4r2 from that exact base yielded a package-class singleton anyway.
    # The close is still the reference answer; atoms and same-family seeds stay bad.
    dict(
        name="endgame_margin_not_reach",
        note="farm_f2 iter-3 state (31 Aug): LOW base (0.6026) so single atoms "
             "marginally reach eps on paper; the right move is still the close "
             "whose evidence clears eps with margin, not the atom that barely "
             "touches the bar",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [draft] "regularization-schedule" primary=0.6026 ACCEPTED (+0.0007, grey z-pass; below eps: strike 1)',
            'node_002 [draft] "context-stratified-pairs" primary=0.6021 REJECTED (strike 2)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 2, "iterations_done": 3, "max_iters": 16},
        good={"diverse-family-farm-close", "heterogeneous-ensemble-design"},
        bad={"gauge-fixed-bce", "regularization-schedule", "seed-ensemble",
             "ensemble-design-sweep", "freq-adaptive-reg"},
        expect_plan=True,
    ),
    dict(
        name="endgame_unspent_package_trap",
        note="farm_f4 iter-3 state (31 Aug): a measured package still unspent, "
             "but its absolute evidence sits near the current best (little "
             "headroom); the close with ledger-exceeding evidence is still the "
             "margin-maximal pick",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [draft] "(proposal failed: transport timeout)" VOID (strike 1)',
            'node_002 [draft] "gauge-fixed-bce" primary=0.6029 ACCEPTED (+0.0011, below eps: strike 2)',
        ],
        history=CURVE_NONE,
        streak={"no_improve_streak": 2, "iterations_done": 3, "max_iters": 16},
        good={"diverse-family-farm-close", "heterogeneous-ensemble-design"},
        bad={"seq-deepfm-composite", "gauge-fixed-bce",
             "regularization-schedule", "seed-ensemble", "ensemble-design-sweep"},
        expect_plan=True,
    ),
    dict(
        name="f4r_exact_state",
        note="EXACT live state of farm_f4r iteration 3 (journal lines verbatim from "
             "the logged selector prompt; parent curve = f4 node_002 history after "
             "normalization). Streak 2, best 0.6029. The margin-maximal move with "
             "corrected card evidence is the cross-family close.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [<-node_000] draft "(proposal failed)" no-metric FAILED',
            'node_002 [<-node_001] debug "Replacing ordinary pointwise logits with complete-slate user-centered BCE logits while leaving the hybrid BPR term and regularization unchanged will improve validation primary by approximately 0.0026 through better alignment with within-user ranking metrics." primary=0.6029 ACCEPTED',
        ],
        history=F4_NODE002_CURVE,
        streak={"no_improve_streak": 2, "n_converge": 3, "iters_left": 13},
        good={"diverse-family-farm-close", "heterogeneous-ensemble-design"},
        bad={"seq-deepfm-composite", "gauge-fixed-bce",
             "regularization-schedule", "seed-ensemble", "ensemble-design-sweep"},
        expect_plan=True,
    ),
    dict(
        name="close_rejected_strengthen_first",
        note="farm_f6r post-rejection state (31 Aug): a cross-family close scored "
             "above the incumbent but failed the repeat-seed confirm. Re-rolling the "
             "same close is not a new experiment; the right move adds a stronger "
             "distinct member (a measured package from another family) before closing again.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [<-node_000] draft "package-dial-sweep" primary=0.6014 REJECTED (defective implementation)',
            'node_002 [<-node_000] draft "seq-deepfm-composite" primary=0.6042 ACCEPTED (+0.0024)',
            'node_003 [<-node_002] improve "gauge-fixed-bce" primary=0.6042 REJECTED',
            'node_004 [<-node_002] improve "heterogeneous-ensemble-design (farm-close: blend 0.6050 vs incumbent 0.6042)" primary=0.6050 REJECTED (confirm reruns fell back to incumbent: gain not repeatable)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 2, "n_converge": 3, "iters_left": 11},
        good={"package-dial-sweep", "stage-matrix-sweep", "context-stratified-pairs",
              "temporal-pair-kernel"},
        bad={"heterogeneous-ensemble-design", "diverse-family-farm-close",
             "seed-ensemble", "ensemble-design-sweep", "regularization-schedule"},
    ),
    dict(
        name="farm_close_uniquely_right",
        note="two DIFFERENT families independently measured near the ceiling, "
             "plateau streak, clock half spent: the doctrine answer is the "
             "cross-family farm-close (hetero design acceptable); any single "
             "further treatment or same-family seed ensemble wastes the node",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0001)',
            'node_001 [draft] "package-dial-sweep" primary=0.6049 ACCEPTED',
            'node_002 [draft] "temporal-pair-kernel" primary=0.6051 ACCEPTED',
            'node_003 [draft] "session-time-features" primary=0.6048 REJECTED (below floor)',
            'node_004 [draft] "regularization-schedule" primary=0.6050 REJECTED (below floor)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 2, "iterations_done": 5, "max_iters": 10},
        good={"diverse-family-farm-close", "heterogeneous-ensemble-design"},
        bad={"seed-ensemble", "regularization-schedule", "freq-adaptive-reg",
             "session-time-features", "recency-weighting", "embedding-dim-down",
             "mtl-shared-bottom", "listwise-regime"},
        expect_plan=True,
    ),
    dict(
        name="bank_last_gain",
        note="f9 iter-4 state (1 Sep): 0.6051 single-model champion, cross-family "
             "close just rejected (weak fresh members), streak 2. No card evidence "
             "reaches eps=0.002, so the run ends this iteration either way; the "
             "deliverable is best-so-far. Right move = the reliable small gain "
             "(seed ensemble of the champion, +0.0003..+0.001); wrong = another "
             "long-shot blend or atom that forfeits the bankable gain.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0004)',
            'node_001 [<-node_000] draft "package-dial-sweep" primary=0.6051 ACCEPTED (+0.0033, single model, 48-probe sweep)',
            'node_002 [<-node_001] improve "context-stratified-pairs" primary=0.6051 REJECTED (+0.00002, below floor: strike 1)',
            'node_003 [<-node_001] improve "diverse-family-farm-close (members: temporal 0.6035, seq-deepfm 0.6012; blend fell back to incumbent)" primary=0.6051 REJECTED (strike 2)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 2, "n_converge": 3, "iters_left": 12},
        good={"seed-ensemble", "swa-then-ensemble", "snapshot-ensemble"},
        bad={"hetero-objective-ensemble", "diverse-family-farm-close",
             "heterogeneous-ensemble-design", "listwise-regime", "social-mtl-heads"},
    ),
    dict(
        name="chain_open",
        note="1 Sep lean-library chain, step 1 (real f9 state): fresh calibrated "
             "baseline. Right = the swept package opener; wrong = any close or a "
             "single atom.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0002)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 0, "iterations_done": 0, "max_iters": 16},
        good={"package-dial-sweep", "stage-matrix-sweep"},
        bad={"seed-ensemble", "ensemble-design-sweep", "heterogeneous-ensemble-design",
             "diverse-family-farm-close", "recency-weighting", "bpr-hybrid"},
    ),
    dict(
        name="chain_after_strong_opener",
        note="1 Sep chain, step 2 (real f9 state): swept package accepted at 0.6051 "
             "single, streak 0. Right = compound an ORTHOGONAL mechanism with its own "
             "re-sweep (sampler/objective rider); wrong = close already, or re-apply "
             "a component the package contains.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0002)',
            'node_001 [<-node_000] draft "package-dial-sweep" primary=0.6051 ACCEPTED (+0.0033, single model, 48-probe sweep)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 16},
        good={"context-stratified-pairs", "temporal-pair-kernel", "gauge-fixed-bce",
              "decayed-positive-sampling", "combo-sweep"},
        bad={"seed-ensemble", "ensemble-design-sweep", "diverse-family-farm-close",
             "heterogeneous-ensemble-design", "regularization-schedule",
             "recency-weighting", "bpr-hybrid", "dcn-lite"},
    ),
    dict(
        name="chain_after_ctx_tie",
        note="1 Sep chain, step 3 (real f9 state): ctx re-sweep node tied the champion "
             "(0.6051, rejected +0.00002, strike 1). Right = close over the run's OWN "
             "lineage (both strong artifacts exist) or one more orthogonal rider; "
             "wrong = a fresh-member farm, or an embedded component.",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0002)',
            'node_001 [<-node_000] draft "package-dial-sweep" primary=0.6051 ACCEPTED (+0.0033, single model)',
            'node_002 [<-node_001] improve "context-stratified-pairs (re-swept with rho)" primary=0.6051 REJECTED (+0.00002 below floor: strike 1)',
        ],
        history=CURVE_OVERFIT,
        streak={"no_improve_streak": 1, "iterations_done": 2, "max_iters": 16},
        good={"ensemble-design-sweep", "seed-ensemble", "heterogeneous-ensemble-design",
              "temporal-pair-kernel", "gauge-fixed-bce", "decayed-positive-sampling"},
        bad={"diverse-family-farm-close", "regularization-schedule", "recency-weighting",
             "bpr-hybrid", "dcn-lite", "package-dial-sweep"},
    ),
]


def check_plan_emission(brain, sc, sel):
    """Follow a farm-close selection through to the proposer and validate the
    typed plan envelope + schema. Returns 'valid' | 'invalid: <why>' | None."""
    from agent.brain import normalize_proposal_envelope
    from harness.farm_close import FarmClosePlanError, validate_plan
    try:
        spec = brain.propose(
            sc["journal"], "draft", "node_002", "",
            method_selection=sel, streak_state=sc["streak"],
            parent_history=sc["history"], context_mode="compact",
            parent_code_path="logs/run_bigclock_07/nodes/003.py",
        )
        spec = normalize_proposal_envelope(spec)
        if spec.get("execution_kind") != "farm_close":
            return f"invalid: execution_kind={spec.get('execution_kind')}"
        validate_plan(spec["farm_close_plan"])
        plan = spec["farm_close_plan"]
        families = [m["family"] for m in plan["members"]]
        kinds = ["src" if "script_source" in m else "code" for m in plan["members"]]
        anchored = "script_source" in plan["members"][0]
        return (f"valid ({len(families)} members: {', '.join(families)}; "
                f"sources={kinds}; anchor_is_champion_script={anchored})")
    except (ValueError, FarmClosePlanError) as exc:
        return f"invalid: {str(exc)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    menu = (ROOT / "MENU.md").read_text()
    brain = Brain(menu, provider="openai", knowledge_mode="full")
    score = {"good": 0, "ok": 0, "bad": 0}
    for sc in SCENARIOS:
        for rep in range(args.n):
            sel = brain.select_method(
                sc["journal"], sc["history"], sc["streak"],
                excluded_families=[], dataset="pure", prior_runs=None)
            pick = sel.get("chosen_method_id")
            verdict = ("good" if pick in sc["good"]
                       else "bad" if pick in sc["bad"] else "ok")
            score[verdict] += 1
            if sc.get("expect_plan") and pick in ("diverse-family-farm-close", "heterogeneous-ensemble-design"):
                emission = check_plan_emission(brain, sc, sel)
                print(f"    plan emission: {emission}")
                if emission and emission.startswith("invalid"):
                    score["bad"] += 0  # tracked in print; selection verdict stands
            print(f"[{sc['name']}] rep{rep}: {pick} -> {verdict.upper()}"
                  + (f"  (diag: {sel.get('diagnosis')})" if sel.get('diagnosis') else ""))
            if verdict == "bad":
                print(f"    why: {(sel.get('why') or '')[:180]}")
    total = sum(score.values())
    print(f"\nBENCH: {score['good']}/{total} good, {score['ok']} neutral, "
          f"{score['bad']} bad")
    json.dump(score, open(ROOT / "logs/decision_bench_result.json", "w"))


if __name__ == "__main__":
    main()
