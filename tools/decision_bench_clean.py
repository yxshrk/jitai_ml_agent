"""Clean-mode decision bench: evaluate the literature-only selector's STRATEGIC
judgment against frozen decision states. No campaign results appear anywhere in
this file or in the clean knowledge it exercises: every metric value in the
scenario journals is synthetic (0.71xx scale), and picks are scored by card
FAMILY properties (screen / compose-what-worked / avoid-dead-family / honest
telemetry / ensemble close), not by matching specific measured winners.

Usage: uv run python tools/decision_bench_clean.py           # all scenarios
       uv run python tools/decision_bench_clean.py --n 2     # repeat each 2x
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain, CLEAN_METHODS_PATH, parse_method_cards  # noqa: E402
from harness.cli import CLEAN_TASK_CONTEXT  # noqa: E402

# Synthetic learning curves (0.71xx scale — deliberately NOT the real benchmark
# scale, so no measured value can be mistaken for or leaked as campaign data).
CURVE_PEAKED = [{"epoch": e, "train_loss": 0.55 - 0.01 * e,
                 "val_gauc": 0.755 + 0.002 * min(e, 4) - 0.001 * max(0, e - 4),
                 "val_primary": 0.710 + 0.001 * min(e, 4) - 0.0006 * max(0, e - 4)}
                for e in range(1, 11)]
CURVE_NONE: list = []

SCREEN = {"hyperparam-random-search", "mechanism-screen"}
ENSEMBLE = {"seed-ensemble", "seed-architecture-ensemble"}
RANKING = {"bpr-hybrid", "dndcg-lambda", "listwise-softmax", "lightgbm-lambdarank"}
AUX = {"mtl-shared-bottom", "ordinal-watch-ratio-fm", "cwm-censored-fm"}
BROAD_SAFE = SCREEN | {"item-aggregates", "session-time-features",
                       "regularization-schedule"}
NARROW_BETS = {"finalmlp", "covisit-svd-init", "duration-regime-heads",
               "dndcg-lambda", "cwm-censored-fm", "mtl-shared-bottom"}

SCENARIOS = [
    dict(
        name="close_across_families",
        note="strong single model, single-model moves stopped clearing the bar, "
             "streak building; strategy property tested: close by combining "
             "DIFFERENT mechanism families rather than re-seeding one family or "
             "spending the last iteration on another small treatment",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED (sigma=0.0004)',
            'node_001 [draft] "hyperparam-random-search" primary=0.7161 ACCEPTED (+0.0056)',
            'node_002 [draft] "bpr-hybrid" primary=0.7170 ACCEPTED (+0.0009, below epsilon: strike 1)',
            'node_003 [draft] "recency-weighting" primary=0.7166 REJECTED (strike 2)',
        ],
        history=CURVE_PEAKED,
        streak={"no_improve_streak": 2, "iterations_done": 4, "max_iters": 16},
        good={"heterogeneous-ensemble-design"},
        bad={"regularization-schedule", "item-aggregates", "session-time-features",
             "finalmlp", "covisit-svd-init", "dndcg-lambda"},
    ),
    dict(
        name="opening_screen_first",
        note="one baseline node exists; strategy property tested: open with a "
             "broad cheap screen, not a single narrow mechanism bet",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED (sigma=0.0004)',
        ],
        history=CURVE_PEAKED,
        streak={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 16},
        good=SCREEN,
        bad=NARROW_BETS,
    ),
    dict(
        name="exploit_own_ranking_win",
        note="its OWN ledger shows a pairwise-ranking loss cleared acceptance by "
             "a wide margin; good = compose/tune within that direction, bad = "
             "ignore the ledger for an unrelated narrow bet",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED',
            'node_001 [draft] "bpr-hybrid" primary=0.7168 ACCEPTED (+0.0063, well above epsilon)',
        ],
        history=CURVE_PEAKED,
        streak={"no_improve_streak": 0, "iterations_done": 2, "max_iters": 16},
        good=(RANKING - {"bpr-hybrid"}) | {"hyperparam-random-search",
                                           "regularization-schedule"},
        bad={"covisit-svd-init", "mtl-shared-bottom", "cwm-censored-fm",
             "duration-regime-heads", "content-features"},
    ),
    dict(
        name="dead_family_twice",
        note="auxiliary-outcome heads rejected twice in its own journal; bad = "
             "retry the same mechanism family a third time",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED',
            'node_001 [draft] "mtl-shared-bottom" primary=0.7096 REJECTED (aux heads hurt)',
            'node_002 [draft] "ordinal-watch-ratio-fm" primary=0.7101 REJECTED (aux target flat)',
        ],
        history=CURVE_PEAKED,
        streak={"no_improve_streak": 2, "iterations_done": 3, "max_iters": 16},
        good=BROAD_SAFE | {"dcn-lite", "recency-weighting", "swa-ema"},
        bad=AUX,
    ),
    dict(
        name="blind_telemetry",
        note="parent emitted no learning curve; good = honest insufficient-"
             "telemetry diagnosis plus a low-risk broad move, bad = a treatment "
             "chosen on a guessed pathology",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED',
            'node_001 [draft] "session-time-features" primary=0.7112 ACCEPTED (no curve logged)',
        ],
        history=CURVE_NONE,
        streak={"no_improve_streak": 0, "iterations_done": 2, "max_iters": 16},
        good=SCREEN | {"item-aggregates", "regularization-schedule"},
        bad={"embedding-dim-down", "duration-regime-heads", "finalmlp",
             "covisit-svd-init"},
    ),
    dict(
        name="late_run_ensemble_close",
        note="strong champion, streak building, few iterations left; good = "
             "close with a diverse ensemble of the champion family",
        journal=[
            'node_000 [baseline] draft "baseline FM" primary=0.7105 ACCEPTED',
            'node_002 [draft] "regularization-schedule" primary=0.7139 ACCEPTED',
            'node_005 [improve] "hyperparam-random-search" primary=0.7160 ACCEPTED',
            'node_008 [improve] "item-aggregates" primary=0.7163 REJECTED (below epsilon)',
            'node_010 [improve] "recency-weighting" primary=0.7158 REJECTED',
        ],
        history=CURVE_PEAKED,
        streak={"no_improve_streak": 2, "iterations_done": 12, "max_iters": 14},
        good=ENSEMBLE,
        bad={"finalmlp", "covisit-svd-init", "dcn-lite", "mtl-shared-bottom",
             "dndcg-lambda", "cwm-censored-fm"},
    ),
]


# ---------------------------------------------------------------------------
# REAL-STATE TRACK (--real): replay the REAL frozen decision states from
# tools/decision_bench.py against the CLEAN brain. The real journals (with
# their real metric values) are imported AT RUNTIME and presented as the run's
# OWN ledger — a clean agent legitimately sees its own journal's numbers; the
# taboo is campaign knowledge in the LIBRARY, which stays METHODS_CLEAN.md.
# Scoring stays property-based, translated to clean card ids per scenario.
# This file contains no campaign values (fixtures are imported, not inlined).
# ---------------------------------------------------------------------------

_ENS_CROSS = {"heterogeneous-ensemble-design", "seed-architecture-ensemble"}
_ENS_ANY = _ENS_CROSS | {"seed-ensemble"}
_NARROW = {"mtl-shared-bottom", "cwm-censored-fm", "dndcg-lambda",
           "covisit-svd-init", "duration-regime-heads", "embedding-dim-down"}

REAL_STATE_RUBRIC = {
    # scenario name -> (good clean ids, bad clean ids); property in comment
    "strong_opener_full_streak": (  # exploit the accepted direction / big composite, not a small atom
        {"seq-deepfm-composite", "context-stratified-pairs",
         "hyperparam-random-search"} | _ENS_CROSS, _NARROW),
    "two_strikes_on_good_base": (  # only sane move is an ensemble close
        _ENS_ANY, {"regularization-schedule", "embedding-dim-down",
                   "recency-weighting", "listwise-regime"}),
    "blind_telemetry": (  # no curve -> honest diagnosis + broad low-risk move
        SCREEN | {"item-aggregates", "session-time-features",
                  "regularization-schedule"},
        {"embedding-dim-down", "finalmlp", "duration-regime-heads",
         "covisit-svd-init"}),
    "dead_family_temptation": (  # journal shows aux family dead twice: do not retry it
        SCREEN | {"dcn-lite", "bpr-hybrid", "regularization-schedule",
                  "recency-weighting"}, AUX),
    "opening_expected_value": (  # full clock, fresh baseline: broad opener, not a narrow bet
        SCREEN | {"regularization-schedule"}, _NARROW - {"embedding-dim-down"}),
    "endgame_eps_math": (  # streak 2, must clear eps THIS iteration: cross-family close
        _ENS_CROSS, {"seed-ensemble", "regularization-schedule",
                     "recency-weighting", "swa-ema", "item-aggregates"}),
    "endgame_margin_not_reach": (  # pick the move whose evidence clears eps WITH margin
        _ENS_CROSS, _NARROW),
    "endgame_unspent_package_trap": (  # near-ceiling package is a trap; close anyway
        _ENS_ANY, {"finalmlp", "covisit-svd-init", "duration-regime-heads",
                   "embedding-dim-down"}),
    "f4r_exact_state": (  # the real f4r prompt state: margin-maximal close
        _ENS_CROSS, {"mtl-shared-bottom", "cwm-censored-fm", "dndcg-lambda",
                     "listwise-regime"}),
    "close_rejected_strengthen_first": (  # failed-confirm close: strengthen members, do not re-roll
        {"seq-deepfm-composite", "dcn-lite", "bpr-hybrid",
         "context-stratified-pairs", "hyperparam-random-search"}, _ENS_ANY),
    "chain_open": (  # swept opener first
        {"hyperparam-random-search"}, {"seed-ensemble", "seed-architecture-ensemble",
                                       "heterogeneous-ensemble-design", "recency-weighting"}),
    "chain_after_strong_opener": (  # compound a NEW mechanism before closing; in the
        # clean world the swept opener is the logloss FM, so a pairwise objective
        # (bpr-hybrid) is a new mechanism, as is the sampler
        {"context-stratified-pairs", "bpr-hybrid", "dcn-lite"},
        {"seed-ensemble", "seed-architecture-ensemble", "heterogeneous-ensemble-design",
         "regularization-schedule", "recency-weighting"}),
    "chain_after_ctx_tie": (  # close over own lineage (or one more rider)
        {"seed-ensemble", "seed-architecture-ensemble", "heterogeneous-ensemble-design"},
        {"regularization-schedule", "recency-weighting", "bpr-hybrid", "dcn-lite",
         "hyperparam-random-search"}),
    "farm_close_uniquely_right": (  # two families near ceiling + plateau: cross-family close
        _ENS_CROSS, {"seed-ensemble", "regularization-schedule",
                     "recency-weighting", "swa-ema"}),
}


def real_state_scenarios():
    from tools.decision_bench import SCENARIOS as REAL  # runtime import keeps this file value-free
    out = []
    for sc in REAL:
        rubric = REAL_STATE_RUBRIC.get(sc["name"])
        if rubric is None:
            continue
        good, bad = rubric
        # the clean brain has never seen full-library ids: render the journal in
        # clean-card vocabulary (same events, its own names for them)
        journal = [line.replace("package-dial-sweep",
                                "hyperparam-random-search (wide dial sweep: lr, weight decay, dropout, k)")
                   for line in sc["journal"]]
        out.append(dict(sc, journal=journal, good=set(good), bad=set(bad)))
    return out


def build_clean_brain() -> Brain:
    menu = CLEAN_TASK_CONTEXT.format(dataset="pure")
    brain = Brain(menu, provider="openai", knowledge_mode="clean")
    brain.methods_text = CLEAN_METHODS_PATH.read_text()
    brain.method_cards = parse_method_cards(brain.methods_text)
    return brain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--real", action="store_true",
                    help="replay the REAL frozen states (imported at runtime) "
                         "against the clean brain, property-scored in clean ids")
    args = ap.parse_args()
    brain = build_clean_brain()
    scenarios = real_state_scenarios() if args.real else SCENARIOS
    score = {"good": 0, "ok": 0, "bad": 0}
    for sc in scenarios:
        for rep in range(args.n):
            sel = brain.select_method(
                sc["journal"], sc["history"], sc["streak"],
                excluded_families=[], dataset="pure", prior_runs=None)
            pick = sel.get("chosen_method_id")
            verdict = ("good" if pick in sc["good"]
                       else "bad" if pick in sc["bad"] else "ok")
            score[verdict] += 1
            print(f"[{sc['name']}] rep{rep}: {pick} -> {verdict.upper()}"
                  + (f"  (diag: {sel.get('diagnosis')})" if sel.get('diagnosis') else ""))
            if verdict == "bad":
                print(f"    why: {(sel.get('why') or '')[:180]}")
    total = sum(score.values())
    tag = "CLEAN BENCH (REAL STATES)" if args.real else "CLEAN BENCH"
    out = "decision_bench_clean_real_result.json" if args.real else "decision_bench_clean_result.json"
    print(f"\n{tag}: {score['good']}/{total} good, {score['ok']} neutral, "
          f"{score['bad']} bad")
    json.dump(score, open(ROOT / "logs" / out, "w"))


if __name__ == "__main__":
    main()
