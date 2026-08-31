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


def build_clean_brain() -> Brain:
    menu = CLEAN_TASK_CONTEXT.format(dataset="pure")
    brain = Brain(menu, provider="openai", knowledge_mode="clean")
    brain.methods_text = CLEAN_METHODS_PATH.read_text()
    brain.method_cards = parse_method_cards(brain.methods_text)
    return brain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    brain = build_clean_brain()
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
            print(f"[{sc['name']}] rep{rep}: {pick} -> {verdict.upper()}"
                  + (f"  (diag: {sel.get('diagnosis')})" if sel.get('diagnosis') else ""))
            if verdict == "bad":
                print(f"    why: {(sel.get('why') or '')[:180]}")
    total = sum(score.values())
    print(f"\nCLEAN BENCH: {score['good']}/{total} good, {score['ok']} neutral, "
          f"{score['bad']} bad")
    json.dump(score, open(ROOT / "logs/decision_bench_clean_result.json", "w"))


if __name__ == "__main__":
    main()
