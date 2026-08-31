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
]


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
