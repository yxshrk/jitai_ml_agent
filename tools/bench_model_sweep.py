"""Decision-bench sweep across selector models and reasoning efforts.

Runs the full-knowledge decision bench once per (model, effort) cell and
tabulates good/ok/bad, so selector quality can be compared across the
gpt-5.6 family variants and effort tiers. Uses the same scenarios and
scoring as tools/decision_bench.py; costs ~5 selector calls per cell.

Usage: uv run python tools/bench_model_sweep.py \
           --models gpt-5.6-sol gpt-5.6-luna gpt-5.6-terra \
           --efforts low medium
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_cell(model: str, effort: str, temp: str | None) -> dict:
    os.environ["AGENT_REASONING_EFFORT"] = effort
    if temp is not None and temp != "default":
        os.environ["AGENT_TEMPERATURE"] = temp
    else:
        os.environ.pop("AGENT_TEMPERATURE", None)
    # fresh import context so the effort env is read per cell
    for name in list(sys.modules):
        if name.startswith(("agent.", "tools.decision_bench")):
            del sys.modules[name]
    from agent.brain import Brain  # noqa: E402
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "decision_bench", ROOT / "tools" / "decision_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    menu = (ROOT / "MENU.md").read_text()
    brain = Brain(menu, provider="openai", knowledge_mode="full")
    brain.models["selector"] = model
    score = {"good": 0, "ok": 0, "bad": 0}
    picks = []
    for sc in bench.SCENARIOS:
        try:
            sel = brain.select_method(
                sc["journal"], sc["history"], sc["streak"],
                excluded_families=[], dataset="pure", prior_runs=None)
            pick = sel.get("chosen_method_id")
            verdict = ("good" if pick in sc["good"]
                       else "bad" if pick in sc["bad"] else "ok")
        except Exception as exc:  # a model variant may not exist or may fail
            pick, verdict = f"ERROR: {str(exc)[:80]}", "error"
            score.setdefault("error", 0)
        score[verdict] = score.get(verdict, 0) + 1
        picks.append({"scenario": sc["name"], "pick": pick, "verdict": verdict})
        print(f"  [{sc['name']}] {pick} -> {verdict.upper()}")
    return {"model": model, "effort": effort, "temperature": temp or "default",
            "score": score, "picks": picks,
            "usd": round(getattr(brain, "usd_total", 0.0), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt-5.6-sol"])
    ap.add_argument("--efforts", nargs="+", default=["medium"])
    ap.add_argument("--temps", nargs="+", default=["default"])
    args = ap.parse_args()
    results = []
    for model in args.models:
        for effort in args.efforts:
            for temp in args.temps:
                print(f"\n=== {model} @ effort={effort} temp={temp} ===")
                results.append(run_cell(model, effort, temp))
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'model':22}{'effort':9}{'temp':9}{'good':6}{'ok':5}{'bad':5}{'err':5}{'usd':7}")
    for r in results:
        s = r["score"]
        print(f"{r['model']:22}{r['effort']:9}{str(r['temperature']):9}{s.get('good',0):<6}{s.get('ok',0):<5}"
              f"{s.get('bad',0):<5}{s.get('error',0):<5}{r['usd']:<7}")
    out = ROOT / "logs" / "bench_model_sweep.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
