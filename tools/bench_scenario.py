"""Run one named decision-bench scenario N times (with plan-emission check when
the scenario expects a plan). Usage: uv run python tools/bench_scenario.py <name> [reps]"""
import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain  # noqa: E402

name = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
spec = importlib.util.spec_from_file_location("db", ROOT / "tools" / "decision_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
brain = Brain((ROOT / "MENU.md").read_text(), provider="openai", knowledge_mode="full")
sc = [x for x in bench.SCENARIOS if x["name"] == name][0]
for rep in range(reps):
    sel = brain.select_method(sc["journal"], sc["history"], sc["streak"],
                              excluded_families=[], dataset="pure", prior_runs=None)
    pick = sel.get("chosen_method_id")
    verdict = "GOOD" if pick in sc["good"] else "BAD" if pick in sc["bad"] else "OK"
    print(f"{name} rep{rep}: {pick} -> {verdict} | diag: {sel.get('diagnosis')}", flush=True)
    if verdict == "BAD":
        print("   why:", (sel.get("why") or "")[:300], flush=True)
    if sc.get("expect_plan") and pick in sc["good"]:
        print("   emission:", bench.check_plan_emission(brain, sc, sel), flush=True)
