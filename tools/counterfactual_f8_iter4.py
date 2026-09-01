"""Counterfactual: replay f8's exact post-node_003 state through the live
selector and proposer to see what iteration 4 would have been without the 503.
Decision + emitted plan only — no training. Experiment lineage, evidence only."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain, normalize_proposal_envelope
from harness.loop import normalize_history

rows = [json.loads(l) for l in open(ROOT / "logs/run_f8_journal.jsonl")]
sigma_line = f'node_000 [baseline] draft "baseline FM" primary={rows[0]["metrics"]["primary"]:.4f} ACCEPTED (sigma=0.0004)'
journal = [sigma_line]
for r in rows[1:4]:
    status = "ACCEPTED" if r["accepted"] else "REJECTED"
    journal.append(f'{r["node_id"]} [<-{r["parent"]}] {r["action"]} "{r["hypothesis"]}" '
                   f'primary={r["metrics"]["primary"]:.4f} {status}')
hist = normalize_history((json.load(open(ROOT / "logs/run_f8_node002_metrics.json")).get("history") or []))
streak = {"no_improve_streak": 2, "n_converge": 3, "iters_left": 12}

brain = Brain((ROOT / "MENU.md").read_text(), provider="openai", knowledge_mode="full")
for rep in range(1):
    sel = brain.select_method(journal, hist, streak, excluded_families=[],
                              dataset="pure", prior_runs=None)
    print(f"\n=== rep{rep} SELECTION: {sel.get('chosen_method_id')} (diag {sel.get('diagnosis')})")
    print("why:", str(sel.get("why"))[:300])
    spec = brain.propose(journal, "improve", "node_002",
                         (ROOT / "logs/run_f8_nodes/002.py").read_text(),
                         method_selection=sel, streak_state=streak,
                         parent_history=hist, context_mode="compact",
                         parent_code_path="logs/run_f8_nodes/002.py")
    spec = normalize_proposal_envelope(spec)
    if spec.get("execution_kind") == "farm_close":
        plan = spec["farm_close_plan"]
        print("PLAN: farm_close,", len(plan["members"]), "members:")
        for m in plan["members"]:
            src = "ANCHOR(script)" if "script_source" in m else "generated-code"
            print(f"  - {m['family']} [{src}] seed={m.get('seed')}")
        print("blend:", json.dumps(plan.get("blend"))[:200])
    else:
        kind = "edits(" + str(len(spec.get("edits") or [])) + " hunks)" if spec.get("edits") else "whole-code"
        print(f"PLAN: script via {kind}; hypothesis: {str(spec.get('hypothesis'))[:200]}")
    print("expected_delta:", spec.get("expected_delta"))
    out = ROOT / f"logs/counterfactual_f8_iter4_spec{rep}.json"
    out.write_text(json.dumps({"selection": sel, "spec": spec}, indent=1))
    print("saved:", out)
