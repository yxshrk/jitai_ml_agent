"""Proposer/fixer FIDELITY bench: same decision, different coder — measured by
actually running the generated code.

Fixture: the exact ff1 iteration-1 state (clean brain, mechanism-screen selected
on a fresh baseline) where the live proposer's code failed the smoke sanity gate
(its own baseline reproduction scored ~0.5901 vs the real baseline probe 0.6018).
For each (proposer/fixer model, effort) config, n reps:
  propose -> run generated script at SMOKE_EPOCHS=1 -> gate vs baseline probe;
  on gate failure, ONE fix attempt with the SAME model -> rerun.
Scores: gate pass rate + achieved primaries (mean/spread) per config.

Usage: uv run python tools/bench_proposer_fidelity.py --run-dir logs/run_ff1 --n 3
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain  # noqa: E402
from harness.cli import CLEAN_TASK_CONTEXT  # noqa: E402

CONFIGS = [
    {"name": "sol-medium", "model": "gpt-5.6-sol", "effort": "medium"},   # control (live config)
    {"name": "sol-high", "model": "gpt-5.6-sol", "effort": "high"},
    {"name": "terra-medium", "model": "gpt-5.6-terra", "effort": "medium"},
]
GATE_MARGIN = 0.005


def load_fixture(run_dir: Path):
    journal = [json.loads(l) for l in (run_dir / "journal.jsonl").open()]
    journal_lines = [
        f'node_000 [baseline] draft "baseline FM" primary={journal[0]["metrics"]["primary"]:.4f} ACCEPTED'
    ]
    sel_prompt = sorted(run_dir.glob("prompts/*_selector.md"))[0].read_text()
    reply = sel_prompt.split("## REPLY", 1)[1]
    start, end = reply.find("{"), reply.rfind("}")
    selection = json.loads(reply[start:end + 1])
    parent_code = (run_dir / "nodes/000.py").read_text()
    ref = json.loads((run_dir / "smoke_reference.json").read_text())
    return journal_lines, selection, parent_code, ref


def make_brain(model: str) -> Brain:
    brain = Brain(CLEAN_TASK_CONTEXT.format(dataset="pure"), provider="openai",
                  knowledge_mode="clean")
    brain.models = dict(brain.models)
    brain.models["proposer"] = model
    brain.models["fixer"] = model
    return brain


def run_script(code: str, out_dir: Path, data_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "script.py"
    script.write_text(code)
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
           "SMOKE_EPOCHS": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "NODE_TIMEOUT_S": "900"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--data-dir", str(data_dir),
             "--out-dir", str(out_dir), "--seed", "42"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": "timeout 900s"}
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "?").splitlines()[-6:])
        return {"ok": False, "why": f"crash: {tail[-300:]}"}
    mp = out_dir / "metrics.json"
    if not mp.exists():
        return {"ok": False, "why": "no metrics.json"}
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError as exc:
        return {"ok": False, "why": f"bad metrics.json: {exc}"}
    return {"ok": True, "primary": m.get("primary"), "gauc": m.get("gauc")}


def gate(result: dict, ref_primary: float) -> str | None:
    if not result["ok"]:
        return result["why"]
    g, p = result.get("gauc"), result.get("primary")
    if not isinstance(p, (int, float)):
        return "non-numeric primary"
    if isinstance(g, (int, float)) and g < 0.5:
        return f"gauc {g:.4f} below chance"
    if p < ref_primary - GATE_MARGIN:
        return f"primary {p:.4f} more than {GATE_MARGIN} below baseline probe {ref_primary:.4f}"
    return None


def one_rep(cfg, rep, fixture, data_dir, work):
    journal_lines, selection, parent_code, ref = fixture
    os.environ["AGENT_REASONING_EFFORT"] = cfg["effort"]  # read per-call by the backend
    brain = make_brain(cfg["model"])
    out = {"config": cfg["name"], "rep": rep}
    t0 = time.time()
    try:
        spec = brain.propose(
            journal_lines, "draft", "node_000", parent_code,
            method_selection=selection,
            streak_state={"no_improve_streak": 0, "iterations_done": 1, "max_iters": 12},
            context_mode="compact")
        code = spec.get("code")
        if not code:
            raise ValueError("no code in proposal")
    except Exception as exc:
        out.update(stage="propose", verdict=f"propose-error: {str(exc)[:160]}")
        return out
    rep_dir = work / f"{cfg['name']}_rep{rep}"
    result = run_script(code, rep_dir / "first", data_dir)
    why = gate(result, ref["primary"])
    out["first"] = {"primary": result.get("primary"), "fail": why}
    if why is None:
        out.update(verdict="PASS-first", primary=result["primary"], llm_s=round(time.time() - t0))
        return out
    try:
        fixed = brain.fix(code, f"smoke sanity gate: {why}")
    except Exception as exc:
        out.update(verdict=f"FAIL (fixer error: {str(exc)[:120]})")
        return out
    result2 = run_script(fixed, rep_dir / "fixed", data_dir)
    why2 = gate(result2, ref["primary"])
    out["fixed"] = {"primary": result2.get("primary"), "fail": why2}
    out.update(verdict="PASS-after-fix" if why2 is None else f"FAIL ({why2[:80]})",
               primary=result2.get("primary"), llm_s=round(time.time() - t0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=ROOT / "logs/run_ff1")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--config", default=None, help="run only this named config")
    args = ap.parse_args()
    fixture = load_fixture(args.run_dir)
    data_dir = args.run_dir / "workspace"
    work = ROOT / "logs/bench_proposer_fidelity"
    work.mkdir(parents=True, exist_ok=True)
    print(f"baseline probe primary={fixture[3]['primary']:.4f}; gate margin {GATE_MARGIN}")
    configs = [c for c in CONFIGS if args.config in (None, c["name"])]
    jobs = [(cfg, rep) for cfg in configs for rep in range(args.n)]
    results = []
    # LLM calls parallel; script runs serialize inside reps naturally (2 pool workers
    # keep CPU shared with anything else running on this machine).
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(one_rep, cfg, rep, fixture, data_dir, work): (cfg["name"], rep)
                for cfg, rep in jobs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            print(f"[{r['config']} rep{r['rep']}] {r['verdict']}"
                  + (f" primary={r.get('primary'):.4f}" if isinstance(r.get('primary'), float) else ""))
    print("\nSUMMARY")
    for cfg in configs:
        rs = [r for r in results if r["config"] == cfg["name"]]
        passes = [r for r in rs if str(r["verdict"]).startswith("PASS")]
        prims = [r["primary"] for r in passes if isinstance(r.get("primary"), float)]
        spread = (f" primaries {min(prims):.4f}-{max(prims):.4f}" if prims else "")
        print(f"  {cfg['name']}: {len(passes)}/{len(rs)} pass"
              f" ({sum(1 for r in passes if r['verdict']=='PASS-first')} first-try){spread}")
    (work / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
