"""Live test of the edits-proposal path: real proposer, real parent script.

Fixture: f7's accepted node_001 (the tuned DCN package, 419 lines) as the
champion, with a crafted improve selection (recency-weighting refit — a small,
well-understood dosage change). For each rep: propose in improve mode ->
verify the reply carries edit blocks (not a whole script) -> apply them ->
run the edited script at SMOKE_EPOCHS=1 on real_ws -> report.

Usage: uv run python tools/bench_edit_apply.py --n 3
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import Brain  # noqa: E402
from harness.loop import EditApplyError, apply_edits, rewrite_ratio  # noqa: E402

PARENT = ROOT / "logs/run_farm_f7/nodes/001.py"
JOURNAL = [
    'node_000 [baseline] draft "baseline FM" primary=0.6018 ACCEPTED (sigma=0.0004)',
    'node_001 [<-node_000] draft "package-dial-sweep" primary=0.6045 ACCEPTED (+0.0027)',
]
SELECTION = {
    "diagnosis": "data-shift",
    "chosen_method_id": "recency-weighting",
    "citation": "temporal-dynamics CF literature (Koren onward); recency-weighting card",
    "why": "The accepted package trains with a fixed recency half-life; the card "
           "and journal support refitting the half-life on the accepted regularized "
           "stack as a small, low-risk dosage change.",
    "rejected": [],
}


def run_script(code: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    script = out_dir / "script.py"
    script.write_text(code)
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
           "SMOKE_EPOCHS": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "NODE_TIMEOUT_S": "900"}
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--data-dir", str(ROOT / "data/real_ws"),
             "--out-dir", str(out_dir), "--seed", "42"],
            capture_output=True, text=True, timeout=900, cwd=str(ROOT), env=env)
    except subprocess.TimeoutExpired:
        return {"ok": False, "why": "timeout"}
    if proc.returncode != 0:
        return {"ok": False,
                "why": "crash: " + "\n".join((proc.stderr or "?").splitlines()[-4:])[-240:]}
    try:
        return {"ok": True,
                "primary": json.loads((out_dir / "metrics.json").read_text())["primary"]}
    except Exception as exc:
        return {"ok": False, "why": f"metrics: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    parent_code = PARENT.read_text()
    brain = Brain((ROOT / "MENU.md").read_text(), provider="openai", knowledge_mode="full")
    work = ROOT / "logs/bench_edit_apply"
    work.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for rep in range(args.n):
        t0 = time.time()
        tag = f"rep{rep}"
        try:
            spec = brain.propose(
                JOURNAL, "improve", "node_001", parent_code,
                method_selection=SELECTION,
                streak_state={"no_improve_streak": 0, "n_converge": 3, "iters_left": 12},
                context_mode="compact")
        except Exception as exc:
            outcomes.append((tag, f"PROPOSE-ERROR {str(exc)[:120]}"))
            print(outcomes[-1]); continue
        edits = spec.get("edits")
        if not edits:
            outcomes.append((tag, f"NO-EDITS (emitted {'code' if spec.get('code') else 'nothing'})"))
            print(outcomes[-1]); continue
        try:
            code = apply_edits(parent_code, edits)
        except EditApplyError as exc:
            outcomes.append((tag, f"APPLY-FAIL after {len(edits)} hunks: {str(exc)[:140]}"))
            print(outcomes[-1]); continue
        ratio = rewrite_ratio(parent_code, code)
        result = run_script(code, work / tag)
        verdict = (f"OK primary={result['primary']:.4f}" if result["ok"]
                   else f"RUN-FAIL {result['why'][:140]}")
        outcomes.append((tag, f"{len(edits)} hunks, kept {ratio:.0%}, {verdict}, "
                              f"{time.time()-t0:.0f}s"))
        print(outcomes[-1])
    print("\nEDIT-APPLY BENCH:",
          sum("OK primary" in o[1] for o in outcomes), f"/{args.n} end-to-end OK")


if __name__ == "__main__":
    main()
