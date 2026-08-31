"""Evidence probe (NOT a scored run): re-run ruby_x1's agent-authored composite
node (nodes/003.py, accepted 0.60443) across 3 seeds and rank-average — measuring
the ceiling of the composite+close chain inside our stack. Same class as our
replication audits: agent-authored code, team-run measurement, disclosed.
Usage: python tools/probe_composite_close.py  (on ruby; needs run_ruby_x1 logs)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate

NODE = ROOT / "logs/run_ruby_x1/nodes/003.py"
WS = ROOT / "logs/run_ruby_x1/workspace"
OUT = ROOT / "logs/probe_composite_close"
SEEDS = (42, 43, 44)


def main():
    val = np.load(WS / "val.npz")
    members = []
    for seed in SEEDS:
        out = OUT / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "predictions.csv").exists():
            r = subprocess.run([sys.executable, str(NODE), "--data-dir", str(WS),
                                "--out-dir", str(out), "--seed", str(seed)],
                               env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin",
                                    "HOME": str(Path.home())},
                               capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                sys.exit(f"seed {seed} failed: {r.stderr[-400:]}")
        scores = np.empty(len(val["y"]))
        import csv
        with open(out / "predictions.csv") as fh:
            rd = csv.reader(fh); next(rd)
            for row in rd:
                scores[int(row[0])] = float(row[3])
        m = evaluate(val["user"], val["y"].astype(int), scores)
        print(f"seed {seed}: primary {m['primary']:.6f}")
        members.append(scores)
    ranks = [np.argsort(np.argsort(s, kind="stable"), kind="stable") for s in members]
    ens = np.mean(ranks, axis=0)
    m = evaluate(val["user"], val["y"].astype(int), ens)
    print(f"3-SEED RANK-AVERAGE (composite+close chain ceiling): "
          f"primary {m['primary']:.6f} gauc {m['GAUC']:.6f} ndcg5 {m['nDCG@5']:.6f}")
    json.dump({"seeds": SEEDS, "ensemble_primary": m['primary']},
              open(OUT / "result.json", "w"))


if __name__ == "__main__":
    main()
