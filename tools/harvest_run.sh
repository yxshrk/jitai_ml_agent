#!/bin/bash
# Pull a finished remote run's logs to the laptop (no workspace/checkpoints) and
# print the summary line to paste into logs/RUNS.md. Commit + push afterwards.
#   usage: bash tools/harvest_run.sh <run_name> [ssh_host=gpubox] [remote_repo=mle-agent]
RUN="$1"; HOST="${2:-gpubox}"; REPO="${3:-mle-agent}"
[ -z "$RUN" ] && { echo "usage: harvest_run.sh <run_name> [host] [repo]"; exit 2; }
rsync -az --exclude workspace --exclude '*.pt' "$HOST:$REPO/logs/$RUN" logs/ || exit 1
python3 - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
try:
    d = json.load(open(f"logs/{run}/summary.json"))
    print(f"| {run.replace('run_','')} | <machine> | {d['stop_reason']} | {d['iterations']} | "
          f"{d['best_metrics']['primary']:.5f} | <one-line lesson> ({d['tokens_total']} tokens, {d['wall_s']/60:.0f} min) |")
except FileNotFoundError:
    print(f"| {run.replace('run_','')} | <machine> | KILLED | - | - | <why> |")
PY
