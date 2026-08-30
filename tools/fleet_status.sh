#!/bin/bash
# Live fleet view: every active/finished agent run on laptop + coral + ruby.
# Usage: bash tools/fleet_status.sh          (one shot)
#        watch -n 60 bash tools/fleet_status.sh   (live, if watch installed)
summarize() { # $1 = label, stdin = base dirs to scan (newline sep) via ssh or local
  :
}
show_local() {
  for d in logs/run_*/; do
    n=$(basename "$d")
    if [ -f "$d/summary.json" ]; then
      python3 - "$d" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]+"/summary.json"))
print(f"  {sys.argv[1].split('/')[-2]:24s} DONE  {s['stop_reason']:14s} best={s['best_metrics']['primary']:.5f}")
PY
    else
      nodes=$(ls "$d/nodes" 2>/dev/null | wc -l | tr -d ' ')
      last=$(tail -1 "$d/journal.jsonl" 2>/dev/null | python3 -c "import json,sys
try: r=json.loads(sys.stdin.read()); print(r.get('node_id'),round((r.get('metrics') or {}).get('primary',0),5))
except: print('-')" 2>/dev/null)
      prog=$(tail -1 "$d"/node_*/progress.log 2>/dev/null | tail -1 | cut -c1-60)
      echo "  $(printf %-24s "$n") LIVE  nodes=$nodes last=[$last] ${prog:+probe: $prog}"
    fi
  done
}
echo "== laptop =="; show_local
REMOTE_SNIPPET='cd REPO 2>/dev/null || exit 0
for d in logs/run_*/; do
  n=$(basename "$d")
  if [ -f "$d/summary.json" ]; then
    python3 -c "import json;s=json.load(open(\"$d/summary.json\"));print(f\"  {\"$n\":24s} DONE  {s[\"'"'"'stop_reason'"'"'"\"]:14s} best={s[\"'"'"'best_metrics'"'"'"\"][\"'"'"'primary'"'"'"\"]:.5f}\")" 2>/dev/null
  else
    nodes=$(ls "$d/nodes" 2>/dev/null | wc -l | tr -d " ")
    prog=$(tail -1 "$d"/node_*/progress.log 2>/dev/null | tail -1 | cut -c1-60)
    echo "  $(printf %-24s "$n") LIVE  nodes=$nodes ${prog:+probe: $prog}"
  fi
done'
echo "== coral =="; ssh -o ConnectTimeout=8 pallav@coral.local "${REMOTE_SNIPPET//REPO/~\/techjam\/mle-agent}" 2>/dev/null
echo "== ruby =="; ssh -o ConnectTimeout=8 gpubox "${REMOTE_SNIPPET//REPO/~\/mle-agent}" 2>/dev/null
