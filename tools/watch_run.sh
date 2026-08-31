#!/bin/bash
# Watch a remote autonomous run and exit on the first meaningful event, so a
# coding agent can arm it in the background and be woken up.
#   usage: bash tools/watch_run.sh <run_name> [ssh_host=gpubox] [remote_repo=~/mle-agent]
# Events printed: "journal A -> B" (an iteration was recorded), DONE (summary.json
# exists), DEAD (no process AND no file written for 20 min). Re-arm after each event.
RUN="$1"; HOST="${2:-gpubox}"; REPO="${3:-~/mle-agent}"
[ -z "$RUN" ] && { echo "usage: watch_run.sh <run_name> [host] [repo]"; exit 2; }
LAST=""
while true; do
  out=$(ssh -o ConnectTimeout=15 "$HOST" "cd $REPO; d=logs/$RUN
    [ -f \$d/summary.json ] && { echo DONE; exit; }
    alive=\$(ps aux | grep \"[r]un_dir logs/$RUN\" | wc -l)
    fresh=\$(find \$d -type f -mmin -20 2>/dev/null | wc -l)
    if [ \"\$alive\" -eq 0 ] && [ \"\$fresh\" -eq 0 ]; then echo DEAD; exit; fi
    wc -l < \$d/journal.jsonl 2>/dev/null || echo 0" 2>/dev/null) || out=SSH_FAIL
  case "$out" in
    DONE|DEAD) echo "=== $RUN EVENT: $out ==="; exit 0;;
    SSH_FAIL) sleep 120;;
    *) [ -z "$LAST" ] && LAST=$out
       if [ "$out" != "$LAST" ]; then echo "=== $RUN EVENT: journal $LAST -> $out ==="; exit 0; fi;;
  esac
  sleep 90
done
