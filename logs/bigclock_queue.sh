#!/bin/bash
cd /Users/rohan/Documents/progwork/techjam2026/mle-agent
for n in 03 05; do
  while [ ! -f logs/run_bigclock_01/summary.json ] && ! ls logs/run_bigclock_${n}_prev 2>/dev/null; do sleep 120; done
  AGENT_REASONING_EFFORT=medium AGENT_MAX_CODE_TOKENS=24000 uv run python -u -m harness.cli run \
    --data-dir data/real_ws --baseline-script zoo/baseline_ws.py --accept-floor 0.0009 \
    --timeout-s 7200 --max-iters 12 --context-mode compact \
    --run-dir logs/run_bigclock_$n > logs/run_bigclock_$n.out 2>&1
done
echo QUEUE_DONE > logs/bigclock_queue.done
