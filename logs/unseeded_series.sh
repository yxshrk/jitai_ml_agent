#!/bin/bash
cd /Users/rohan/Documents/progwork/techjam2026/mle-agent
for pair in "05 06" "07 08" "09 10"; do
  set -- $pair
  for n in $1 $2; do
    uv run python -u -m harness.cli run --data-dir data/real_ws \
      --baseline-script zoo/baseline_ws.py --accept-floor 0.0009 \
      --timeout-s 2400 --max-iters 12 --context-mode compact \
      --run-dir logs/run_unseeded_$n > logs/run_unseeded_$n.out 2>&1 &
  done
  wait
done
echo SERIES_DONE > logs/unseeded_series.done
