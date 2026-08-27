# Autonomous runs — overnight 28 Aug

All runs: 0 interventions, official convergence rule, validation only.
Champion candidate: run_real_01 node_002 (0.6042). Manual 3-seed best: zoo/best.py 0.6039±0.0010.

| run | directive | stop | iters | best valid primary | delta | wall | LLM cost |
|---|---|---|---|---|---|---|---|
| 01 | default tiers | converged | 5 | **0.6042** | **+0.0026** | 23.5m | $0.90 |
| 02 | + overfit insight | converged | 5 | 0.6040 | +0.0024 | 14.2m | $0.57 |
| 03 | watch-time (ignored: policy bug, fixed) | converged | 3 | 0.6033 | +0.0015 | 11.6m | $0.66 |
| 04 | watch-time themes (FM base) | converged | 3 | 0.6026 | +0.0008 | 15.6m | $0.63 |
| 05 | compound DCN+themes | converged | 3 | 0.6033 | +0.0015 | 12.8m | $0.68 |
