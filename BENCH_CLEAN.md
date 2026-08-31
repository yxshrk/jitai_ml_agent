# Clean decision bench (tools/decision_bench_clean.py)

Selector-only bench of the literature-only (--knowledge clean) agent against 5
frozen decision states, scored by strategy FAMILY (screen-first, exploit own
ledger, avoid dead family, honest telemetry, ensemble close). All journal
metrics in the bench are synthetic (0.71xx); no campaign results anywhere.

## Before (master clean knowledge), --n 2
- opening_screen_first: regularization-schedule x2 -> OK (no screening card existed)
- exploit_own_ranking_win: regularization-schedule x2 -> GOOD
- dead_family_twice: regularization-schedule x2 -> GOOD
- blind_telemetry: recency-weighting x2 -> OK (guessy narrow move)
- late_run_ensemble_close: swa-ema x2 -> OK (no close)
Score: 4/10 good, 6 neutral, 0 bad

## After (doctrine preamble + CLEAN_TASK_CONTEXT doctrine + kind tiers +
## hyperparam-random-search / mechanism-screen cards), --n 2
- opening_screen_first: hyperparam-random-search x2 -> GOOD
- exploit_own_ranking_win: regularization-schedule x2 -> GOOD
- dead_family_twice: regularization-schedule x2 -> GOOD
- blind_telemetry: mechanism-screen x2 (diag insufficient-telemetry) -> GOOD
- late_run_ensemble_close: seed-ensemble x2 -> GOOD
Score: 10/10 good, 0 neutral, 0 bad

Selector spend: 20 calls total (10 before + 10 after).
