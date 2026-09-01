# Results table & reported resource usage (required deliverable 4)

> NOTE: numbers below describe the currently designated runs; if a final-wave run
> supersedes a designation before submission, regenerate this file (values come from
> the runs' summary.json — nothing here is hand-estimated except where labeled).

## Results (validation-best at convergence; hidden test scored by organizers)

| Benchmark | Official baseline (valid) | Ours (valid) | Absolute delta |
|---|---|---|---|
| KuaiRand-Pure (required) | GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 | GAUC 0.6728 / nDCG@5 0.5383 / **primary 0.605575** | **+0.00398** primary (+0.0054 GAUC, +0.0026 nDCG@5) |
| KuaiRand-1K (bonus) | n/a (no official baseline; our tuned single: 0.6208) | **primary 0.66892** (designated, triple-audited) | +0.048 vs single-model start |
| KuaiRand-27K (bonus, out-of-protocol scaling demo) | n/a | primary 0.67263 | — (GPU demo, not an agent run) |

Designated runs: Pure = `logs/run_bigclock_07` (no executable seed; stopped under the ε=0.002/N=3 rule; harness at commit `dc354b6`, tag `designated-run-harness` — later commits are post-designation hardening);
1K = `logs/run_omega_1k` (causal session features; faithful A-form artifact). Submission CSVs:
`evidence/test_submission_pure.csv` (170,588 rows) and `evidence/test_submission_1k_faithful.csv`
(4,132,081 rows), both built by
training on the train split only and validated with the official checker; test labels
are never read anywhere in the pipeline (see tools/predict_test*.py guards).

## Resource usage — designated runs (what the rules ask to be reported)

| | Pure (run_bigclock_07) | 1K (run_omega_1k) |
|---|---|---|
| LLM tokens (in+out) | 115,315 | 320,048 |
| Agent wall-clock | 17.0 min (1,019 s) | 344.8 min (20,690 s) |
| Iterations used (of 50) | 6 | 8 |
| GPU-hours in the run | 0 (CPU only) | ~5.7 (RTX 4090; wall-clock is the scored measure) |
| Mid-run human actions | 0 (team attestation + no intervention events journaled) | 0 |

## Full development disclosure (voluntary)

The complete campaign comprises ~146 completed runs (139 at the 31 Aug snapshot +
the 1 Sep hardening wave; post-mortem: evidence/POSTMORTEM_1SEP_FINAL_RUNS.md) plus
incomplete/aborted ones (machine-generated inventory: logs/RUNS_INVENTORY.md,
evidence/run_inventory.csv): aggregate ≈10.6M LLM tokens and ≈143 run-hours
wall-clock across all completed runs, ≈US$122 real LLM spend for the whole campaign
(per-machine ledgers overcount by design — conservative pricing, rounded up; the
provider console totals are the authority) plus ≈US$5 of rented CPU-pod time. The designated runs' own usage is the scored figure; the
campaign totals are disclosed because cross-run knowledge from the campaign
informed the designated runs' method cards. GPU usage (RTX 4090): the designated 1K run (~5.7 GPU-hours) plus development and
bonus work (hyperparameter farms, 27K scaling demo, some development runs) —
estimated ~18 GPU-hours total; the designated Pure run used none. Wall-clock is the scored measure per
the brief; GPU-hours are reported for completeness.
