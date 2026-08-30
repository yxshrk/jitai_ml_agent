# Results table & reported resource usage (required deliverable 4)

> NOTE: numbers below describe the currently designated runs; if a final-wave run
> supersedes a designation before submission, regenerate this file (values come from
> the runs' summary.json — nothing here is hand-estimated except where labeled).

## Results (validation-best at convergence; hidden test scored by organizers)

| Benchmark | Official baseline (valid) | Ours (valid) | Absolute delta |
|---|---|---|---|
| KuaiRand-Pure (required) | GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 | GAUC 0.6728 / nDCG@5 0.5383 / **primary 0.60558** | **+0.0040** primary (+0.0054 GAUC, +0.0026 nDCG@5) |
| KuaiRand-1K (bonus) | n/a (no official baseline; our tuned single: 0.6208) | **primary 0.63874** | +0.0179 vs our single-model start |
| KuaiRand-27K (bonus, out-of-protocol scaling demo) | n/a | primary 0.67263 | — (GPU demo, not an agent run) |

Designated runs: Pure = `logs/run_bigclock_07` (fully unseeded; converged, ε=0.002/N=3);
1K = `logs/run_desig_1k_01`. Submission CSVs: `evidence/test_submission_pure.csv`
(170,588 rows) and `evidence/test_submission_1k.csv` (4,132,081 rows), both built by
training on the train split only and validated with the official checker; test labels
are never read anywhere in the pipeline (see tools/predict_test*.py guards).

## Resource usage — designated runs (what the rules ask to be reported)

| | Pure (run_bigclock_07) | 1K (run_desig_1k_01) |
|---|---|---|
| LLM tokens (in+out) | 115,315 | 101,549 |
| Agent wall-clock | 17.0 min (1,019 s) | 62.7 min (3,763 s) |
| Iterations used (of 50) | 6 | 7 |
| GPU-hours in the run | 0 (CPU only) | 0 (CPU only) |
| Manual interventions | 0 (machine-counted from journals) | 0 |

## Full development disclosure (voluntary)

~45 additional agent runs were executed during development and harness evolution
(all journals in `logs/run_*/`, indexed in `logs/RUNS.md`); typical run: 35k–120k
tokens, 10–70 min wall-clock. Aggregate development LLM spend ≈ US$95 (per-machine
ledgers). GPU usage occurred only in development and bonus work (RTX 4090:
hyperparameter farms, 27K scaling demo, some development runs) — estimated ~12
GPU-hours total, none in the designated runs. Wall-clock is the scored measure per
the brief; GPU-hours are reported for completeness.
