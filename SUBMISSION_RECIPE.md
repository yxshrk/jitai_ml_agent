# Final submission recipe (drafted 02:20 Sat, confirm Sunday)

## Pure (primary)
Best validation: **0.60577** — rank-average ensemble of 5 frozen-stack seeds
{46, 74, 93, 91, 60}, greedy-selected on validation from the 60-seed farm
(coral, farm_results.jsonl; optimizer log ens_opt.log).
- Baseline comparison: +0.0042 (vs 0.6016). Best single seed 0.6053; seed-42 0.6050.
- DECISION (made Sat evening, three-signal triangulation): KEEP the greedy 5-seed
  ensemble. Signals: full-val 0.6058 (best); late-val (Apr 26-28) inconclusive —
  collapses to single-seed selection on 3 days of data (selection overfit, discounted);
  random-exposure TEST-WINDOW probe (log_random Apr 29-May 8, evaluation-only, legal —
  not part of hidden test): all candidates within ~0.002, ensemble at the top, stable
  transfer for every candidate. tools/RANDOM_PROBE.md has the table. The ensemble is
  at/near the top of every signal and carries the variance-reduction rationale.
- Test-time procedure: train chosen seeds on TRAIN ONLY (organizer ruling), predict
  test with each, per-user rank-average, submit via evidence/submission.py +
  official submit.py --check. ONE test touch.

## 1K (bonus)
Frozen stack transfers: seeds 42/43/44 = 0.6134/0.6090/0.6156 (mean 0.6127,
spread larger than Pure). Half-life 7 confirmed optimal on 1K (3 and 14 both hurt).
Bonus submission: 3-5 seed rank-average, same procedure, ~6 min/seed CPU.
