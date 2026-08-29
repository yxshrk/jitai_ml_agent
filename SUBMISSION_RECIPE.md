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

## Pure UPDATE (Sat night): extended 159-seed pool greedy = 0.60602 (11 members:
s199, seedfarm_74, seedfarm_46, s196, s109, s191, s147, seedfarm_89, s105, s126,
seedfarm_100 — seedext seeds are 102-200 of the same frozen config). DESIGNATION
PENDING Sunday-morning random-probe transfer check (11-member made ~10 more
val-guided picks than the 5-member; pick by probe + parsimony, not val digit).

## 1K (bonus) — FROZEN Sat evening
Recipe: 5-seed per-user rank-average of the 1K-tuned config
(lr 0.00168, dropout 0.21, wd 3.7e-5, k 24, recency half-life 7, 6 epochs;
zoo/frozen_stack_1k.py), seeds {42,43,44,45,46}.
Validated: singles 0.6073-0.6216 (wide seed variance is why ensembling pays);
**ensemble 0.6323 valid primary**. Arc: default transfer 0.6134 -> tuned single
0.6214 -> ensemble 0.6323. No official 1K baseline exists; we report absolutes.
Test-time: train 5 seeds on train window only, predict test, rank-average, submit.
