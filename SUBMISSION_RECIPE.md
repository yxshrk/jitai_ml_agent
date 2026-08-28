# Final submission recipe (drafted 02:20 Sat, confirm Sunday)

## Pure (primary)
Best validation: **0.60577** — rank-average ensemble of 5 frozen-stack seeds
{46, 74, 93, 91, 60}, greedy-selected on validation from the 60-seed farm
(coral, farm_results.jsonl; optimizer log ens_opt.log).
- Baseline comparison: +0.0042 (vs 0.6016). Best single seed 0.6053; seed-42 0.6050.
- CAUTION (decide Sunday): greedy subset selection = ~15 validation reads; mild
  val-fitting risk of the kind the organizer warned about. Alternatives to compare
  before freezing: (a) greedy subset (0.60577 val), (b) simple top-5-by-val-single
  seeds, (c) all-60 average — pick by robustness argument, not val alone; expect
  (a) to shrink most on test.
- Test-time procedure: train chosen seeds on TRAIN ONLY (organizer ruling), predict
  test with each, per-user rank-average, submit via evidence/submission.py +
  official submit.py --check. ONE test touch.

## 1K (bonus)
Frozen stack transfers: seeds 42/43/44 = 0.6134/0.6090/0.6156 (mean 0.6127,
spread larger than Pure). Half-life 7 confirmed optimal on 1K (3 and 14 both hurt).
Bonus submission: 3-5 seed rank-average, same procedure, ~6 min/seed CPU.
