---
id: ensembling-cross-session-daily-budget-prior
family: ensembling
target_component: ensembling
source: kb/data/facts.md §10.5 (attention decay with exposure position); within-user rank-fusion design
applies_when:
  - impression date and time_ms permit equal-time-safe cumulative user-day exposure features
  - session-local features reset after 30 minutes but cross-session fatigue within the same day remains plausible
  - a competitive base ranker is available to anchor a low-weight behavioral-prior blend
expected_delta: [0.0, 0.0]
expected_delta_basis: attribution from the archived wildcard was flat (single-seed Δ -0.00003, no seed confirmation),
  so no positive gain can be promised for the daily-budget expert itself
cost: 84 changed lines; one train-only aggregate table and rank fusion; measured total runtime 162 s on a ten-member parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, ensembling-multiseed-heterogeneous-rank-blend]
conflicts_with: [ensembling-attention-budget-threshold-expert-fusion, ensembling-cohort-gated-session-rank-fusion]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)]
evidence: [live_09:node_019]
---
## Claim
Fit a train-only tab-conditioned long-view prior over causal user-day exposure count and elapsed-time buckets, then
blend its within-user rank at 10% with a competitive base ranker.

## Mechanism (why it moves within-user ranking)
Cumulative impressions and elapsed time since a user's first exposure that day vary across the user's scored rows.
A tab × count × elapsed rate table can therefore represent cross-session attention depletion that a 30-minute
session reset misses. Rank normalization makes the fixed 10% fusion insensitive to score calibration.

## How to implement on node_000
1. Load `date` and `time_ms` for train, valid, and score-extra rows.
2. Add `daily_buckets(rows, ui, di, mi)` and stable-sort by user, date, time, and original row.
3. Give equal-time rows the same causal prior count using the equal-time group's start minus the user-day start.
4. Bucket prior count with `[1,3,10,30]` and elapsed milliseconds with
   `[600000,1800000,3600000,10800000,21600000]`.
5. Fit train-only tab priors smoothed by 20 observations toward the global long-view rate.
6. Fit the `tab × 5 count × 6 elapsed` rate table, smoothed by 20 observations toward each tab prior.
7. Score valid and extra rows from this table, falling back to the global rate for unseen tabs.
8. Convert base and expert scores to tie-free within-user ranks and emit
   `rank(0.9*base_rank + 0.1*expert_rank)`, using the base rank as tie-break.
9. Apply the same fusion to every validation-history checkpoint and final score-extra predictions.

## Risks / failure modes
- The archived diff included an already-proven ten-member heterogeneous BPR ensemble; only the added daily expert
  is attributable to this card, and it produced no measurable positive gain.
- Daily count may proxy logging volume or tab composition rather than transferable attention state.
- The rate table is pointwise and coarse; within-user ties are common, so deterministic base-rank tie-breaking is required.
- Do not use validation or extra outcomes when fitting rates, and do not let equal-time rows count one another.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)
- live_09:node_019 on [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend]: primary 0.6043, single-seed Δ -0.0000 — rejected; 84 changed lines
