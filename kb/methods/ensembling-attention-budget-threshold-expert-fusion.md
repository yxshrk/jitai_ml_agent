---
id: ensembling-attention-budget-threshold-expert-fusion
family: ensembling
target_component: ensembling
source: kb/data/facts.md §3 and §10.5 (duration-defined threshold and session attention depletion); live_08:node_010
applies_when:
  - `time_ms`, `duration_ms`, and `tab` are available for every scored impression
  - a competitive base ranker lacks cumulative duration-weighted recent-exposure demand
  - train labels may fit a smoothed empirical expert while scored-split history remains label-free
expected_delta: [0.0, 0.00013]
expected_delta_basis: measured seed-mean gain +0.00013 on the ten-model heterogeneous rank blend; the effect was
  far below acceptance scale, so the promise is capped at the exact observed mean
cost: 71 changed lines; measured runtime 215 s versus 179 s parent (~1.2x); numpy only
composes_with: [ensembling-multiseed-heterogeneous-rank-blend, ensembling-seed-average, features-exposure-session]
conflicts_with: [ensembling-cohort-gated-session-rank-fusion, ensembling-long-duration-slot-specialists]
status: dead_under [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ +0.0001)]
evidence: [live_08:node_010]
---
## Claim
Blend 10% of a smoothed empirical expert keyed by tab, current long-view threshold, and cumulative threshold demand
from the user's strictly earlier ten-minute exposures into 90% of a base model's within-user rank.

## Mechanism (why it moves within-user ranking)
The current threshold is `clip(duration_ms, 0, 18000)`. Summing this demand over recent prior impressions estimates
duration-weighted attention depletion, distinguishing otherwise similar session positions. Train-label rates for
`(tab, threshold_bin, load_bin)` convert that state into an expert score that varies across a user's candidates.

## How to implement on node_000
1. Load `time_ms` for train, valid, and `--score-extra`.
2. Add `attention_keys(rows, ui, ti, di, xi)` and stable-sort by `(user_id, time_ms, original_row)`.
3. Set demand to clipped `duration_ms` in `[0,18000]`; maintain a per-user ten-minute rolling demand sum.
4. Process equal-time rows as a group: emit the pre-group load, then add all group demands.
5. Bucket threshold at `0, 6000, 12000, 18000` and load at `1, 18000, 54000, 180000`.
6. Fit train rates for `(tab, threshold_bin, load_bin)` with 50-count shrinkage toward a `(tab, threshold_bin)`
   baseline, itself shrunk by 20 observations toward the global train rate.
7. Convert expert scores and base scores to within-user normalized ranks.
8. Emit `normalized_ranks(users, 0.9*base_rank + 0.1*expert_rank, base_rank)` for valid and extra.
9. Apply the same fusion to every validation-history checkpoint so history describes the emitted predictor.

## Risks / failure modes
- The measured gain was only +0.00013 with z=0.51 and was rejected; treat this as a weak diversity component.
- The ten-model field-aware/standard BPR parent remained unchanged, so only the empirical expert and 10% fusion
  can claim the measured delta.
- Equal-time rows must not observe one another; updating demand before emission introduces future exposure leakage.
- Fixed bins and smoothing may drift with exposure density, and the expert can reverse useful duration ordering.
- Rebuilding state separately for valid and extra is mandatory; never carry validation exposures into test scoring.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ +0.0001)
- live_08:node_010 on [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend]: primary 0.6043, single-seed Δ +0.0001, seed-mean Δ +0.0001 (z 0.51) — rejected; 71 changed lines
