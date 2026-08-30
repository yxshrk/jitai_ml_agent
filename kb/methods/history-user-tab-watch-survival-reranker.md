---
id: history-user-tab-watch-survival-reranker
family: history
target_component: history
source: kb/data/facts.md §3–4 (duration-defined label and strong tab effects); live_04:node_019
applies_when:
  - training rows provide play_time_ms as a legal target and scored rows provide duration_ms
  - the label threshold is min(duration_ms, 18000), allowing threshold-specific reach probabilities
  - a base ranker exists whose within-user ranks can be blended with a train-only reranking statistic
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0000 over 2 measurement(s), so the promise is capped at the record; was: the isolated 15% reranker changed primary by −0.00008 on the multiseed heterogeneous ensemble;
  without seed confirmation or positive movement, no attributable gain should be expected
cost: ~59 lines; measured total runtime 318 s versus 143 s for the parent ensemble; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, ensembling-multiseed-heterogeneous-rank-blend, features-duration-unknown-flag]
conflicts_with: []
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x2 (best Δ +0.0000)]
evidence: [live_04:node_019, live_04:node_026, ceiling:oracle]
---
## Claim
Estimate each user-tab group's probability of reaching a candidate-specific watch threshold from train-only play
times, shrink it toward the tab-level estimate, and blend its within-user rank 15% with the base model rank.

## Mechanism (why it moves within-user ranking)
For each second from 1 to 18, the statistic compares reaches only among training videos long enough to support that
threshold. Candidate duration selects the threshold, while user and tab select the behavioral group, so the value
can vary across one user's rows and legally change their ordering.

## How to implement on node_000
1. Read training `play_time_ms`; clip duration thresholds to `[0,18000]` and play times below at zero.
2. Encode `(user_id,tab)` as `user_code*dims[tab] + tab_code`; obtain compact group ids with `np.unique`.
3. Allocate `surv_den` and `surv_num` with shape `(n_user_tab_groups,19)` and dtype `uint32`.
4. For seconds 1–18, count rows with `duration>=sec` and those also having `play_time>=sec` using `np.bincount`.
5. Aggregate these arrays by tab; fall back to overall rates where a tab-second denominator is zero.
6. Enforce non-increasing tab survival with `np.minimum.accumulate`, then compute
   `surv_prob=(surv_num+5*tab_rate)/(surv_den+5)` and enforce monotonicity again.
7. In `survival_scores`, use user-tab probabilities for known groups, tab probabilities otherwise, and linearly
   interpolate between floor/ceiling seconds of `min(max(duration,0),18000)/1000`; assign duration zero score 0.
8. Convert base and survival scores to within-user normalized ranks and emit the tie-broken rank of
   `0.85*base_rank + 0.15*survival_rank`; apply the identical path to score-extra.

## Risks / failure modes
- This is a direct at-risk reach-rate estimate, not a product-limit Kaplan–Meier estimator.
- Sparse user-tab groups rely heavily on the fixed pseudo-count of 5 and may add noisy personalized ordering.
- A global 15% blend also reranks long videos and can dilute a strong ensemble; the measured node was slightly worse.
- The parent’s ten-model BPR ensemble was unchanged, so the observed movement is attributable to this reranker.
- Computing 18 groupwise count tables and repeated rank transforms materially increased runtime.

## Measured
_Verdict:_ never accepted in 2 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x2 (best Δ +0.0000)
- live_04:node_019 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6044, single-seed Δ -0.0001 — rejected; 59 changed lines
- live_04:node_026 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6045, single-seed Δ +0.0000, seed-mean Δ +0.0000 (t 0.02) — rejected; 36 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'user-context-taste' — facts §11.2 row 'user × tab / duration / tag / type taste': other-half rates ≤ +0.0003 (facts §11, kb/data/screens/CEILING.md)
