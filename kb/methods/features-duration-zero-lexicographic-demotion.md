---
id: features-duration-zero-lexicographic-demotion
family: features
target_component: features
source: kb/data/facts.md §3 (duration_ms = 0 rows are always negative); live_05:node_008
applies_when:
  - scored rows expose legal show-time feature duration_ms
  - duration_ms = 0 is known to imply long_view = 0
  - predictions are evaluated only by within-user ordering
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated FM+BPR probe lost 0.00045 primary on one seed and had no seed confirmation;
  therefore no positive attributable gain is supported
cost: ~20 changed lines; deterministic prediction-time transform; negligible runtime; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, ensembling-multiseed-heterogeneous-rank-blend]
conflicts_with: [features-duration-unknown-flag]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0004)]
evidence: [live_05:node_008]
---
## Claim
Force every zero-duration impression below every known-duration impression for the same user while preserving the
model's ordering within each partition.

## Mechanism (why it moves within-user ranking)
The transform uses a row-varying legal feature and changes only cross-partition comparisons. Subtracting more than
the user's score range from zero-duration rows makes every known-duration row rank above them without changing
ordering among known-duration rows or among zero-duration rows.

## How to implement on node_000
1. Add `demote_duration_zero(scores, user_groups, duration_zero)`.
2. Copy scores as float64 and compute each user's minimum and maximum with `np.minimum.at` and `np.maximum.at`.
3. For masked rows subtract `(user_max - user_min + 1.0)`, indexed by inverse user group.
4. Build validation groups with `np.unique(user_ids, return_inverse=True)`.
5. Build the mask as `float(duration_ms) == 0.0`.
6. Apply the transform during every validation evaluation so early stopping selects the emitted policy.
7. Apply it again to final validation predictions.
8. Build groups and masks independently for score-extra rows and apply the identical transform.

## Risks / failure modes
- The archived isolated probe lost 0.00045 primary, so forcing the rule may be worse than letting the model rank
  uncertain rows softly.
- Applying the transform during validation can change checkpoint selection, not merely final post-processing.
- The measured parent already contained same-user BPR; that gain belongs to
  `loss-bpr-pairwise-within-user`, not to this card.
- This overlaps the learned `features-duration-unknown-flag`; combining both does not provide an isolated test.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0004)
- live_05:node_008 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6032, single-seed Δ -0.0004 — rejected; 20 changed lines
