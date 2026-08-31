---
id: ensembling-relation-support-adaptive-architecture-fusion
family: ensembling
target_component: ensembling
source: live_08:node_014; Juan et al., RecSys 2016, Field-aware Factorization Machines for CTR Prediction
applies_when:
  - field-aware and standard FM branches are already combined by within-user rank fusion
  - train-only user-video and user-author exposure counts are available for every scored candidate
  - relation support varies enough that field-aware parameters may be less reliable on sparse relations
expected_delta: [0.0, 0.0]
expected_delta_basis: measured seed-flat on the heterogeneous multiseed stack (fresh-seed mean Δ -0.00001);
  the adaptive gate cannot claim the gain of the underlying fixed 0.6/0.4 blend
cost: 33 changed lines beyond the heterogeneous blend; two train count tables; negligible inference overhead
composes_with: [ensembling-multiseed-heterogeneous-rank-blend, model-field-aware-fm-embeddings,
  loss-bpr-pairwise-within-user, ensembling-seed-average]
conflicts_with: [ensembling-user-tab-support-gated-objective-blend]
status: dead_under [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)]
evidence: [live_08:node_014]
---
## Claim
Replace a fixed field-aware/standard rank-blend weight with a candidate-specific weight derived from train-only
user-video and user-author exposure support.

## Mechanism (why it moves within-user ranking)
Supported relations receive more weight from the less-shared field-aware branch, while sparse or unseen relations
move toward the standard FM branch. Because relation counts differ across a user's candidates, the gate can alter
within-user order rather than merely calibrating the user.

## How to implement on node_000
1. Apply the existing BPR, field-aware FM, seed-average, and heterogeneous-rank-blend cards.
2. From encoded train rows, count `(user, video)` and `(user, author)` occurrences with packed keys
   `user * dim + feature_id`, storing sorted unique keys and counts.
3. For each scored row, retrieve both counts with `np.searchsorted`, assigning zero when a key is absent.
4. Compute `combined = video_count + 0.5 * author_count`.
5. Set `confidence = combined / (combined + 3.0)` and `field_weight = 0.5 + 0.2 * confidence`.
6. Blend branch-average within-user ranks as
   `field_weight * field_rank + (1 - field_weight) * standard_rank`.
7. Normalize the result within user with the field rank as tie-breaker.
8. Use the identical train-derived tables and gate for validation history, final validation, and score-extra.

## Risks / failure modes
- The measured gate was seed-flat; relation support may not identify which architecture is locally more accurate.
- Exact user-author history is extremely sparse, and closed-catalogue item support does not imply user-item support.
- This card can claim only the adaptive weighting effect, not BPR, seed averaging, field-aware modeling, or the
  underlying heterogeneous blend.
- Combining this gate with another candidate-specific objective gate may create unstable, uninterpretable weights.
- Packed relation keys require an integer multiplier larger than every encoded feature ID to avoid collisions.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)
- live_08:node_014 on [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend]: primary 0.6042, single-seed Δ +0.0001, seed-mean Δ -0.0000 (z -0.06) — rejected; 33 changed lines
