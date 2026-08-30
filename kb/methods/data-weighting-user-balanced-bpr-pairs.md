---
id: data-weighting-user-balanced-bpr-pairs
family: data-weighting
target_component: data-weighting
source: task.md metric definitions (positive-weighted GAUC and user-unweighted nDCG@5); live_04:node_022
applies_when:
  - within-user BPR currently draws one pair per positive, thereby weighting users approximately by positive count
  - training has mixed-label users from which both positive and negative examples can be sampled
  - the objective should balance GAUC's positive weighting against nDCG@5's equal per-user weighting
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0019 over 1 measurement(s), so the promise is capped at the record; was: the isolated wildcard lost 0.00192 primary on the ten-model field-aware/standard-FM rank
  ensemble; there is no attributable positive evidence for this fixed 50/50 sampling recipe
cost: 9 lines; unchanged pair count and approximately 1x training cost; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-bpr-hard-negatives, model-field-aware-fm-embeddings, model-dcn-cross-head]
conflicts_with: [loss-listwise-softmax-within-user]
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0019)]
evidence: [live_04:node_022, ceiling:oracle]
---
## Claim
Construct each BPR epoch from an equal mixture of the usual positive-proportional sample and positives drawn by
uniformly sampling mixed-label users, while retaining uniform same-user negative sampling.

## Mechanism (why it moves within-user ranking)
One pair per positive emphasizes users with many positives, resembling GAUC's weighting. Uniformly sampling users
gives sparse mixed-label users more influence, moving part of the training objective toward nDCG@5's equal
per-user weighting while every pair still changes only within-user order.

## How to implement on node_000
1. First implement `loss-bpr-pairwise-within-user`; retain its negative arrays and BPR update unchanged.
2. Stable-sort eligible positive indices by encoded user into `pos_order`.
3. Build `mixed_users, pos_starts, pos_counts = np.unique(..., return_index=True, return_counts=True)`.
4. Each epoch, take a random half of `pair_pos`; fill the other half by uniformly drawing indices into
   `mixed_users`, then uniformly selecting one positive from each selected user's block.
5. Concatenate and reshuffle both halves, then sample negatives with the existing same-user logic.

## Risks / failure modes
- The archived isolated edit reduced primary by 0.00192, with GAUC and nDCG@5 both lower; the 50/50 mixture
  appears to over-correct away from the useful positive-proportional distribution on that ensemble stack.
- It does not increase information or model capacity; it only changes which users dominate optimization.
- Uniform-user sampling repeatedly presents positives from sparse users and can amplify label noise.
- The parent already contained BPR and multiseed heterogeneous rank blending; only the pair-distribution change
  is attributable to this card.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0019)
- live_04:node_022 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6026, single-seed Δ -0.0019 — rejected; 9 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0000 for the signal family 'pair-sampling' — facts §11.3: same-tab negatives at 30 / 70 / 100 % 0.6030 / 0.6024 / 0.5880 vs 0.6031; matched / hard / cohort pair cards measured ≤ +0.0001 (facts §11, kb/data/screens/CEILING.md)
