---
id: loss-ranksvm-margin-pairs
family: ranking-loss
target_component: loss
source: Thorsten Joachims, "Optimizing Search Engines using Clickthrough Data," KDD 2002;
  https://www.cs.cornell.edu/~tj/publications/joachims_02c.pdf ([cs.cornell.edu](https://www.cs.cornell.edu/~tj/publications/joachims_02c.pdf))
applies_when:
  - same-user positive-negative pairs are available (facts §7: most train users are discriminative)
  - BPR improves GAUC but its top-list gain is smaller (journal nodes 001/002)
  - LambdaRank, sampled hard negatives, and ListNet have already failed on the relevant FM stacks
expected_delta: [0.000, 0.0015]
expected_delta_basis: this changes only the pair surrogate, not information or capacity; BPR is already strong, so
  expect at most an acceptance-scale improvement from suppressing updates on comfortably ordered pairs
cost: ~6 changed lines on the proven BPR edit; runtime 1x; numpy only
composes_with: [model-field-aware-fm-embeddings, model-dcn-cross-head, ensembling-multiseed-heterogeneous-rank-blend]
conflicts_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0003)]
evidence: [live_06:node_008]
---
## Claim
Replace BPR's logistic pair loss with RankSVM's margin hinge,
`max(0, margin − (score_pos − score_neg))`, on the same legal within-user pairs.

## Mechanism (why it moves within-user ranking)
Same-user differencing cancels every user-constant score term. Unlike logistic BPR, hinge loss gives zero gradient
once a pair clears the margin, concentrating capacity on violated or weakly separated pairs and potentially reducing
the sharp post-peak overfitting seen in every competitive branch.

## How to implement on node_000
1. Reuse the proven BPR card's same-user positive/negative pair arrays and unchanged FM forward pass.
2. Set `d = zp - zn`, `active = d < margin`, and `g = -active.astype(np.float32) / B`.
3. Feed `(Xp, g)` and `(Xn, -g)` through the existing gradient scatter.
4. Report `np.maximum(0.0, margin - d).mean()` as training loss.
5. Test margins `{0.5, 1.0}` independently; preserve primary-based early stopping and prediction logits.

## Risks / failure modes
- Constant active-pair gradients may be less stable than BPR near the boundary; retain Adam and gradient clipping.
- A large margin can keep noisy pairs permanently active and worsen top-five ordering.
- The pair sampler must remain within user; cross-user hinge pairs optimize an irrelevant global order.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0003)
- live_06:node_008 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6033, single-seed Δ +0.0005, seed-mean Δ -0.0003 (z -0.53) — rejected; 9 changed lines
