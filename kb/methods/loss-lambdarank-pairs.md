---
id: loss-lambdarank-pairs
family: ranking-loss
target_component: loss
source: kb/literature/losses/burges2010_ranknet-lambdarank-lambdamart.pdf (LambdaRank: pair gradients scaled by |ΔnDCG|)
applies_when:
  - nDCG@5 is half of the score and is top-heavy (scoring.md); BPR treats all pairs of a user equally
  - a pairwise loss is already in place (loss-bpr-pairwise-within-user) — this card is its top-weighted refinement
expected_delta: [0.001, 0.006]
expected_delta_basis: the metric's nDCG half rewards top positions; LambdaRank's weighting targets exactly that;
  with ~7 rows per user in valid the list is short, so the top-weighting matters less than in web search
cost: ~40 lines on top of the BPR card (per-user ranks each step via lexsort); runtime ~1.5x; numpy only
composes_with: [features-duration-unknown-flag, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head]
conflicts_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user]
status: dead_under {run: live_02, stack: official FM + loss-bpr-pairwise-within-user, delta: -0.0010}
evidence: [live_01:node_005, live_02:node_009]
---
## Claim
Keep the BPR pair loss but multiply each pair's gradient by how much nDCG would change if the two rows swapped
positions in the user's current ranking — pairs near the top of the list get the strongest push.

## Mechanism (why it moves within-user ranking)
RankNet/BPR gradients for a pair (i, j) are the same whether the pair is at ranks 1–2 or 40–41; nDCG@5 does not
care about the latter. LambdaRank's lambda_ij = (BPR gradient) x |1/log2(r_i+1) − 1/log2(r_j+1)| / IDCG focuses
learning where the metric is measured (Burges §3). Within-user ranks are cheap here because lists are short.

## How to implement on node_000
1. Start from the BPR implementation (pairs of one user's positive and negative rows).
2. Once per epoch (or per step, cheap), compute each row's rank within its user from current scores:
   order = np.lexsort((-z, user_block_id)); ranks via cumulative counts per block.
3. weight_ij = |1/log2(rank_i + 1) − 1/log2(rank_j + 1)|, normalised by the user's IDCG@5 (from label counts).
4. Multiply the pair gradient g by weight_ij before the `np.add.at` accumulation; keep a floor (e.g. 0.05) so deep
   pairs still learn a little.

## Risks / failure modes
- Ranks from an untrained model are noise in epoch 1 — warm up with plain BPR for one epoch.
- Truncating weights to the top-5 only starves users whose positives start deep; use the floor.

## Measured
- live_01:node_005 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6031, single-seed Δ -0.0005 — rejected; 112 changed lines
- live_02:node_009 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6021, single-seed Δ -0.0010 — rejected; 36 changed lines
