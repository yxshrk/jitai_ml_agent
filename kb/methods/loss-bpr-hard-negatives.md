---
id: loss-bpr-hard-negatives
family: ranking-loss
target_component: loss
source: kb/literature/losses/1205.2618_bpr.pdf (LearnBPR sampling); standard practice for pairwise ranking (importance / adaptive negative sampling)
applies_when:
  - a within-user pairwise loss is already the champion (live_01: node_001 BPR accepted)
  - users have several negatives to choose from (facts: train mean 43.5 rows/user, positive rate 0.34)
expected_delta: [0.0, 0.0001]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0001 over 1 measurement(s), so the promise is capped at the record; was: uniform negatives waste gradient on pairs the model already orders; sampling negatives the
  model currently ranks high (or several negatives per positive) is a classic pairwise-ranking booster — small here
cost: ~25 lines on top of BPR (score-aware negative choice or n negatives per positive); runtime 1–2x; numpy only
composes_with: [features-duration-unknown-flag, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head, regularization-embedding-dropout-l2]
conflicts_with: [loss-listwise-softmax-within-user, loss-lambdarank-pairs]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0001)]
evidence: [live_02:node_005]
---
## Claim
Replace uniform same-user negative sampling with (a) several negatives per positive, or (b) choosing among m
sampled negatives the one with the highest current score — the pairs the model gets wrong.

## Mechanism (why it moves within-user ranking)
The BPR gradient −σ(−d) vanishes for pairs already ordered with margin; uniform sampling spends most pairs there.
Hard negatives keep the gradient on the mis-ordered pairs that GAUC counts, so the same epochs do more work.

## How to implement on node_000 (with BPR in place)
1. Multi-negative: for each positive row draw n = 3 negatives from the same user; average the pair losses.
2. Hard: draw m = 4 candidates, compute their current logits (one extra forward pass), keep the max-score one.
3. Keep everything else (early stopping, outputs, seed handling) identical.

## Risks / failure modes
- Hard negatives amplify label noise (a mislabelled "negative" that is really a near-positive) — start with n = 3
  uniform negatives, then try the hard variant.
- Extra forward passes cost time; keep m small.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0001)
- live_02:node_005 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6037, single-seed Δ +0.0006, seed-mean Δ +0.0001 (t 0.63) — rejected; 7 changed lines
