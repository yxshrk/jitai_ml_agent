---
id: training-schedule-cyclic-bpr-pair-coverage
family: training-schedule
target_component: training-schedule
source: BPR LearnBPR sampling (Rendle et al. 2012); live_07:node_015
applies_when:
  - same-user BPR draws one negative per positive each epoch
  - each mixed-label user has a reusable negative pool
  - repeated random draws may leave some eligible negatives uncovered across epochs
expected_delta: [0.000, 0.000]
expected_delta_basis: the only measurement lost 0.00040 primary on one seed and had no fresh-seed confirmation,
  so deterministic cyclic coverage has no attributable positive expected gain
cost: 14 changed lines; unchanged update count; measured runtime 84 s on a five-member ensemble; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average]
conflicts_with: [loss-bpr-hard-negatives, loss-warp-within-user-rank-weighting]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0004)]
evidence: [live_07:node_015]
---
## Claim
Shuffle each user's negative pool once per model member, then cycle each positive through that pool across epochs,
covering eligible negatives before repeating them without increasing the number of BPR updates.

## Mechanism (why it moves within-user ranking)
Ordinary with-replacement sampling can repeatedly select the same negatives while omitting others. A member-specific
cyclic queue makes pair coverage more uniform across epochs; independent pool shuffles preserve diversity between
ensemble members while keeping every comparison within user.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; for the measured stack also apply `ensembling-seed-average`.
2. While constructing pools, append `range(len(pos))` to `neg_offset` for each mixed-label user.
3. Convert `neg_offset` to `int32`; derive `pool_starts` and `pool_counts` from unique `neg_start` values.
4. For each member RNG, copy `neg_pool` and shuffle every slice `pool[start:start + count]` once.
5. At epoch `ep`, select negatives with
   `pool[neg_start + (neg_offset + ep - 1) % neg_count]`.
6. Keep pair count, BPR updates, member-specific early stopping, and normalized-rank aggregation unchanged.

## Risks / failure modes
- The only probe regressed primary by 0.0004, reducing both GAUC and nDCG@5; broader coverage may replace useful
  stochastic regularization with repeatedly scheduled weak pairs.
- Pool shuffling consumes each member's batching RNG, so batch permutations also differ from the parent unless a
  separate seeded RNG is introduced.
- The measured node bundled proven BPR and five-seed rank averaging; this card can claim only the sampler change.
- It conflicts with adaptive or violation-driven negative samplers because they intentionally abandon fixed cycles.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0004)
- live_07:node_015 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6037, single-seed Δ -0.0004 — rejected; 14 changed lines
