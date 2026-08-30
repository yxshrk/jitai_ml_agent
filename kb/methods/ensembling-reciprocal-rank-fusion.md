---
id: ensembling-reciprocal-rank-fusion
family: ensembling
target_component: ensembling
source: reciprocal-rank fusion (Cormack, Clarke, and Buettcher, SIGIR 2009); live_04:node_023
applies_when:
  - two model branches already produce competitive, complementary within-user rankings
  - the current ensemble linearly averages normalized ranks and nDCG@5 lags GAUC
  - branch score scales are irrelevant or incomparable, so fusion should operate on within-user positions
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated replacement measured single-seed Δ -0.00001 on the five-seed field-aware/standard
  BPR ensemble; nDCG@5 rose only 0.00007 while GAUC fell 0.00004, providing no attributable positive primary gain
cost: ~16 lines; fusion overhead only, but total runtime inherits the underlying ensemble (~316 s measured); numpy only
composes_with: [ensembling-multiseed-heterogeneous-rank-blend, model-field-aware-fm-embeddings, model-dcn-cross-head]
conflicts_with: [ensembling-heterogeneous-rank-average]
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0000)]
evidence: [live_04:node_023]
---
## Claim
Replace linear Borda-style averaging of branch ranks with fixed reciprocal-rank fusion, using `1/(2+r)` for each
within-user position before applying the existing 0.6/0.4 branch weights.

## Mechanism (why it moves within-user ranking)
Reciprocal ranks make disagreements near the top contribute more than equally sized disagreements lower in a
user's list. This can alter top-five ordering while remaining invariant to each branch's raw score scale.

## How to implement on node_000
1. Start from a heterogeneous ensemble that exposes one score array per branch.
2. Add `reciprocal_ranks(users, scores)`.
3. Sort with `np.lexsort((row_index, -scores, users))`.
4. Find user-block starts and counts in the sorted order.
5. Assign zero-based within-user positions back to original row order.
6. Return `1.0 / (2.0 + positions)`.
7. Convert both branch rank-average arrays with this function.
8. Replace the linear blend by `0.6*rr_field_aware + 0.4*rr_standard`.
9. Apply the existing final `normalized_ranks(..., tiebreak=field_aware_scores)`.
10. Use the identical transformation for validation history and `predictions_extra.csv`.

## Risks / failure modes
- The measured edit changed only fusion; all gains from five-seed averaging, BPR, field-aware embeddings, and
  heterogeneous branches belong to the parent ensemble, not this card.
- Reciprocal compression can sacrifice broad pair ordering and GAUC while moving only a few top-five positions.
- With lists averaging about 5.6 rows, linear and reciprocal fusion often induce nearly identical orders.
- Deterministic final tie-breaking remains necessary because reciprocal ranks are discrete.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0000)
- live_04:node_023 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6045, single-seed Δ -0.0000 — rejected; 16 changed lines
