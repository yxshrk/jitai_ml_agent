---
id: ensembling-top5-gated-hybrid-rank-fusion
family: ensembling
target_component: ensembling
source: live_04:node_025; live_04:node_023; standard reciprocal-rank fusion
applies_when:
  - two ensemble branches already produce normalized within-user ranks
  - branch disagreements near the top five are suspected to matter more than disagreements deeper in the list
  - linear rank blending is the current fusion rule and deterministic row-level tie-breaking is available
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated probe lost 0.00016 primary on the multiseed heterogeneous ensemble; no positive
  seed-mean measurement exists, so this mechanism currently supports no attributable gain
cost: ~30 changed lines; fusion-only runtime negligible, though parent ensemble training measured 256 s; numpy only
composes_with: [ensembling-multiseed-heterogeneous-rank-blend, model-field-aware-fm-embeddings]
conflicts_with: [ensembling-reciprocal-rank-fusion, ensembling-heterogeneous-rank-average]
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0002)]
evidence: [live_04:node_025]
---
## Claim
Use reciprocal-rank fusion only for rows where two branches disagree and at least one places the row in its top
five; retain the existing 0.6/0.4 linear rank blend for all other rows.

## Mechanism (why it moves within-user ranking)
The gate confines the nonlinear, top-heavy reciprocal transform to branch disagreements that can affect nDCG@5.
Rows outside that set retain the linear blend intended to preserve broad pair ordering and GAUC.

## How to implement on node_000
1. Add `rank_positions(users, scores, tiebreak=None)` using `np.lexsort` on user and descending score.
2. Assign zero-based positions within each user from sorted block starts and counts.
3. In `top5_gated_fusion`, compute `linear = 0.6*branch0 + 0.4*branch1`.
4. Convert linear scores to normalized within-user ranks with branch 0 as the tie-break.
5. Compute `reciprocal = 0.6/(2+pos0) + 0.4/(2+pos1)` and normalize it similarly.
6. Set `affected = (minimum(pos0,pos1) < 5) & (pos0 != pos1)`.
7. Select reciprocal ranks for affected rows and linear ranks elsewhere, then normalize once more.
8. Apply the identical function to validation history, final predictions, and score-extra predictions.

## Risks / failure modes
- The measured child lost 0.00016 primary, including lower GAUC and nDCG, so gating did not retain node_023's tiny
  nDCG movement on this stack.
- Replacing scores row by row after separate normalizations can create cross-boundary order changes beyond the
  intended top-five disagreements.
- This is a fusion alternative, not a new model signal; it cannot claim gains from the parent’s five-seed
  heterogeneous ensemble.
- Lists with five or fewer rows make the gate broad, reducing its distinction from unrestricted reciprocal fusion.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0002)
- live_04:node_025 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6043, single-seed Δ -0.0002 — rejected; 30 changed lines
