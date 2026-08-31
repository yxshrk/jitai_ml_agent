---
id: ensembling-user-tab-support-gated-objective-blend
family: ensembling
target_component: ensembling
source: live_08:node_005; loss-bpr-pairwise-within-user; kb/data/facts.md §2
applies_when:
  - competitive pointwise and same-user BPR ensembles are both available
  - user-by-tab training histories vary in positive and negative class support
  - BPR may be unreliable for sparse or single-class user-tab histories
expected_delta: [0.0, 0.0008]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0008 over 1 measurement(s), so the promise is capped at the record; was: attribution-adjusted same-parent contrast: the complete gated node gained +0.00085 seed-mean,
  but an ungated five-seed BPR sibling gained +0.00115, so no positive gain can be assigned to the support gate itself
cost: 85 changed lines; ten independently stopped FM phases; measured runtime 138 s; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, history-user-aggregates]
conflicts_with: [ensembling-multiseed-heterogeneous-rank-blend, ensembling-heterogeneous-rank-average]
status: proven — accepted on [official FM + ensembling-seed-average]
evidence: [live_08:node_005]
---
## Claim
Blend pointwise-FM and BPR-FM seed ensembles using a row-specific weight derived from the minimum positive and
negative support in the candidate's user-by-tab training history.

## Mechanism (why it moves within-user ranking)
Pointwise training retains information from single-class histories, while BPR directly optimizes ordering but needs
both classes. The gate `min(n_pos,n_neg)/(min(n_pos,n_neg)+2)` shifts candidates toward BPR only where their
user-tab history contains both outcomes. Because tab varies across a user's scored rows, the gate can change order.

## How to implement on node_000
1. Add `step_pair(Xp, Xn)` implementing logistic BPR gradients for the existing FM.
2. Build dense `support_pos[user,tab]` and `support_neg[user,tab]` arrays from training labels.
3. Define `confidence_gate(X)` as `s/(s+2)`, where `s=min(support_pos,support_neg)` for each row.
4. Build per-user negative pools and retain positive rows belonging to users with at least one negative.
5. Train five pointwise members and five BPR members with matched seeds and independent primary-based stopping.
6. Within each branch, combine members using the existing tie-free within-user `rank_average`.
7. Add `gated_rank_average`: `(1-gate)*point_rank + gate*bpr_rank`, then normalize ranks again.
8. Use the identical gate and fusion for every history checkpoint, validation output, and score-extra output.

## Risks / failure modes
- The diff bundles an existing BPR branch and five-seed averaging with the new gate; its accepted gain is not
  attributable to gating.
- On the same parent, the ungated BPR ensemble gained +0.00115 seed-mean versus +0.00085 for this gated node,
  suggesting the gate reduced rather than added value.
- Dense user-by-tab support arrays assume small tab cardinality; sparse maps are safer for larger context fields.
- Training ten models doubles the five-seed parent's runtime and validation-selection exposure.

## Measured
_Verdict:_ ACCEPTED 1x (live_08:node_005 on [official FM + ensembling-seed-average] Δ +0.0008)
- live_08:node_005 on [official FM + ensembling-seed-average]: primary 0.6036, single-seed Δ +0.0007, seed-mean Δ +0.0008 (z 3.25) — ACCEPTED; 85 changed lines
