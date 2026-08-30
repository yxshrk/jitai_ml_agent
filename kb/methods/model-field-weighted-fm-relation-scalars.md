---
id: model-field-weighted-fm-relation-scalars
family: model
target_component: model
source: Pan et al., "Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising," WWW 2018
applies_when:
  - the ranker is an FM over distinct categorical fields
  - field-pair relations may differ substantially, as with tab and duration context
  - full field-aware embeddings would add too many parameters
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0002 over 1 measurement(s), so the promise is capped at the record; was: the isolated probe on standard FM+BPR lost 0.00018 primary on seed 0 and received no seed
  confirmation; therefore no positive attributable gain is currently supported
cost: 23 changed lines; ten scalar parameters for five fields; measured runtime 24 s (~1.3x parent); numpy only
composes_with: [loss-bpr-pairwise-within-user, features-fine-duration-and-tab-cross, regularization-embedding-dropout-l2]
conflicts_with: [model-field-aware-fm-embeddings]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0002)]
evidence: [live_05:node_010]
---
## Claim
Multiply every FM field-pair dot product by its own learned scalar, allowing interaction families such as
user-video, user-tab, and tab-duration to have different strengths without partner-specific embedding tables.

## Mechanism (why it moves within-user ranking)
A standard FM gives all field-pair dot products the same fixed coefficient. Learned relation scalars can amplify or
suppress particular interaction families; because video, author, tab, and duration vary within users, these changes
can alter the evaluated ordering.

## How to implement on node_000
1. Apply `loss-bpr-pairwise-within-user` so the implementation matches the measured parent.
2. Define `FIELD_PAIRS = [(i,j) for i in range(F) for j in range(i+1,F)]`.
3. Initialize `R = ones(len(FIELD_PAIRS))` and matching Adam states `mR`, `vR`.
4. Replace the FM interaction shortcut with
   `sum_p R[p] * sum_k(E[:,i,k] * E[:,j,k])`.
5. In each positive and negative gradient pass, accumulate
   `gR[p] += dot(h, sum_k(E_i*E_j))`.
6. Scatter `h*R[p]*E_j` to field `i` embeddings and `h*R[p]*E_i` to field `j`.
7. Update `R` with the same Adam step as `V` and `W`.
8. Include `R.copy()` in the best checkpoint and restore it before prediction.

## Risks / failure modes
- The measured node lost 0.00018 primary and peaked earlier than its BPR parent, so the scalars may destabilize or
  accelerate overfitting rather than provide useful capacity.
- `R` and embedding norms are partially non-identifiable; regularization or constrained scalars may be needed.
- The implementation also contains the separately proven BPR loss; none of BPR's gain is attributable to this card.
- This is lower-capacity than field-aware FM but overlaps its purpose, so combining both is redundant.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0002)
- live_05:node_010 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6035, single-seed Δ -0.0002 — rejected; 23 changed lines
