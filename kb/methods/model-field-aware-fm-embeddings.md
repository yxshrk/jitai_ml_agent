---
id: model-field-aware-fm-embeddings
family: model
target_component: model
source: Juan et al., RecSys 2016, Field-aware Factorization Machines for CTR Prediction
applies_when:
  - the model is an FM over distinct categorical fields whose interactions may require different representations
  - the catalogue is closed, so field-specific user, video, and author vectors are estimable
  - a within-user BPR loss is already present, allowing the architecture effect to be tested in isolation
expected_delta: [0.0, 0.0012]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0012 over 1 measurement(s), so the promise is capped at the record; was: the bundled field-aware+BPR node gained +0.00122 seed-mean over pointwise FM, while the same-parent
  BPR sibling gained +0.00114; only about +0.00008 can be attributed to field-aware embeddings
cost: ~20 model lines; embedding memory ~5x and measured runtime ~2.3x versus standard-FM BPR; numpy only
composes_with: [loss-bpr-pairwise-within-user, data-weighting-recency, features-duration-unknown-flag, regularization-embedding-dropout-l2, training-schedule-weight-averaging]
conflicts_with: []
status: proven — accepted on [official FM]
evidence: [live_04:node_001]
---
## Claim
Give each feature value a separate latent vector for every partner field, so its user-video representation need
not also serve user-tab, video-duration, or other interaction pairs.

## Mechanism (why it moves within-user ranking)
A standard FM shares one vector across all pair types. Field-aware vectors remove that coupling and can represent
distinct interaction geometries for the row-varying video, author, tab, and duration fields.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; the archived wildcard included that loss change.
2. Change `V` and its Adam states from `(dim, k)` to `(dim, len(FIELDS), k)`.
3. In `logits`, loop over `i < j` and add
   `(V[X[:, i], j] * V[X[:, j], i]).sum(1)`.
4. In each positive/negative BPR gradient pass, set `Ei = V[X[:, i], j]` and `Ej = V[X[:, j], i]`.
5. Scatter `h[:,None] * Ej` into `(X[:,i], j)` and `h[:,None] * Ei` into `(X[:,j], i)`.
6. Preserve the BPR sampler, optimizer, early stopping, encoding, and prediction logic.

## Risks / failure modes
- The archived diff also replaced pointwise logloss with BPR; its full +0.00122 seed-mean gain is not attributable
  to this model change. Same-parent BPR alone gained +0.00114.
- Parameter count grows by the number of fields and the curve overfit sharply after epoch 5.
- With only five fields, partner-specific vectors may mostly duplicate capacity already available to the FM.

## Measured
_Verdict:_ ACCEPTED 1x (live_04:node_001 on [official FM] Δ +0.0012)
- live_04:node_001 on [official FM]: primary 0.6030, single-seed Δ +0.0016, seed-mean Δ +0.0012 (t 5.22) — ACCEPTED; 46 changed lines
