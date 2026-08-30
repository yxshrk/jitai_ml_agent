---
id: ensembling-heterogeneous-rank-average
family: ensembling
target_component: ensembling
source: live_04 node_012; task.md within-user monotone invariance; standard rank-aggregation practice
applies_when:
  - two independently selected models have similar primary scores but structurally different representations
  - raw score scales are not comparable, while evaluation depends only on within-user ordering
  - runtime permits training and retaining both parent models
expected_delta: [0.0, 0.0006]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0006 over 1 measurement(s), so the promise is capped at the record; was: the archived field-aware-FM plus standard-FM BPR rank blend gained +0.00056 seed-mean,
  narrowly missing acceptance at t=2.43; claim no larger gain until a changed stack or stronger averaging confirms it
cost: ~77 lines; runtime ~1.7x versus the field-aware parent (49 s measured); numpy only
composes_with: [ensembling-seed-average, model-field-aware-fm-embeddings, model-dcn-cross-head, regularization-embedding-dropout-l2]
conflicts_with: []
status: dead_under [official FM + field-aware FM embeddings x1 (best Δ +0.0006)]
evidence: [live_04:node_012, kb-screens:screen-g00]
---
## Claim
Independently early-stop a field-aware FM and a standard FM trained with the same within-user BPR sampler, then
average their normalized within-user ranks with a tiny field-aware-rank tie-break.

## Mechanism (why it moves within-user ranking)
Field-aware and shared FM embeddings impose different interaction constraints and therefore make different ordering
errors. Converting each model's scores to within-user normalized ranks removes irrelevant scale differences; averaging
can preserve agreements while correcting errors unique to either representation.

## How to implement on node_000
1. Retain the field-aware BPR model and add `StandardFM`, with `V.shape = (dim, k)`.
2. In `StandardFM.logits`, compute `E=V[X]`, `S=E.sum(1)`, and the standard FM pairwise interaction.
3. In its BPR step, scatter `h * (S-E[:,i])` into each field's shared embedding using `np.add.at`.
4. Train the two models separately with the same legal pair sampler and separate RNG/model state.
5. Maintain separate validation-primary best scores, patience counters, and checkpoints; never select on the blend.
6. Restore both checkpoints before blending.
7. Implement `normalized_ranks` using `lexsort((row_index, score, user))` and divide positions by `count-1`.
8. Emit `0.5*rank_field_aware + 0.5*rank_standard + 1e-6*rank_field_aware`.
9. Apply the identical rank conversion and blend to `predictions_extra.csv`.

## Risks / failure modes
- The diff includes a full standard-FM BPR sibling, whose BPR mechanism is already covered by
  `loss-bpr-pairwise-within-user`; only the incremental rank-ensemble gain belongs to this card.
- Joint early stopping on blended validation performance leaks model selection across parents and exaggerates gains.
- Similar models may have highly correlated errors; the measured +0.00056 missed acceptance by a narrow t margin.
- Rank averaging discards meaningful confidence margins and can create ties; retain a deterministic tiny tie-break.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings x1 (best Δ +0.0006); screened 1x (kb-screens:screen-g00 kept +0.0006)
- live_04:node_012 on [official FM + field-aware FM embeddings]: primary 0.6036, single-seed Δ +0.0006, seed-mean Δ +0.0006 (t 2.43) — rejected; 77 changed lines
- kb-screens:screen-g00 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: SCREENED kept best_gain +0.0006 (gbdt_member z-blend w=1.0); stack +0.0006; gbdt_member: varies 1.0, GAUC 0.5996, additive +0.0006 (w 0.25/0.5/0.75/1.0 → +0.0000/+0.0002/+0.0005/+0.0006; within-user rank blend +0.0004; against single-seed BPR node_003 the same member gave +0.0021) — new_signal: lambdarank LightGBM on OOF target statistics + session features + tab/dur, blended in z-space with the seed-averaged champion (kb/data/screens/blend009.py, review session, 2026-08-31)
