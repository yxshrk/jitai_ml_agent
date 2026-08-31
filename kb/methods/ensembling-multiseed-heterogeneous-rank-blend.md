---
id: ensembling-multiseed-heterogeneous-rank-blend
family: ensembling
target_component: ensembling
source: live_04:node_015; kb/literature/agent-design-notes.md (seed averaging and heterogeneous ensembling)
applies_when:
  - two competitive model branches make complementary within-user ordering errors
  - single-seed variance remains material and runtime permits ten independently early-stopped models
  - branch score scales differ, making within-user rank normalization safer than raw-logit averaging
expected_delta: [0.0, 0.0017]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0017 over 8 measurement(s), so the promise is capped at the record; was: the complete five-seed-per-branch 0.6/0.4 recipe measured +0.00166 seed-mean; the gain cannot
  be assigned separately to seed averaging, heterogeneous blending, or the changed blend weight
cost: ~120 lines; ten training phases; measured runtime 143 s (~5x the field-aware parent); numpy only
composes_with: [model-dcn-cross-head, model-field-aware-fm-embeddings, loss-bpr-pairwise-within-user]
conflicts_with: [ensembling-heterogeneous-rank-average, ensembling-seed-average]
status: proven — accepted on [official FM + ensembling-seed-average], [official FM + field-aware FM embeddings], [official FM + loss-bpr-pairwise-within-user]
evidence: [live_04:node_015, live_05:node_012, live_05:node_014, live_06:node_015, live_08:node_006, live_08:node_013, live_09:node_014, live_09:node_018, live_09:node_021]
---
## Claim
Train five field-aware FM-BPR models and five standard FM-BPR models, average normalized within-user ranks inside
each branch, then emit a tie-free 0.6/0.4 field-aware-to-standard rank blend.

## Mechanism (why it moves within-user ranking)
Seed averaging cancels initialization and pair-sampling variance within each architecture, while the shared and
partner-specific embedding parameterizations retain different ordering errors. Within-user ranks remove irrelevant
score-scale differences before the stronger field-aware branch receives 60% of the blend weight.

## How to implement on node_000
1. Add `normalized_ranks(users, scores, tiebreak=None)` using `np.lexsort`, per-user starts and `[0,1]` ranks.
2. Extend `FM(..., standard=False)` so standard models use `V.shape=(dim,k)` and ordinary FM interactions.
3. Train branches `standard=False/True`, with five members using seeds `seed + branch*5 + member`.
4. Early-stop and restore every member independently on its own validation primary.
5. Cache each member's best-so-far validation prediction and loss after every epoch.
6. Synchronize histories by padding stopped members with their final cached state.
7. Per epoch, average normalized ranks within each branch and blend `0.6*field_aware + 0.4*standard`.
8. Apply `normalized_ranks(blend, tiebreak=field_aware)` and use the resulting ensemble for history and outputs.
9. Apply the identical branch averaging, blend, and tie-break path to `predictions_extra.csv`.

## Risks / failure modes
- This is a joint composition of `ensembling-heterogeneous-rank-average` and `ensembling-seed-average`; the measured
  improvement does not identify which component contributed the gain.
- Both branches used the parent's existing within-user BPR sampler, so this card does not establish a gain for
  pointwise models or arbitrary model pairs.
- Ten validation-selected members increase compute and repeated validation exposure; preserve independent stopping.
- Final history must describe the emitted ensemble, not one member, and its final metrics must match predictions.csv.

## Measured
_Verdict:_ ACCEPTED 3x (live_04:node_015 on [official FM + field-aware FM embeddings] Δ +0.0017; live_08:node_006 on [official FM + ensembling-seed-average] Δ +0.0016; live_09:node_014 on [official FM + loss-bpr-pairwise-within-user] Δ +0.0014); implementation failed in live_05:node_014
- live_04:node_015 on [official FM + field-aware FM embeddings]: primary 0.6045, single-seed Δ +0.0014, seed-mean Δ +0.0017 (t 7.65) — ACCEPTED; 122 changed lines
- live_05:node_012 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6040, single-seed Δ +0.0003, seed-mean Δ +0.0005 (z 1.76) — rejected; 100 changed lines
- live_05:node_014 on [official FM + loss-bpr-pairwise-within-user]: FAILED at implement — no runnable script produced (recovery: None)
- live_06:node_015 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6034, single-seed Δ -0.0006 — rejected; 43 changed lines
- live_08:node_006 on [official FM + ensembling-seed-average]: primary 0.6041, single-seed Δ +0.0012, seed-mean Δ +0.0016 (z 5.58) — ACCEPTED; 118 changed lines
- live_08:node_013 on [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend] (variant: same-tab sampler-diverse member): primary 0.6041, single-seed Δ -0.0000 — rejected; 21 changed lines
- live_09:node_014 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6043, single-seed Δ +0.0007, seed-mean Δ +0.0014 (z 4.58) — ACCEPTED; 109 changed lines
- live_09:node_018 on [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend] (variant: latent-dimension-diverse-members): primary 0.6045, single-seed Δ +0.0002, seed-mean Δ -0.0005 (z -1.62) — rejected; 4 changed lines
- live_09:node_021 on [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend] (variant: row-conditioned-attention member diversity): primary 0.6039, single-seed Δ -0.0004 — rejected; 85 changed lines
