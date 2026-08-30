---
id: regularization-id-field-targeted-l2
family: regularization
target_component: regularization
source: kb/literature/models/1706.06978_din.pdf (mini-batch-aware regularisation for sparse ids);
  live_06:node_017
applies_when:
  - an FM uses contiguous field vocabularies with user, video, and author as the first three fields
  - sparse identity embeddings are suspected of overfitting while tab and duration priors should remain weakly penalized
  - a mild refinement of uniform L2 is desired without changing the loss, sampler, or ensemble
expected_delta: [0.000, 0.00015]
expected_delta_basis: the originating five-seed FM-BPR ensemble probe measured fresh-seed mean Δ +0.00015
  (single-seed +0.00027), below acceptance; no larger attributable gain is supported
cost: 10 changed lines; one length-dim penalty vector; runtime approximately 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-dcn-cross-head]
conflicts_with: [regularization-embedding-dropout-l2, regularization-embedding-only-l2]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0001)]
evidence: [live_06:node_017]
---
## Claim
Apply L2=3e-6 to user, video, and author latent embeddings while retaining L2=1e-6 for tab and duration
embeddings and every linear FM weight.

## Mechanism (why it moves within-user ranking)
Identity embeddings have many sparse parameters and can memorize noisy user-item observations. Mildly shrinking
only those vectors reduces their variance while preserving stronger row-varying tab and duration effects that
contribute directly to within-user ordering.

## How to implement on node_000
1. Extend `FM.__init__` with `embedding_l2=None`.
2. Store either `np.full(dim, l2)` or the supplied float32 vector as `self.embedding_l2`.
3. After field dimensions and offsets are built, define `id_feature_mask = np.arange(dim) < off[3]`.
4. Set `embedding_l2 = np.where(id_feature_mask, 3e-6, 1e-6).astype(np.float32)`.
5. Replace `gV += self.l2 * self.V` with `gV += self.embedding_l2[:, None] * self.V`.
6. Keep `gW += self.l2 * self.W`, so every linear weight remains at 1e-6.
7. Pass `embedding_l2=embedding_l2` to every independently trained FM member.

## Risks / failure modes
- The measured gain was only +0.00015 over fresh seeds with z=0.35 and was not accepted.
- The originating parent already contained same-user BPR and five-seed rank averaging; this card claims only the
  incremental field-targeted L2 effect, not gains from either parent mechanism.
- Stronger identity shrinkage can erase the user-video memorization responsible for personalization.
- The `off[3]` boundary is valid only when user, video, and author are the first three contiguous fields.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0001)
- live_06:node_017 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6042, single-seed Δ +0.0003, seed-mean Δ +0.0001 (z 0.35) — rejected; 10 changed lines
