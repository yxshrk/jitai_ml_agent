---
id: regularization-embedding-only-l2
family: regularization
target_component: regularization
source: live_06:node_009; kb/literature/models/1706.06978_din.pdf (sparse-ID regularisation)
applies_when:
  - an FM maintains separate latent embeddings and linear feature weights
  - sparse-ID embeddings are suspected to overfit while tab and duration linear priors remain useful
  - validation primary peaks or plateaus before training ends
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated FM+BPR probe lost 0.00022 primary on seed 0 and had no fresh-seed confirmation,
  so this exact 1e-5 embedding / 1e-6 linear-weight split has no attributable positive evidence
cost: 6 changed lines; separate scalar penalties for V and W; runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, model-dcn-cross-head]
conflicts_with: [regularization-embedding-dropout-l2]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0002)]
evidence: [live_06:node_009]
---
## Claim
Regularize latent FM embeddings at 1e-5 while retaining a 1e-6 penalty on linear feature weights, decoupling
sparse interaction shrinkage from the additive tab, duration, and item priors.

## Mechanism (why it moves within-user ranking)
Latent vectors encode the interaction terms that personalize row ordering and may overfit sparse IDs. Separate
penalties allow stronger shrinkage of those vectors without equally shrinking linear row-varying context effects.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; preserve its sampler, optimizer, and stopping logic.
2. Change `FM.__init__` from `l2=1e-6` to `l2_v=1e-5, l2_w=1e-6`.
3. Store the values as `self.l2_v` and `self.l2_w`.
4. Replace `gV += self.l2 * self.V` with `gV += self.l2_v * self.V`.
5. Replace `gW += self.l2 * self.W` with `gW += self.l2_w * self.W`.
6. Leave Adam updates, validation-primary early stopping, checkpointing, and prediction unchanged.

## Risks / failure modes
- Stronger embedding shrinkage can remove the user-item interaction signal responsible for beating popularity.
- The measured node used the already-proven BPR parent; none of BPR's gain is attributable to this card.
- This fixed split was slightly negative on seed 0, so a retest needs a materially changed model stack or strength.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0002)
- live_06:node_009 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6026, single-seed Δ -0.0002 — rejected; 6 changed lines
