---
id: model-attentional-factorization-machine
family: model
target_component: model
source: Xiao et al., "Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks," IJCAI 2017; https://www.ijcai.org/proceedings/2017/435 ([ijcai.org](https://www.ijcai.org/proceedings/2017/435?utm_source=openai))
applies_when:
  - the current FM assigns every field-pair interaction the same fixed aggregation rule despite sharply different tab and duration effects (Facts §3–4)
  - global field-pair scalars failed, so useful weighting must depend on the particular row rather than only the field names
  - the champion has only five fields and therefore ten interaction vectors, keeping attention computation small on CPU
expected_delta: [0.000, 0.0010]
expected_delta_basis: row-conditioned interaction weighting is new, but field-aware and DCN results show that
  extra interaction capacity is usually worth no more than about 0.0005 here
cost: ~90 lines; ten attention entries per row; approximately 2x FM runtime and under 5 minutes for five members; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-exposure-session]
conflicts_with: [model-field-weighted-fm-relation-scalars]
status: untried
evidence: []
---
## Claim
Replace the FM's equal sum of pairwise dot products with AFM attention over the ten element-wise field-interaction
vectors, allowing interaction importance to change with the candidate video and context.

## Mechanism (why it moves within-user ranking)
For every field pair, AFM forms `e_ij = V[x_i] * V[x_j]` and computes a row-specific attention weight from
`ReLU(A e_ij + b)`. The weighted interaction vector is projected to a score. Because candidate video, author, tab,
and duration embeddings vary across a user's rows, the attention distribution also varies and can change both metrics.

## How to implement on node_000
1. Retain the five existing fields and construct the ten vectors `E[:,i] * E[:,j]` in `FM.logits`.
2. Add attention parameters `A:(h,k)`, `b:h`, `q:h`, and projection `p:k`, with `h=8`.
3. Compute logits `q @ relu(A @ e_ij + b)`, softmax over the ten pairs, and pooled interaction `10*sum(alpha*e_ij)`.
4. Replace the ordinary FM interaction with `pooled @ p`; initialize attention logits to zero and `p` to ones so epoch zero reproduces FM.
5. Back-propagate through the projection, softmax, ReLU, and pair products into the existing embedding gradients.
6. Train with the proven same-user BPR sampler, Adam, attention dropout 0.1, and L2 `1e-5` on attention parameters.
7. Checkpoint all added arrays, preserve validation-primary early stopping, and use the identical forward path for `--score-extra`.

## Risks / failure modes
- Attention may collapse to tab-related pairs already learned by the FM, yielding no complementary ordering.
- The exact-FM initialization requires the factor of ten and an all-ones projection; omitting either changes scale sharply.
- Five fields provide only ten pairs, so an attention hidden width above eight adds needless overfitting capacity.
- Do not feed user-only side features into the attention block unless they interact with row-varying fields.

## Measured
(none yet)
