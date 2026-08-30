---
id: model-neural-factorization-machine
family: model
target_component: model
source: He and Chua, "Neural Factorization Machines for Sparse Predictive Analytics," SIGIR 2017,
  https://hexiangnan.github.io/papers/sigir17-nfm.pdf ([hexiangnan.github.io](https://hexiangnan.github.io/papers/sigir17-nfm.pdf))
applies_when:
  - the model already embeds sparse user, video, author, tab, and duration fields
  - FM's scalar sum of pair interactions may discard which latent dimensions produced the interaction
  - DCN supplied only a small gain on a field-aware stack, while DIN and LightGBM failed and the cheaper
    bi-interaction nonlinear head has not been tested
expected_delta: [0.000, 0.0008]
expected_delta_basis: NFM adds row-dependent nonlinear interaction capacity with very few parameters, but this
  dataset repeatedly shows architecture gains no larger than acceptance scale
cost: ~45 model-gradient lines; one 16-unit hidden layer; approximately 1.3x FM runtime; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-warp-within-user-rank-weighting, ensembling-seed-average,
  features-exposure-session]
conflicts_with: [model-attentional-factorization-machine, model-dcn-cross-head]
status: untried
evidence: []
---
## Claim
Retain the FM score and add a shallow Neural Factorization Machine residual over its vector-valued bi-interaction
pool, allowing nonlinear combinations of latent field interactions without a large deep network.

## Mechanism (why it moves within-user ranking)
Instead of immediately summing all latent interaction dimensions to one scalar, NFM forms
`p = 0.5 * (S² − sum_i E_i²)` and passes `p` through a small nonlinear layer. Candidate video, author, tab, and
duration change `p` across a user's rows, so the residual can change both metrics; purely user-constant information
still cannot affect the ordering.

## How to implement on node_000
1. Prefer the proven BPR edit, then expose the existing vector `p` whose element sum is the FM interaction score.
2. Add `A:(k,16)`, bias `c:16`, and output vector `q:16`, seeded from `--seed`.
3. Score `base_fm + relu(p @ A + c) @ q`, initializing `q` near zero so training starts close to FM.
4. Backpropagate through the ReLU to obtain `dp`; embedding gradients become
   `g * (1 + dp) * (S[:,None,:] - E)`.
5. Add Adam states for `A,c,q`, L2 `1e-5`, and hidden dropout 0.1 during training only.
6. Checkpoint the head with `V,W,b`; use the identical deterministic forward path for valid and score-extra.

## Risks / failure modes
- The five fields provide only ten interaction pairs, so the nonlinear head may merely relearn the FM sum.
- A wide or multilayer head will overfit sparse IDs; keep one 16-unit layer and early-stop on official primary.
- Hidden dropout must use each model member's seeded generator or five-seed runs cease to be reproducible.
- The head may improve calibration rather than order; BPR training is preferred so user-constant effects cancel.

## Measured
(none yet)
