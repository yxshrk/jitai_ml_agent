---
id: model-dcn-cross-head
family: model
target_component: model
source: kb/literature/models/2008.13535_dcn-v2.pdf (cross network); kb/literature/models/1703.04247_deepfm.pdf; organizer README "unexplored" #5
applies_when:
  - the loss/feature levers have been tried first (organizers: capacity is not the bottleneck — k = 8/16/32 flat)
  - a head that adds explicit higher-order interactions on the same 5 x 16 embeddings can be back-propagated by hand (numpy only)
expected_delta: [0.000, 0.005]
expected_delta_basis: organizers rank architecture after losses/features and measured embedding size flat; DCN-V2's
  gain in the literature comes from explicit bounded-degree crosses, which an FM lacks — modest here
cost: ~120 lines (cross layer forward/backward, Adam for new matrices); runtime 2–3x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, features-duration-unknown-flag, data-weighting-recency, aux-targets-is-click, regularization-embedding-dropout-l2]
conflicts_with: []
status: untried
evidence: []
---
## Claim
Concatenate the field embeddings (x0, 80-d), apply one DCN-V2 cross layer x1 = x0 * (W x0 + b) + x0, and score with
w . x1 added to the FM logit.

## Mechanism (why it moves within-user ranking)
The FM only models pairwise dot products of field vectors; a cross layer learns weighted element-wise products
across all fields (degree-2 crosses with a full matrix), so "user-taste x tab x duration" style interactions become
representable without a deep net that would overfit 1.1 M rows.

## How to implement on node_000
1. x0 = E.reshape(B, F*k). Forward: h = x0 @ W.T + b; x1 = x0 * h + x0; s = FM logit + x1 @ v.
2. Backward for a batch gradient g (d loss / d s): dv = x1.T @ g; dx1 = g[:, None] * v; dh = dx1 * x0;
   dW = dh.T @ x0; db = dh.sum(0); dx0 = dx1 * (h + 1) + dh @ W; scatter dx0 back into the field embeddings
   (add to gV via np.add.at with dx0.reshape(B, F, k)).
3. Adam for W (80 x 80), b, v as the existing code does for V and W; init W small (0.01).
4. Regularise: dropout on x0 (p 0.1) or L2 1e-4 on W — the head overfits faster than the FM.

## Risks / failure modes
- Hand-written backprop errors: verify with a finite-difference check on a tiny batch before the full run.
- 2–3x runtime: still ~40 s, but keep SMOKE_EPOCHS honoured.

## Measured
(none yet)
