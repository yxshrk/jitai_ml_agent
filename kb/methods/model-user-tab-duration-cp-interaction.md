---
id: model-user-tab-duration-cp-interaction
family: model
target_component: model
source: Blondel et al., 2016, Higher-Order Factorization Machines; tensor-factorization practice
applies_when:
  - the ranker exposes categorical user, tab, and duration-bucket fields
  - pairwise FM interactions cannot represent a user's duration preference changing by tab
  - same-user BPR training and analytic embedding-gradient updates are available
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0007 over 1 measurement(s), so the promise is capped at the record; was: the isolated probe gained only +0.00007 on seed 0 and had fresh-seed mean Δ -0.00069,
  so no positive attributable gain is supported for this exact rank-k cubic construction
cost: 27 changed lines; one additional `(dim, k)` table and Adam states; measured runtime 17 s (~1.2x); numpy only
composes_with: [loss-bpr-pairwise-within-user, features-fine-duration-and-tab-cross, regularization-embedding-dropout-l2, ensembling-seed-average]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0007)]
evidence: [live_06:node_006, ceiling:oracle]
---
## Claim
Add a rank-k CP term `sum(T[user] * T[tab] * T[duration_bucket])` to an FM score, allowing personalized duration
preferences to vary by tab rather than being represented only through pairwise interactions.

## Mechanism (why it moves within-user ranking)
The cubic term varies across a user's rows when tab or duration changes. Its shared latent factors directly model
a user×tab×duration interaction absent from a second-order FM, so it can alter both within-user metrics.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; preserve its sampler and ordinary FM parameters.
2. Add `T ~ Normal(0, 0.05)` with shape `(dim, k)` plus zeroed Adam states `mT` and `vT`.
3. In `logits`, gather `C = T[X[:, (0, 3, 4)]]` and add `(C[:,0] * C[:,1] * C[:,2]).sum(1)`.
4. In each positive/negative BPR pass, scatter `h*C_other1*C_other2` into the corresponding user, tab, and
   duration rows of `gT`.
5. Add `l2*T` to `gT` and update `T` through the same Adam loop as `V` and `W`.
6. Include `T` in every best-validation checkpoint and restore it before writing valid or extra predictions.

## Risks / failure modes
- The measured node inherited proven BPR from its parent; none of BPR's gain is attributable to this cubic term.
- A single shared `T` table couples its user, tab, and duration roles, while initialization at 0.05 adds substantial
  capacity and caused the validation peak to move to epoch 4 before declining.
- The exact construction had negative fresh-seed mean evidence and should only be retested after a material change
  such as role-specific factor tables, stronger regularization, or a different ensemble stack.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0007)
- live_06:node_006 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6029, single-seed Δ +0.0001, seed-mean Δ -0.0007 (z -1.54) — rejected; 27 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'user-context-taste' — facts §11.2 row 'user × tab / duration / tag / type taste': other-half rates ≤ +0.0003 (facts §11, kb/data/screens/CEILING.md)
