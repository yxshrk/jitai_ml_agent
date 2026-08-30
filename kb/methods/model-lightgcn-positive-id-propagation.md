---
id: model-lightgcn-positive-id-propagation
family: model
target_component: model
source: He et al., SIGIR 2020, "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
applies_when:
  - train outcomes may define a positive-only user-video graph without using validation or test outcomes
  - user and video ids recur across splits, making propagated endpoint embeddings usable at scoring time
  - a standard-FM branch already contains a raw user-video dot product that can be replaced in isolation
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0002 over 1 measurement(s), so the promise is capped at the record; was: the only probe lost 0.0002 primary without seed confirmation when replacing the standard-FM
  branch inside a multiseed heterogeneous ensemble, so there is no attributable positive gain to bracket
cost: ~42 changed lines; measured runtime 423 s versus 143 s for the ensemble parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-multiseed-heterogeneous-rank-blend, regularization-embedding-dropout-l2]
conflicts_with: []
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0002)]
evidence: [live_04:node_024, ceiling:oracle]
---
## Claim
Replace a standard FM's raw user-video dot product with one layer of degree-normalized propagation over the
positive training user-video graph, while retaining its tab, duration, author, and other FM interactions.

## Mechanism (why it moves within-user ranking)
For embeddings `V`, compute `G = 0.5 * (V + A_norm V)` and score the user-video interaction as `G_u·G_v`.
The propagated user representation pools positively connected videos, while propagated videos pool their positive
users; because candidate videos vary within a user, this term can change the scored ordering.

## How to implement on node_000
1. Build graph edges from training rows with `long_view=1`, using encoded user and video ids only.
2. Set each edge weight to `1/sqrt(degree[user]*degree[video])`; store both sorted adjacency directions.
3. Implement `neighbor_sum(P)` with `np.add.reduceat` and `np.add.at`.
4. Define `propagate()` as `0.5*(V + neighbor_sum(V))`.
5. In standard-FM logits, replace raw `E_user·E_video` by `G_user·G_video`.
6. Recompute `G` from current parameters for every BPR update; reuse one `G` across prediction batches.
7. Subtract the raw user-video FM gradient, accumulate endpoint gradient `dG` for the propagated interaction,
   then apply `dV += 0.5*dG + 0.5*neighbor_sum(dG)`.
8. Preserve all other FM interactions, BPR sampling, early stopping, and output logic.

## Risks / failure modes
- Full-graph propagation and its Jacobian on every update made the measured ensemble roughly three times slower.
- Building the graph from all interactions rather than positive training rows changes the tested mechanism.
- Stale epoch-level propagation or direct endpoint updates without the adjacency Jacobian optimize the wrong score.
- The archived node retained the parent's BPR loss, five-seed branches, and 0.6/0.4 rank blend; its −0.0002
  movement reflects replacing only the standard branch's user-video interaction, attenuated by the ensemble weight.
- Sparse positive neighborhoods may amplify exposure bias rather than provide useful smoothing.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0002)
- live_04:node_024 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6043, single-seed Δ -0.0002 — rejected; 42 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0002 for the signal family 'history-cf' — facts §11.2 row 'train-history taste, item-kNN, repeats': ≤ +0.0002 each; history-user-aggregates measured 0 on BPR (facts §11, kb/data/screens/CEILING.md)
