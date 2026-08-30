---
id: model-user-video-tab-cp-interaction
family: model
target_component: model
source: Karatzoglou et al., "Multiverse Recommendation: N-dimensional Tensor Factorization for
  Context-aware Collaborative Filtering," RecSys 2010; kb/data/facts.md §1, §4
applies_when:
  - user_id, video_id, and tab are legal row features and tab varies within users
  - the catalogue is closed, so user-video factors can be estimated without item cold start
  - an FM ranker supports analytic gradients for an additional factorized interaction term
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated addition measured fresh-seed mean Δ -0.00053 and seed-0 Δ +0.00006 on the
  five-seed FM-BPR ensemble, providing no attributable positive evidence
cost: 27 changed lines; one additional `(dim, k)` table plus Adam states; measured runtime 56 s versus 45 s for
  the parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, regularization-embedding-dropout-l2,
  model-field-aware-fm-embeddings]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)]
evidence: [live_06:node_018]
---
## Claim
Add a separate rank-k CP interaction
`sum(T[user] * T[video] * T[tab])` to the FM score so a user's preference for a video can vary by tab.

## Mechanism (why it moves within-user ranking)
The cubic term varies across a user's videos and tabs, so it can alter both metrics. Unlike the FM's pairwise
user-video, user-tab, and video-tab terms, the CP product represents a tab-conditioned user-video interaction with
parameters shared through low-rank factors.

## How to implement on node_000
1. Apply `loss-bpr-pairwise-within-user`; retain the existing five categorical field order.
2. Add `T ~ Normal(0, 0.05)` with shape `(dim, k)` and matching zero-initialized Adam states.
3. In `logits`, set `C = T[X[:, (0, 1, 3)]]` and add `(C[:,0] * C[:,1] * C[:,2]).sum(1)`.
4. Return `C` with the ordinary FM intermediates for both positive and negative rows.
5. Scatter each BPR logit gradient into the three selected `T` rows, multiplying by the other two CP factors.
6. Add `l2*T`, update `T` with Adam, and include `T` in every early-stop checkpoint and restoration.

## Risks / failure modes
- The measured fresh-seed effect was negative despite a tiny positive seed-0 screen, indicating seed noise or
  harmful redundant capacity.
- Tab is already a strong FM field, and user-video identity is already modeled pairwise; the cubic term may mostly
  duplicate existing interactions and overfit sparse user-video-tab triples.
- Initializing each CP factor at 0.05 makes cubic gradients sensitive to scale; stronger L2 or smaller initialization
  would be a materially changed retest.
- The measurement used both proven BPR and five-seed rank averaging, but the diff against that parent isolated only
  this CP term; no gain from those parent mechanisms is attributable to this card.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)
- live_06:node_018 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6040, single-seed Δ +0.0001, seed-mean Δ -0.0005 (z -1.23) — rejected; 27 changed lines
