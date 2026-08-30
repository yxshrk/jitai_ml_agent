---
id: aux-targets-is-click
family: aux-targets
target_component: aux-targets
source: kb/literature/multitask/1804.07931_esmm.pdf (shared embeddings, multi-task over the impression space); kb/data/facts.md §6; organizer README "unexplored" #3
applies_when:
  - other feedback columns exist on train rows as legal targets (task.md)
  - is_click correlates 0.76 with long_view and is present on 46 % of rows (facts §6); the rarer signals (like 1.9 %, follow 0.1 %) are too sparse
expected_delta: [0.000, 0.005]
expected_delta_basis: extra supervision on shared embeddings helps sparse users, but P(click | long_view) = 0.996
  means is_click carries little information the label does not — gains may be within noise; test cheaply
cost: ~30 lines (second bias vector + shared-V gradient); runtime ~1.2x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, features-duration-unknown-flag, data-weighting-recency, model-dcn-cross-head]
conflicts_with: []
status: dead_under {run: live_02, stack: official FM + loss-bpr-pairwise-within-user, delta: -0.0003}
evidence: [live_02:node_011]
---
## Claim
Train a second logistic head for is_click on the same embeddings (ESMM-style shared bottom), weight 0.2–0.5, and
keep the long_view head as the score.

## Mechanism (why it moves within-user ranking)
is_click is "valid play >= 7 s" in this UI — a coarser threshold on the same watch-time mechanism. Rows that are
clicked-but-not-long-viewed (28 % of clicks) are the near-misses; a shared embedding trained on both thresholds
sees a smoother version of the label, which regularises sparse user vectors.

## How to implement on node_000
1. Read is_click for train rows (train.csv only; valid has no outcome columns).
2. Second head: s2 = b2 + w2[x].sum() + interaction(V) (same code path as logits, own bias vector w2, b2).
3. Gradient: g1 = sigmoid(s1) − y, g2 = lambda x (sigmoid(s2) − y_click); V receives both, w1 only g1, w2 only g2.
4. Prediction = s1. Sweep lambda in {0.2, 0.5}.

## Risks / failure modes
- Near-duplicate signal: if flat, do not try like/follow/comment (rarer and less related) — park the family.
- Two heads double the interaction computation; keep it vectorised.

## Measured
- live_02:node_011 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6028, single-seed Δ -0.0003 — rejected; 35 changed lines
