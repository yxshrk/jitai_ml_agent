---
id: features-field-aware-author-pooling
family: features
target_component: features
source: Juan et al., RecSys 2016, Field-aware Factorization Machines; kb/data/facts.md §1–2
applies_when:
  - author_id is legally joined from video_features_basic.csv and has no evaluation cold start
  - a heterogeneous ensemble contains a field-aware branch and a separate standard-FM branch
  - author identity is already encoded, but extra author-specific capacity is being tested only in the field-aware branch
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated five-seed-per-branch experiment measured seed-mean Δ -0.00002 (t -0.09);
  the positive seed-0 movement was only +0.00012 and provides no attributable evidence above zero
cost: 26 changed lines; one extra field and author-sized id range in five field-aware members; numpy only
composes_with: [model-field-aware-fm-embeddings, loss-bpr-pairwise-within-user, ensembling-multiseed-heterogeneous-rank-blend]
conflicts_with: []
status: dead_under [official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0000)]
evidence: [live_04:node_027]
---
## Claim
Append a separately encoded copy of `author_id` only to the field-aware ensemble branch, giving that branch
additional author-specific interaction parameters while leaving the standard-FM branch unchanged.

## Mechanism (why it moves within-user ranking)
Authors vary across a user's impressions, so author interactions can alter within-user order. The duplicated field
adds no new information—the base model already contains author_id—but gives the field-aware branch a second,
independent author parameterization that can pool behavior across an author's videos.

## How to implement on node_000
1. First compose the field-aware and multiseed heterogeneous-ensemble cards.
2. Define `FIELD_AWARE_FIELDS = FIELDS + ['author_id_pool']`.
3. Set field-aware `V.shape` to `(field_aware_dim, len(FIELD_AWARE_FIELDS), k)` and iterate over that field count.
4. Set `field_aware_dim = dim + dims[2]`, where field 2 is the existing author vocabulary.
5. Build `Xtr_field_aware = column_stack((Xtr, Xtr[:,2] - off[2] + dim))`; do the same for valid.
6. Train and predict branch 0 with the expanded matrices; keep branch 1 on the original five-field matrices.
7. For score-extra, construct the identical appended author ids before field-aware prediction.
8. Preserve BPR sampling, five seeds per branch, independent early stopping, and the 0.6/0.4 rank blend.

## Risks / failure modes
- This duplicates an existing author field rather than introducing new author information, so it mainly adds
  redundant capacity and an artificial interaction between the two author copies.
- The measured stack already contained BPR, field-aware embeddings, seed averaging, and heterogeneous blending;
  none of their gains are attributable to this feature.
- The seed-confirmed delta was effectively zero despite a slightly positive seed-0 result.
- Applying the extra field to both branches would be a different experiment from the archived implementation.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings + heterogeneous-node-rank-average x1 (best Δ -0.0000)
- live_04:node_027 on [official FM + field-aware FM embeddings + heterogeneous-node-rank-average]: primary 0.6046, single-seed Δ +0.0001, seed-mean Δ -0.0000 (t -0.09) — rejected; 26 changed lines
