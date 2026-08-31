---
id: ensembling-creator-role-format-branch
family: ensembling
target_component: ensembling
source: task foundations §2 (user-only features require row-varying interactions); kb/data/facts.md §1 (closed catalogue and sparse unseen-user cohort)
applies_when:
  - a heterogeneous ensemble contains field-aware and standard FM branches
  - legal user creator-role fields and candidate video-format fields are available from the side tables
  - the standard branch can remain unchanged as an anchor while the field-aware branch receives pooled cohort context
expected_delta: [0.0, 0.0]
expected_delta_basis: the only run was single-seed flat at −0.00002 with no seed confirmation, so no positive gain can
  be claimed; the parent already contained BPR, five-member branch averaging, field-aware embeddings, and rank fusion
cost: 51 changed lines; ten training phases; measured runtime 211 s versus 121 s for the parent; numpy only
composes_with: [ensembling-multiseed-heterogeneous-rank-blend, model-field-aware-fm-embeddings,
  loss-bpr-pairwise-within-user]
conflicts_with: [ensembling-relation-support-adaptive-architecture-fusion]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)]
evidence: [live_09:node_015]
---
## Claim
Add a categorical `user creator role × candidate video format × tab` cross only to the field-aware members of a
heterogeneous FM ensemble, leaving the standard members unchanged as an ID-based anchor.

## Mechanism (why it moves within-user ranking)
User creator role is constant within a user and cannot affect either metric alone. Crossing it with candidate
`video_type`, `upload_type`, and `tab` makes the value vary across that user's rows, allowing the field-aware branch
to learn pooled format preferences while the unchanged standard branch limits disruption.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`, `model-field-aware-fm-embeddings`, and the ten-member heterogeneous rank blend.
2. Read `video_type,upload_type` with each video and `is_video_author,is_live_streamer` with each user.
3. Define `role_format(user, video, tab)` as the joined tuple of those four side attributes and `tab`.
4. Build `cross_vocab` from training rows only and reserve one unknown ID.
5. Append the offset cross ID to `Xtr_field` and `Xva_field`; keep standard `Xtr` and `Xva` unchanged.
6. Set `field_dim = dim + len(cross_vocab) + 1`.
7. Parameterize field-aware FM shapes and interaction loops by `field_count=Xtr_field.shape[1]`.
8. Train field-aware members on the augmented matrices and standard members on the original matrices.
9. Apply the identical cross encoding to `--score-extra`, then preserve the existing 0.6/0.4 rank blend.

## Risks / failure modes
- The measured diff retained the parent's proven BPR and heterogeneous multiseed blend, so this card can claim only
  the branch-local cross effect, not the ensemble's overall performance.
- Static format information may already be absorbed by closed-catalogue video embeddings.
- Sparse role-format-tab combinations map to unknown or overfit despite their low-cardinality ingredients.
- Applying the cross to both branches would no longer test the anchored branch-diversification mechanism.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)
- live_09:node_015 on [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend]: primary 0.6043, single-seed Δ -0.0000 — rejected; 51 changed lines
