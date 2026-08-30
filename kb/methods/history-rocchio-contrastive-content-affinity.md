---
id: history-rocchio-contrastive-content-affinity
family: history
target_component: history
source: Rocchio relevance feedback; kb/data/facts.md §1–2; live_07:node_018
applies_when:
  - training outcomes and `time_ms` permit strictly earlier per-user positive and negative content profiles
  - video tag, music_id, and video_type are legally available from video_features_basic.csv
  - the ranker accepts an additional low-cardinality categorical field
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0003 over 1 measurement(s), so the promise is capped at the record; was: the only measurement was single-seed Δ -0.00030 on a five-seed FM-BPR ensemble, with no
  fresh-seed confirmation; therefore this exact fixed-level construction has no attributable positive gain
cost: 85 changed lines; 146 s measured versus 51 s for the parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-dcn-cross-head]
conflicts_with: [history-ordered-user-tag-affinity, history-last-positive-attribute-recurrence]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0003)]
evidence: [live_07:node_018]
---
## Claim
Append a seven-level candidate-content affinity field equal to the clipped difference between the user's strictly
earlier positive and negative match counts for first tag, music ID, and video type.

## Mechanism (why it moves within-user ranking)
Candidates matching attributes seen more often in prior positive than negative impressions receive a different
categorical value. Because candidate attributes vary across a user's rows, the field can alter within-user ordering;
equal-time groups are scored before their outcomes update the profile.

## How to implement on node_000
1. Join each video to cleaned `first_tag`, `music_id`, and `video_type`, mapping missing values to `UNK`.
2. Load train `time_ms` and stable-sort indices by `(user_id, time_ms, original_row)`.
3. Maintain three positive and three negative attribute-count dictionaries independently for each user.
4. Before updating an equal-time group, score each row by summing `positive[f][value] - negative[f][value]`.
5. Clip the sum to `[-3, 3]`, then update dictionaries from that group's `long_view` labels.
6. For valid or score-extra rows, compute levels from each user's final train dictionaries only.
7. Add `content_affinity` to `FIELDS`, predefine its vocabulary as the seven strings `-3` through `3`, and append
   the level in `raw()` and `encode()`.
8. To reproduce the measured stack, first apply BPR and five-member normalized-rank seed averaging unchanged.

## Risks / failure modes
- Any update before all equal-time rows are scored leaks a row or tied peer's outcome.
- Learned quantile edges from history scores can indirectly depend on later labels; use the fixed seven levels.
- The measured diff also inherited proven BPR and seed averaging from its parent; those gains are not attributable
  to this history field.
- First-tag parsing discards secondary tags, while raw count differences saturate quickly at the clipping limits.
- The exact construction reduced primary by 0.00030 in its only run and should be retested only on a changed stack.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0003)
- live_07:node_018 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6038, single-seed Δ -0.0003 — rejected; 85 changed lines
