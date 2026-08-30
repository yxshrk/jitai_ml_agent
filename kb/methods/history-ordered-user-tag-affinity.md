---
id: history-ordered-user-tag-affinity
family: history
target_component: history
source: kb/data/facts.md §1 (closed catalogue and available video tags); §10.3 (creator affinity is sparse)
applies_when:
  - video tags are legally available from video_features_basic.csv
  - training rows have time_ms so user-tag outcomes can be restricted to strictly earlier impressions
  - user-author history is sparse enough that semantic sharing across creators may be useful
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0004 over 1 measurement(s), so the promise is capped at the record; was: the isolated five-seed FM-BPR ensemble probe lost 0.00037 primary on seed 0 and received no
  fresh-seed confirmation, so this exact ordered, smoothed, bucketized construction has no attributable positive gain
cost: 83 changed lines; equal-time-safe history preprocessing plus one FM field; 65 s measured on a five-model ensemble
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0004)]
evidence: [live_06:node_014, ceiling:oracle]
---
## Claim
Add a categorical field representing the user's smoothed prior long-view rate for the current video's tags,
computed from strictly earlier training impressions and mapped to ten quantile buckets plus no-history.

## Mechanism (why it moves within-user ranking)
Different videos carry different tags, so user-tag affinity varies across one user's rows and can change both
metrics. Pooling evidence by semantic tag can transfer preference information across creators for which direct
user-author history is unavailable.

## How to implement on node_000
1. Read `tag` with video and author metadata; split on comma, semicolon, or pipe, strip wrappers, remove missing
   markers, and deduplicate each video's tags.
2. Read training `time_ms`, sort rows stably by time, and process all equal-time rows as one batch.
3. Before committing a batch, sum `(count, positives)` over every `(user, tag)` attached to each row.
4. Compute `(tag_positives + 5 * user_prior) / (tag_count + 5)` when tag history exists.
5. After scoring the equal-time batch, update user totals and every `(user, tag)` total with its labels.
6. Fit nine quantile edges on known training rates and encode ten levels plus `NOHIST`.
7. Append `user_tag_rate` to `FIELDS`, `raw`, vocab construction, and encoded matrices.
8. For validation and score-extra rows, derive buckets only from final training totals.

## Risks / failure modes
- Multi-tag rows sum evidence across tags, so frequently co-occurring tags can count the same historical impression
  multiple times and dominate the statistic.
- Sparse or noisy tags leave many rows in `NOHIST`; video identity may already absorb most tag information.
- Committing equal-time rows before updating is mandatory; immediate updates would leak peer outcomes.
- The archived node inherited BPR and five-seed rank averaging already measured by sibling nodes; their gains are
  not attributable to this history field.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0004)
- live_06:node_014 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6036, single-seed Δ -0.0004 — rejected; 83 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'user-context-taste' — facts §11.2 row 'user × tab / duration / tag / type taste': other-half rates ≤ +0.0003 (facts §11, kb/data/screens/CEILING.md)
