---
id: features-online-item-exposure-momentum
family: features
target_component: features
source: kb/data/facts.md §5 (traffic-volume and temporal drift); Koren, KDD 2009, TimeSVD++; live_07:node_016
applies_when:
  - impression timestamps permit global video and author exposure histories to be computed strictly online
  - item or author exposure intensity changes over time and static identity embeddings cannot represent that drift
  - the model accepts low-cardinality categorical fields alongside video, author, tab, and duration
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0001 over 1 measurement(s), so the promise is capped at the record; was: the only probe measured single-seed Δ -0.00012 on a five-seed FM-BPR ensemble, trading
  GAUC −0.0008 for nDCG@5 +0.0006; without fresh-seed evidence, no positive net gain is attributable
cost: 97 changed lines; four categorical fields; measured runtime 157 s versus 51 s for the parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-exposure-session,
  model-lightgbm-lambdarank]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0001)]
evidence: [live_07:node_016]
---
## Claim
Append trailing-24-hour exposure-count and previous-exposure-gap buckets for the current video and author, using
only strictly earlier impression timestamps, to represent short-term catalogue momentum absent from static IDs.

## Mechanism (why it moves within-user ranking)
Exposure counts and gaps vary across the videos and authors shown to one user, so they can alter within-user order.
They encode logging momentum rather than outcomes: a rapidly circulating video or author receives different fields
from a dormant one even when their static embeddings are unchanged.

## How to implement on node_000
1. Add `deque` and four fields: `video_24h_count`, `video_previous_gap`, `author_24h_count`, `author_previous_gap`.
2. Implement `momentum_features(rows, vi, ti, vid2author, initial=None)` and stable-sort rows globally by `time_ms`.
3. Maintain per-video and per-author 24-hour timestamp deques plus dictionaries holding the last exposure time.
4. Process all rows sharing a timestamp before committing any of them, preventing equal-time rows from becoming history.
5. Encode counts as `min(count.bit_length(), 12)`.
6. Encode missing gaps as zero; otherwise use `min(max(gap_ms // 1000, 1).bit_length(), 20)`.
7. Build train features from empty state, then valid features from the returned train state.
8. Append the four bucket values in `raw()` and fit their vocabularies from train.
9. For `--score-extra`, rebuild from the same train state independently of valid and write rows in original order.

## Risks / failure modes
- Exposure momentum reflects the logging policy and traffic volume, not necessarily long-view relevance.
- Global traffic drift can make fixed 24-hour count buckets unstable between train, validation, and test.
- The measured node inherited BPR and five-seed rank averaging from its parent; none of their gain belongs to this card.
- The implementation retains last-seen times beyond 24 hours, so gap is lifetime previous-exposure gap while count
  alone is restricted to the trailing window.
- Dictionary/deque preprocessing materially increased runtime and memory pressure.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0001)
- live_07:node_016 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6040, single-seed Δ -0.0001 — rejected; 97 changed lines
