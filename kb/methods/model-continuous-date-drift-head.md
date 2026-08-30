---
id: model-continuous-date-drift-head
family: model
target_component: model
source: Koren, "Collaborative Filtering with Temporal Dynamics," KDD 2009 (TimeSVD++); kb/data/facts.md §5
applies_when:
  - impression date is available as a legal show-time feature
  - training and evaluation periods exhibit temporal drift in volume or positive rate
  - the ranker supports row-specific numeric terms and gradients, such as same-user BPR
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0001 over 1 measurement(s), so the promise is capped at the record; was: the isolated date-head probe on standard FM+BPR had fresh-seed mean Δ -0.00001 despite
  single-seed Δ +0.00013, providing no attributable positive evidence for this linear extrapolation recipe
cost: ~61 changed lines; one scalar and one dim-sized slope array with Adam states; measured runtime ~1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, data-weighting-recency, model-field-aware-fm-embeddings,
  regularization-embedding-dropout-l2]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0000)]
evidence: [live_05:node_013, ceiling:oracle]
---
## Claim
Add a normalized continuous-date term
`date * (global_slope + video_slope[video] + tab_slope[tab])` to the FM score.

## Mechanism (why it moves within-user ranking)
Video- and tab-specific slopes let otherwise static item/context preferences change over calendar time. The term
can alter within-user ordering when impressions differ in date, video, or tab; the global slope matters only when
a user's compared rows occur on different dates.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; load `date` for train, valid, and score-extra rows.
2. Parse dates to ordinals and set `t=(day-max_train_day)/(max_train_day-min_train_day)`.
3. Add zero-initialized `D.shape=(dim,)`, scalar `G`, and matching Adam first/second moments.
4. Change logits to accept `t` and add `t*(G + D[X[:,1]] + D[X[:,3]])`.
5. In each positive/negative BPR pass, scatter `h*t` into `gD` at video and tab ids and add `dot(h,t)` to `gG`.
6. Apply the existing L2 and Adam updates to `D` and `G`.
7. Include `D,G` in best-state checkpoint restoration and pass normalized dates through every prediction path.

## Risks / failure modes
- Linear slopes extrapolate beyond the training date range and can become unstable on later score-extra dates.
- Sparse video-specific slopes may fit transient noise rather than persistent drift.
- Same-date global terms cancel under BPR and cannot affect within-user order.
- The archived edit was isolated on an existing BPR parent; none of BPR's established gain is attributable to this head.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0000)
- live_05:node_013 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6038, single-seed Δ +0.0001, seed-mean Δ -0.0000 (z -0.02) — rejected; 61 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0000 for the signal family 'recency-volume' — facts §11.3: train + half of valid on the other half 0.5821 vs 0.5826; windows from 04-12 / 04-14 / 04-15 lose 0.003–0.008 — volume, not recency (facts §11, kb/data/screens/CEILING.md)
