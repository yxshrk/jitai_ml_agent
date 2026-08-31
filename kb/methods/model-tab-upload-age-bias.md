---
id: model-tab-upload-age-bias
family: model
target_component: model
source: Koren, "Collaborative Filtering with Temporal Dynamics," KDD 2009 (TimeSVD++); live_09:node_022
applies_when:
  - impression date and legal video upload_dt are available, allowing candidate age at show time to be computed
  - upload ages vary across a user's candidate rows, so a tab-conditioned age term can alter within-user ordering
  - a field-aware branch can receive the temporal term while an unchanged ensemble branch anchors ID-based rankings
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0001 over 1 measurement(s), so the promise is capped at the record; was: attribution-adjusted measurement: the only measured fresh-seed mean was -0.00015 on an existing
  BPR heterogeneous ensemble, so no positive gain can be promised for the upload-age bias
cost: 106 changed lines on the heterogeneous parent; ten training phases; measured runtime 165 s; numpy and datetime
composes_with: [model-field-aware-fm-embeddings, loss-bpr-pairwise-within-user, ensembling-multiseed-heterogeneous-rank-blend]
conflicts_with: [model-continuous-date-drift-head]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0001)]
evidence: [live_09:node_022]
---
## Claim
Add a zero-initialized additive bias indexed by impression tab and bucketed candidate age since upload to the
field-aware members of a heterogeneous BPR ensemble, leaving the standard-FM members unchanged.

## Mechanism (why it moves within-user ranking)
Candidate age is row-varying even when impression date is constant within a user. A learned tab-by-age bias can
therefore express tab-specific freshness preferences and change within-user order without relying on leaky outcome
statistics. Restricting it to one branch preserves architecture diversity.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`, `model-field-aware-fm-embeddings`, and the heterogeneous rank blend.
2. Add `day_number()` to parse YYYYMMDD, Unix timestamps, or ISO dates into ordinal days.
3. Load `upload_dt` from the basic video table and `date` from train, valid, and score-extra rows.
4. Compute `log1p(max(show_day-upload_day, 0))`; retain a separate missing-age level.
5. Fit six quantile knots on finite training ages, producing seven finite buckets plus missing.
6. Encode `temporal_id = tab_code * 8 + age_bucket`.
7. Give field-aware members a zero-initialized `age_bias` and matching Adam states.
8. Add `age_bias[temporal_id]` in their logits and scatter positive/negative BPR gradients into it.
9. Include `age_bias` in L2, checkpoint restoration, validation prediction, and score-extra prediction.
10. Pass no temporal term to standard members and preserve the existing 0.6/0.4 rank blend.

## Risks / failure modes
- The measured implementation had fresh-seed mean delta -0.00015 (z -0.51), despite seed-0 delta +0.00011.
- The parent already contained BPR, five field-aware members, five standard members, and rank fusion; none of their
  gain is attributable to this card.
- Closed-catalogue video embeddings may already absorb upload-age preferences, and the video-side oracle bound is
  only about +0.0003.
- Invalid upload dates need a dedicated bucket; silently mapping them to newly uploaded videos creates distortion.
- This overlaps conceptually with continuous date-drift heads, so combining both may duplicate temporal capacity.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0001)
- live_09:node_022 on [official FM + loss-bpr-pairwise-within-user + ensembling-multiseed-heterogeneous-rank-blend]: primary 0.6044, single-seed Δ +0.0001, seed-mean Δ -0.0001 (z -0.51) — rejected; 106 changed lines
