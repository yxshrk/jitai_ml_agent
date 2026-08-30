---
id: encoding-ordered-item-context-target-statistics
family: encoding
target_component: encoding
source: Prokhorenkova et al. 2018, CatBoost (ordered target statistics); kb/data/facts.md §1, §4
applies_when:
  - training rows have timestamps permitting target statistics from strictly earlier impressions
  - video and author identities recur across splits, and tab varies within users
  - the model can consume bucketized categorical fields
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0004 over 1 measurement(s), so the promise is capped at the record; was: the isolated single-seed probe was negative, so this encoding currently has no attributable
  positive gain; retain it only for materially changed model or smoothing stacks
cost: ~57 lines; measured runtime ~2x the field-aware parent; numpy and standard library only
composes_with: [loss-listwise-softmax-within-user, model-dcn-cross-head, data-weighting-recency]
conflicts_with: [history-user-aggregates]
status: dead_under [official FM + field-aware FM embeddings x1 (best Δ -0.0004)]
evidence: [live_04:node_009, ceiling:oracle]
---
## Claim
Append categorical buckets representing ordered, smoothed long-view rates for `video×tab` and `author×tab`,
using only earlier training impressions for each training row and full training totals for later splits.

## Mechanism (why it moves within-user ranking)
The fields vary across a user's impressions and expose context-specific item quality directly. Ordered construction
prevents a training row's label, including labels at the same timestamp, from entering its own encoded rate.

## How to implement on node_000
1. Add `video_tab_rate` and `author_tab_rate` to `FIELDS`; read training `time_ms`.
2. Define `rate_bucket(stat)` as `min(19, int(20*(positives+2)/(count+6)))`, defaulting to `(0,0)`.
3. In `ordered_rate_bins`, stable-sort training rows by `time_ms`.
4. For each equal-time group, first encode rates from existing `(video,tab)` and `(author,tab)` dictionaries.
5. Only after encoding the entire group, update its counts and positive totals.
6. Return both training bucket arrays and the completed statistics dictionaries.
7. For validation and score-extra, use `full_rate_bins` against completed training statistics only.
8. Extend `raw` and `encode` to append both bucket strings; build vocabularies from ordered training buckets.
9. Preserve the parent model, loss, optimizer, early stopping, and prediction logic unchanged.

## Risks / failure modes
- The actual edit added only these two fields; the parent already contained field-aware embeddings and BPR, so
  no gain from those mechanisms is attributable to this card.
- Ordered train encodings see shorter histories than validation encodings, creating train/evaluation mismatch.
- Twenty hard bins can discard confidence from well-supported keys and amplify noise for rare combinations.
- On field-aware FM, explicit item-tab statistics may duplicate interactions already represented by embeddings.
- Equal-timestamp rows must be encoded before any row in that timestamp updates the dictionaries.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings x1 (best Δ -0.0004)
- live_04:node_009 on [official FM + field-aware FM embeddings]: primary 0.6027, single-seed Δ -0.0004 — rejected; 57 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'item-side' — facts §11 row 'video / author side, any period': valid-week LOO rate from the valid labels +0.0003, the whole-month statistics file +0.0000 (facts §11, kb/data/screens/CEILING.md)
