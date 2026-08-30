---
id: aux-targets-discrete-watch-survival-queries
family: aux-targets
target_component: aux-targets
source: kb/data/facts.md §3 (duration-defined long_view and play_time_ms); discrete-time survival modelling
applies_when:
  - play_time_ms is available as a legal training-only target
  - scored rows provide duration_ms, allowing the required threshold min(duration_ms, 18000) to be encoded
  - a shared FM can consume a categorical watch-threshold query field
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0000 over 1 measurement(s), so the promise is capped at the record; was: the isolated pointwise-FM probe supplied no positive fresh-seed evidence, so this exact
  three-query construction has no attributable expected gain until a materially changed stack confirms otherwise
cost: ~30 changed lines; expanded training batches measured at ~4.5x baseline runtime; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-dcn-cross-head, model-field-aware-fm-embeddings, data-weighting-recency]
conflicts_with: [aux-targets-ordinal-watch-depth, loss-watchtime-censored]
status: dead_under [official FM x1 (best Δ -0.0007)]
evidence: [live_05:node_001, ceiling:oracle]
---
## Claim
Condition one shared FM on an absolute watch-threshold field and augment native long-view examples with attainable
7 s, 12 s, and 18 s survival queries labelled from training-only play_time_ms.

## Mechanism (why it moves within-user ranking)
The native scored row queries whether watch time reaches `min(duration_ms, 18000)`. Auxiliary absolute-threshold
rows reuse the same interaction parameters with denser supervision about watch survival, while the threshold field
allows user, item, tab, and duration interactions to vary by the queried watch depth.

## How to implement on node_000
1. Append `watch_threshold` to `FIELDS`.
2. Read `play_time_ms` from train only.
3. Define `threshold_bucket(dur)` as `unknown` for nonpositive duration, otherwise ceiling seconds after clipping
   duration to `[0,18000]`; append it in `raw`.
4. Ensure categorical levels `7`, `12`, and `18` exist in the threshold vocabulary.
5. Build `durtr`, `playtr`, and query specifications for 7000, 12000, and 18000 ms.
6. For each query, mark rows attainable only when `duration_ms >= threshold`, and label with
   `play_time_ms >= threshold`.
7. In every native pointwise minibatch, copy attainable rows, replace the final field by the query id, concatenate
   native and query examples, and call the unchanged `FM.step`.
8. Keep validation, checkpoint selection, output scoring, and score-extra inference on each row's native threshold.

## Risks / failure modes
- Auxiliary rows substantially increase training work and can dominate the native long-view objective.
- Excluding rows shorter than a query avoids impossible targets but changes the cohort mixture across thresholds.
- This differs from ordinal relative-depth heads: it uses one categorical query-conditioned FM and absolute times.
- A BPR composition requires an explicit hybrid objective; the measured edit itself remained purely pointwise.
- Duration zero maps to `unknown`, so it receives no auxiliary threshold examples.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM x1 (best Δ -0.0007)
- live_05:node_001 on [official FM]: primary 0.6015, single-seed Δ +0.0000, seed-mean Δ -0.0007 (z -2.83) — rejected; 30 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'item-side' — facts §11.2 row 'video / author side, any period': valid-week LOO video rate +0.0003, leaky month statistics +0.0000; the auxiliary outcomes measured directly as target statistics on node_003 (kb/data/screens/RESULTS.md): video click rate −0.0004, play-through +0.0003; the aux-target cards' own records +0.0002 / +0.0003 (facts §11, kb/data/screens/CEILING.md)
