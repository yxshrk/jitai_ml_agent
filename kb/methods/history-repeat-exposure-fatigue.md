---
id: history-repeat-exposure-fatigue
family: history
target_component: history
source: kb/data/facts.md §8 (5.7% of valid rows belong to repeated user-video pairs); live_04:node_005
applies_when:
  - repeated user-video impressions exist and otherwise receive identical static features
  - time_ms is available so exposure history can be restricted to strictly earlier impressions
  - evaluation rows follow training rows chronologically
expected_delta: [0.000, 0.000]
expected_delta_basis: the only measurement was a single-seed primary loss of 0.00050 on field-aware FM with BPR;
  without seed confirmation there is no attributable positive gain to bracket
cost: ~56 lines; runtime ~1.6x versus the field-aware parent (44 s measured); numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, data-weighting-recency]
conflicts_with: []
status: dead_under [official FM + field-aware FM embeddings x1 (best Δ -0.0005)]
evidence: [live_04:node_005]
---
## Claim
Add categorical fields for the number of strictly prior user-video exposures and the logarithmic time since the
last exposure, allowing repeated impressions with otherwise identical identifiers to receive different scores.

## Mechanism (why it moves within-user ranking)
Prior exposure count and recency vary across repeated rows for the same user and video, so they can change
within-user order. The model can represent familiarity or fatigue instead of assigning every repeated pair the
same score.

## How to implement on node_000
1. Add `uv_count` and `uv_gap` to `FIELDS`, and read `time_ms` for train, valid, and score-extra rows.
2. Implement `repeat_history(rows, ui, vi, mi, counts, last)` and stable-sort row indices by `time_ms`.
3. Process equal-time groups in two passes: first extract features, then update shared history dictionaries.
4. Encode count as `str(min(prior_count, 3))`.
5. Encode gap as `NEVER` for first exposure; otherwise use
   `str(min(20, int(log2(1 + max(0, time-last_time)/1000))))`.
6. Build train history first, then validation history with the same `counts` and `last` dictionaries.
7. Extend `raw()` and `encode()` to append both history values.
8. For score-extra, continue the dictionaries after validation and apply the identical encoding path.

## Risks / failure modes
- Only the repeated-pair cohort can benefit directly, limiting impact.
- The archived field-aware-BPR child lost 0.00050 primary, mainly through nDCG@5, so fatigue categories may
  disturb top-list ordering more than they resolve repeated-row ties.
- Carrying history between splits is valid only when timestamps establish that every incorporated exposure is
  strictly earlier; equal-time rows must never update one another.
- The diff retained the parent's field-aware representation and BPR loss, so the measured movement is attributable
  to these two history fields rather than a simultaneous loss change.

**Retest idea (facts §10.4):** the −0.0005 single seed was unconfirmed and the encoding was count-only. A cleaner
version uses the within-window exposure sequence (earlier valid rows included, labels never), tab-aware (tab 1: 0.292
vs 0.389), and composes with `history-same-author-run-features`; retest on a stack without it.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + field-aware FM embeddings x1 (best Δ -0.0005)
- live_04:node_005 on [official FM + field-aware FM embeddings]: primary 0.6025, single-seed Δ -0.0005 — rejected; 56 changed lines
