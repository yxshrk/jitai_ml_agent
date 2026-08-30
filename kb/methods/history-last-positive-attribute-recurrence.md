---
id: history-last-positive-attribute-recurrence
family: history
target_component: history
source: kb/data/facts.md §10.1 (label-conditioned continuation) and §10.3 (sparse exact-creator history)
applies_when:
  - training outcomes and time_ms can identify each user's latest strictly earlier positive impression
  - video tag, music_id, and video_type are legally available from the basic side table
  - candidate attributes vary within users and can therefore change their impression ordering
expected_delta: [0.0, 0.0009]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0009 over 2 measurement(s), so the promise is capped at the record; was: on pointwise official FM, the exact four-field construction measured fresh-seed mean
  Δ +0.0009 (seed-0 Δ +0.0005, z 3.12); do not attribute more than that confirmed gain
cost: 86 changed lines; 34 s measured versus 15 s parent (~2.3x); numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-exposure-session,
  model-dcn-cross-head]
conflicts_with: [model-din-history-attention, history-ordered-user-tag-affinity]
status: proven — accepted on [official FM]
evidence: [live_07:node_001, live_07:node_023]
---
## Claim
Append categorical matches between the candidate and the user's latest strictly earlier positive video's first
tag, music ID, and video type, plus bucketed time since that positive impression.

## Mechanism (why it moves within-user ranking)
The latest positive impression supplies a short-term interest anchor. Candidate attribute matches and recency differ
across a user's rows, allowing the FM to rank candidates that resemble that anchor above unrelated candidates even
when the exact author has never appeared in the user's history.

## How to implement on node_000
1. Read `tag`, `music_id`, and `video_type`; normalize missing values to `UNK` and retain only the first tag.
2. Add `time_ms` to train, valid, and score-extra loading.
3. Define recency edges at 1 min, 10 min, 1 h, 6 h, 1 d, 3 d, and 7 d.
4. Stable-sort train indices by `(user_id, time_ms, original_row)`.
5. For each equal-time group, emit match values from the user's pre-group latest-positive state.
6. After emitting the group, update that state from positive rows; equal-time rows never observe one another.
7. Encode each attribute match as `none`, `unknown`, `yes`, or `no`, and recency as its fixed bucket.
8. Append the four values to `FIELDS`, vocabulary construction, `raw()`, and `encode()`.
9. For valid and score-extra, use the final train-positive state without scored-split outcomes.
10. Preserve the parent's FM training, early stopping, prediction order, and score-extra path.

## Risks / failure modes
- Updating state before encoding an equal-time group leaks the current row's outcome.
- The implementation keeps only one positive anchor and the first tag, discarding broader or multi-tag interests.
- Valid and extra rows do not advance the positive state because their outcomes are unavailable.
- The confirmed gain belongs to pointwise official FM; changed-stack composition is not guaranteed.

## Measured
_Verdict:_ ACCEPTED 1x (live_07:node_001 on [official FM] Δ +0.0009)
- live_07:node_001 on [official FM]: primary 0.6020, single-seed Δ +0.0005, seed-mean Δ +0.0009 (z 3.12) — ACCEPTED; 86 changed lines
- live_07:node_023 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6039, single-seed Δ -0.0002 — rejected; 88 changed lines
