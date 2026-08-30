---
id: history-same-author-run-features
family: history
target_component: history
source: kb/data/facts.md §10.2 (same-author consecutive exposure is a label-free negative signal; run length separates profile browsing from series); measured by the review session, 2026-08-30
applies_when:
  - impressions carry `time_ms`, so a user's exposures can be ordered and runs of consecutive same-author exposures counted (facts §10.2)
  - the model has no exposure-sequence feature (node_000 has none)
  - `tab` is already a field, so the feature must add WITHIN-tab information (facts §10.2: tab 1 0.268 vs 0.389; tab 4 0.393 vs 0.501)
expected_delta: [0.0, 0.0002]
expected_delta_basis: bounded (ADR-0018) at +0.0002 by the oracle for 'taste-train-history' — facts §11 row 'train-history taste, item-kNN, repeats, weekday/hour': <= +0.0002 each; user × author / music taste is +0.0021 / +0.0016 only with same-week labels at 3 % coverage; was: affects 2.7 % of valid rows; large within-tab lift where it applies but the FM already has tab; the related exposure-count features measured −0.0005 on one seed (live_04:node_005)
cost: ~40 lines (sort by user and time, run-so-far encoding, one or two categorical fields); runtime 1.1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, history-repeat-exposure-fatigue]
conflicts_with: []
status: untried
evidence: [ceiling:oracle]
---
## Claim
A user shown the same author twice in a row is far less likely to long-view the second exposure (0.142 vs 0.337
base), and the length of the same-author run so far tells profile browsing (long runs) from series watching (short
runs) — a label-free, legally computable signal the FM does not have.

## Mechanism (why it moves within-user ranking)
Within a user's impression set the run-so-far length varies from row to row, so it can reorder that user's list —
unlike user-constant features. The effect survives conditioning on tab (facts §10.2), so it is not just the
profile-tab prior the FM already encodes; a small categorical field lets the FM learn its own weight per level and
its interactions with tab and duration.

## How to implement on node_000
1. In the data loading, order each user's rows of EVERY split by `time_ms` (keep `row_id` for output order).
2. For each row compute `run_so_far` = number of consecutive immediately-preceding exposures of the same user with
   the same author (0 = new author), using only rows strictly earlier in time — earlier valid rows are allowed,
   labels never; cap at 5 and encode `run_so_far` as a new categorical field (7 fields).
3. Optionally add `prev_same_author × tab` as a second field; keep everything else of the parent unchanged.
4. Restore the original row order before writing predictions.

## Risks / failure modes
- Position-in-run (first / middle / last) needs later rows, which are not available at inference — only run-so-far is legal.
- Most of the raw effect is tab; the within-tab residual may be inside seed noise once BPR is present.
- Rows at a user's first exposure of the window have no history; encode as level 0, not as missing.

## Measured
_Verdict:_ no measurement yet
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0010 for the signal family 'session-context' — facts §11.1: pairs within 10 min are 2 % of the GAUC pair mass; measured +0.0009 on BPR (z 3.1), +0.0002 on the seed blend; attribute continuation from the most recent earlier positive (history-last-positive-attribute-recurrence) +0.0009 on the plain FM (live_07 node_001), additive 0 on node_009 (kb/data/screens/BEHAVIOUR.md) (facts §11, kb/data/screens/CEILING.md)
