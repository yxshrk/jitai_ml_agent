---
id: features-previous-tab-transition-state
family: features
target_component: features
source: kb/data/facts.md §4 (large tab-rate differences) and §10.2 (tab-concentrated exposure sequences);
  first-order Markov context features
applies_when:
  - impressions expose `user_id`, `tab`, and `time_ms`, permitting legal per-user temporal ordering
  - current tab is already modeled but the immediately previous tab and same-tab streak are absent
  - scored splits can initialize each user's transition state from their final strictly earlier training exposure
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0000 over 1 measurement(s), so the promise is capped at the record; was: the originating five-seed FM-BPR probe had seed-0 Δ +0.00003 but fresh-seed mean
  Δ -0.00032 (z -1.30), so this exact two-field construction has no attributable positive expected gain
cost: 69 changed lines; preprocessing plus two small categorical fields; measured runtime 86 s versus 51 s
  for the five-member parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-exposure-session,
  history-same-author-run-features]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0003)]
evidence: [live_07:node_021]
---
## Claim
Append the previous-tab→current-tab transition and capped same-tab streak as categorical fields, allowing otherwise
similar impressions in the same current tab to differ according to the user's immediately preceding interface state.

## Mechanism (why it moves within-user ranking)
The previous tab and streak vary across a user's rows, so their FM weights and interactions can alter within-user
ordering. They distinguish entering a surface from continuing within it without using outcomes; equal-time groups
share the same pre-group state and therefore cannot become one another's history.

## How to implement on node_000
1. Read `time_ms` for train, valid, and score-extra and add `tab_transition` and `tab_streak` to `FIELDS`.
2. Implement `tab_transition_features(rows, ui, tabi, ti, initial=None)` using `lexsort` by user, time, and row.
3. For each equal-time user group, encode every row from the pre-group `(last_tab, streak)` state.
4. Emit transition `"UNK>current"` when no prior tab exists; otherwise emit `"previous>current"`.
5. Emit streak `min(previous_streak + 1, 5)` when tabs match, otherwise 1.
6. Commit once per equal-time group: update once for a common tab; reset to `(None, 0)` for mixed-tab groups.
7. Build train features from empty state and retain its final state.
8. Build valid and score-extra independently from that train-final state; never carry valid state into extra.
9. Append both values in `raw()`, build vocabularies on train, and pass them through unchanged FM training.

## Risks / failure modes
- Updating tied rows sequentially introduces file-order dependence and falsely counts simultaneous rows as a streak.
- Carrying validation-final state into score-extra would couple independent evaluation splits.
- Much of the transition signal may duplicate the strong current-tab field; the originating probe was seed-negative.
- The measured script also contained proven BPR and five-seed averaging from its parent, so none of their gain is
  attributable to these transition fields.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0003)
- live_07:node_021 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6041, single-seed Δ +0.0000, seed-mean Δ -0.0003 (z -1.3) — rejected; 69 changed lines
