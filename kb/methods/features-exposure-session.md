---
id: features-exposure-session
family: features
target_component: features
source: kb/data/facts.md §10 and eda_report.md (label-free exposure statistics measured by the review session, 2026-08-30); position-bias literature — Joachims et al. 2017 "Unbiased LTR with biased feedback" (WSDM) for why session position carries signal; KuaiRand paper (Gao et al. 2022, CIKM) §3 for the logging setup
applies_when:
  - impressions carry `time_ms`, so each user's exposures can be ordered and grouped into sessions (gap > 30 min = new session)
  - the model has no session or exposure-sequence feature (node_000 has none; the five fields are all row-static)
  - the feature must vary WITHIN a user's impression set — session position and density do (a user's ~5 valid rows span several sessions)
  - measured, train, tab 1 only (label-free): P(long_view) by position in session 1st 0.418 · 2nd–3rd 0.387 · 4th–10th 0.333 · 11th–30th 0.243 · >30th 0.137; by impressions in the previous 10 min 0 → 0.413 · 1–3 → 0.350 · 4–10 → 0.207 · >10 → 0.120; gap since previous impression < 0.5 min 0.361 vs 10–60 min 0.399 vs > 60 min 0.417
expected_delta: [0.0, 0.0009]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0005 over 3 measurement(s), so the promise is capped at the record; was: the largest label-free effect measured on this data (0.42 → 0.12 within tab 1) and it varies within users; but the FM can only use it as bucketed ids, and the related same-author / repeat-exposure features measured −0.0005 (live_04:node_005) — expect the upper half only when a tree model (model-lightgbm-lambdarank) consumes it raw
cost: ~50 lines (sort by user and time over train + the scored split's feature rows, running counters, 3–4 bucketed categorical fields); runtime 1.1x; numpy only (pandas optional)
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-lightgbm-lambdarank, model-din-history-attention, history-same-author-run-features, history-repeat-exposure-fatigue]
conflicts_with: []
status: proven — accepted on [official FM + loss-bpr-pairwise-within-user]
evidence: [live_07:node_002, live_07:node_010, live_07:node_013]
---
## Claim
Where an impression sits in the user's session — first after a break, or the 15th in a fast scroll — predicts
long-view strongly (0.42 vs 0.14 in tab 1), is computable from exposure timestamps alone, and is absent from the
model. It is the one measured signal that both varies within a user's list and survives conditioning on tab.

## Mechanism (why it moves within-user ranking)
A user who has just opened the app watches; a user ten impressions into a scroll skips. The label is a watch-time
threshold, so attention budget per impression is the mechanism. Within a user's evaluation rows, sessions differ, so
the feature reorders the list; user-constant features cannot. Legal: built from `time_ms` of strictly earlier rows —
train rows for train, train + earlier valid rows for valid, train + earlier test rows for the extra file — never from
labels (CONTRACT: history features from earlier rows' features).

## How to implement on node_000
1. Load `user_id, time_ms` for train and for the split being scored; concatenate, stable-sort by (user, time_ms);
   remember each row's origin so features are written back in file order.
2. One pass per user with running state: `pos` (position in session; reset when gap > 30 min), `n10` (impressions in
   the previous 10 min, a deque), `gap` (ms since the previous impression, ∞ for the first).
3. Bucketise into categorical fields: pos ∈ {1, 2–3, 4–10, 11–30, >30}; n10 ∈ {0, 1–3, 4–10, >10}; gap ∈
   {<0.5, 0.5–2, 2–10, 10–60, >60 min, none}. Append to FIELDS as three new fields (they get W and V like any id).
4. Compute the same fields for the `--score-extra` file from train + that file's own feature rows (no labels needed).
5. Keep everything else unchanged; with `--score-extra`, the running state must be rebuilt per split, never carried
   from valid into test.
6. Variant for tree models: pass the raw integers (pos, n10, gap_ms) instead of buckets.

## Risks / failure modes
- Same-day valid rows with identical `time_ms` (facts §8: repeated pairs) — process ties together (same position).
- The effect is partly "tab 1 vs profile tabs"; the within-tab numbers above are the honest size. Do not expect the
  full 0.42 → 0.12 range to convert; users with one session have no within-list variation.
- Volume drift (facts §5): early train days have dense sessions; bucket edges chosen on train may shift on test.
- A leak looks like a win: never let a row's own or a later row's timestamp into its counters.

## Measured
_Verdict:_ ACCEPTED 1x (live_07:node_010 on [official FM + loss-bpr-pairwise-within-user] Δ +0.0009)
- live_07:node_002 on [official FM]: primary 0.6014, single-seed Δ -0.0001 — rejected; 65 changed lines
- live_07:node_010 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6034, single-seed Δ +0.0003, seed-mean Δ +0.0009 (z 3.14) — ACCEPTED; 65 changed lines
- live_07:node_013 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6046, single-seed Δ +0.0005, seed-mean Δ +0.0003 (z 0.97) — rejected; 67 changed lines
