---
id: history-user-aggregates
family: history
target_component: history
source: kb/literature/models/1706.06978_din.pdf (user-behaviour interest; GAUC); kb/data/facts.md §2; organizer README "unexplored" #2
applies_when:
  - users have a train history to aggregate (facts §2: median 35 train rows, p10 = 6) — enough for rates, too few for attention models
  - the catalogue is closed (facts §1), so per-user rates by author / tab / duration bucket are well defined on train
expected_delta: [0.0, 0.0010]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0007 over 6 measurement(s), so the promise is capped at the record; was: organizers' lead #2 (history is entirely unused by the baseline); aggregates capture the
  first-order part of what DIN/SIM learn; histories are short, so cap expectations at 0.006
cost: ~90 lines (time-ordered running counts on train; smoothed rates; bucketised fields); runtime ~1.5x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, features-duration-unknown-flag, data-weighting-recency, model-dcn-cross-head]
conflicts_with: []
status: proven — accepted on [official FM]
evidence: [live_01:node_006, live_02:node_008, live_04:node_003, live_04:node_011, live_05:node_004, live_06:node_004, ceiling:oracle]
---
## Claim
Add per-user historical rates — the user's long_view rate for this author, this tab, this duration bucket, computed
from rows strictly earlier in time — as extra (bucketised) fields.

## Mechanism (why it moves within-user ranking)
The rates vary *across a user's rows* (different authors / tabs / lengths), so they can reorder the list; they
encode the personal preference that the user x author embedding dot product has to learn from scratch, and help
most for users with few rows. DIN's premise — a user's past behaviour toward similar items predicts the next
action — in its simplest, leakage-safe form.

## How to implement on node_000
1. Sort train rows by time_ms. For each key (user, author), (user, tab), (user, dur_bucket): running counts of
   rows and positives *before* the current row (cumsum shifted by one within groups — vectorise with sort + cumsum).
2. Smoothed rate = (pos + a x prior) / (n + a), prior = the user's overall rate so far, a = 5; bucketise into 10
   quantile bins plus an "no history" value.
3. For valid rows, use the totals over all train rows (every train row precedes valid).
4. Add the three bucketised rates as fields; nothing else changes.

## Risks / failure modes
- Train/valid mismatch: early train rows have almost no history while valid rows see the full train window —
  the "no history" bucket must exist, and recency weighting (data-weighting-recency) reduces the mismatch.
- Using a row's own label in its rate is target leakage — the shift-by-one is mandatory (Critic checks this).

## Measured
_Verdict:_ ACCEPTED 1x (live_04:node_003 on [official FM] Δ +0.0010)
- live_01:node_006 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6023, single-seed Δ -0.0014 — rejected; 792 changed lines
- live_02:node_008 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6030, single-seed Δ -0.0001 — rejected; 70 changed lines
- live_04:node_003 on [official FM]: primary 0.6021, single-seed Δ +0.0007, seed-mean Δ +0.0010 (t 4.83) — ACCEPTED; 128 changed lines
- live_04:node_011 on [official FM + field-aware FM embeddings]: primary 0.6017, single-seed Δ -0.0014 — rejected; 69 changed lines
- live_05:node_004 on [official FM]: primary 0.6019, single-seed Δ +0.0004, seed-mean Δ +0.0006 (z 1.89) — rejected; 87 changed lines
- live_06:node_004 on [official FM]: primary 0.6019, single-seed Δ +0.0004, seed-mean Δ +0.0008 (z 1.85) — rejected; 75 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0002 for the signal family 'taste-train-history' — facts §11 row 'train-history taste, item-kNN, repeats, weekday/hour': <= +0.0002 each; user × author / music taste is +0.0021 / +0.0016 only with same-week labels at 3 % coverage (facts §11, kb/data/screens/CEILING.md)
