---
id: features-fine-duration-and-tab-cross
family: features
target_component: features
source: kb/data/facts.md §3–4 (positive rate by duration bucket 0.27–0.38; tab 0 rate 0.04 vs tab 4 0.49); organizer README "unexplored" #6 (context)
applies_when:
  - duration and tab both carry strong marginal signal (facts §3, §4) and the label threshold min(duration, 18 s) makes their interaction non-linear
  - the FM currently sees 10 duration buckets and tab as separate fields; their cross is only reachable through one dot product
expected_delta: [0.0, 0.0001]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0002 over 2 measurement(s), so the promise is capped at the record; was: the organizers measured extra static ID fields flat on FM (README) — this card differs by
  crossing two fields the label mechanism depends on, but it may still be noise; test cheaply, accept only >= 0.002
cost: ~10 lines (more quantile buckets; a crossed categorical tab|bucket field); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0001); official FM x1 (best Δ +0.0001); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0006)]
evidence: [live_02:node_012, live_03:node_004, live_06:node_012]
---
## Claim
Resolve duration more finely (30–50 quantile buckets, plus an explicit "<= 18 s" bit) and add a crossed field
`tab x dur_bucket`, so the model can represent "short video in tab 0" and "long video in tab 4" directly.

## Mechanism (why it moves within-user ranking)
A user's rows differ in tab and length; the label depends on both through the 18 s threshold and the tab's watching
behaviour. A crossed categorical gives each combination its own bias and vector instead of forcing the FM to
express the interaction through one 16-d dot product.

## How to implement on node_000
1. edges = quantiles with n = 30 (or 50) on rows with duration > 0; keep 'UNK0' for zeros (see the flag card).
2. Add a sixth field to FIELDS: `tab_x_dur = f'{tab}|{bucket}'`; add a seventh `short18 = int(dur <= 18000)`.
3. Vocabulary/offsets handle the new fields automatically; nothing else changes.

## Risks / failure modes
- Rare tab x bucket combinations get UNK at valid time — fine (the UNK slot exists), but check the count.
- If flat: this family is dead_under FM + logloss; retest only with a pairwise loss or a DCN head (ADR-0004).

## Measured
_Verdict:_ never accepted in 3 measurements on 3 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0001); official FM x1 (best Δ +0.0001); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0006)
- live_02:node_012 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6033, single-seed Δ +0.0002, seed-mean Δ -0.0001 (t -0.44) — rejected; 10 changed lines
- live_03:node_004 on [official FM]: primary 0.6016, single-seed Δ +0.0001, seed-mean Δ +0.0001 (t 0.3) — rejected; 6 changed lines
- live_06:node_012 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average] (variant: features-fine-duration-and-tab-cross — Append a tab-crossed duration-tail field with levels at 180, 240, 360, and 600 seconds, preserving one shared): primary 0.6033, single-seed Δ -0.0006 — rejected; 10 changed lines
