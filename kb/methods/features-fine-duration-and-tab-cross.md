---
id: features-fine-duration-and-tab-cross
family: features
target_component: features
source: kb/data/facts.md §3–4 (positive rate by duration bucket 0.27–0.38; tab 0 rate 0.04 vs tab 4 0.49); organizer README "unexplored" #6 (context)
applies_when:
  - duration and tab both carry strong marginal signal (facts §3, §4) and the label threshold min(duration, 18 s) makes their interaction non-linear
  - the FM currently sees 10 duration buckets and tab as separate fields; their cross is only reachable through one dot product
expected_delta: [0.000, 0.004]
expected_delta_basis: the organizers measured extra static ID fields flat on FM (README) — this card differs by
  crossing two fields the label mechanism depends on, but it may still be noise; test cheaply, accept only >= 0.002
cost: ~10 lines (more quantile buckets; a crossed categorical tab|bucket field); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head]
conflicts_with: []
status: untried
evidence: []
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
(none yet)
