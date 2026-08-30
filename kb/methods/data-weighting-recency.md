---
id: data-weighting-recency
family: data-weighting
target_component: data-weighting
source: kb/data/facts.md §5 (volume 280 K/day -> 20 K/day; positive rate 0.34 -> 0.29; FM valid->test drop −0.007); organizer README "unexplored" #6 (drift)
applies_when:
  - training data drifts toward the evaluation period (facts §5: the last four train days resemble valid/test)
  - early days dominate the row count (04-10/04-11 hold 44 % of train rows)
expected_delta: [0.001, 0.006]
expected_delta_basis: mechanism-backed by measured drift; the organizers' own valid->test gap (−0.007) shows the
  distribution moves; a re-weighting cannot add information, only re-balance it, so cap expectations at 0.006
cost: ~8 lines (per-row weight from date; weighted gradient); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, loss-watchtime-censored, features-duration-unknown-flag, features-fine-duration-and-tab-cross, aux-targets-is-click, history-user-aggregates, model-dcn-cross-head]
conflicts_with: []
status: dead_under {run: live_01, stack: official FM + loss-bpr-pairwise-within-user, delta: +0.0003}
evidence: [live_01:node_002, live_01:node_004]
---
## Claim
Weight each training row by exp(−age / tau) (half-life 3 / 7 / 14 days from 2022-04-21) so the model fits the recent
regime that validation and test come from, instead of the early high-volume days.

## Mechanism (why it moves within-user ranking)
The loss becomes sum_i w_i * loss_i: gradients from April 10 rows shrink to ~0.37 of an April 20 row at a 7-day
half-life. Popularity and user tastes measured on recent rows transfer better to the next weeks; the FM's
user x video embeddings are otherwise dominated by early traffic.

## How to implement on node_000
1. Read `date` for train rows; age = days between the row's date and 20220421.
2. w = 0.5 ** (age / half_life); normalise so mean(w) = 1.
3. In `FM.step`, replace `g = (sigmoid(z) − y) / B` by `g = w_batch * (sigmoid(z) − y) / w_batch.sum()`.
4. Sweep half_life in {3, 7, 14}; also try the hard variant "train on dates >= 20220415 only".

## Risks / failure modes
- Short half-lives throw away most rows — watch the learning curve for higher variance / earlier peak.
- Interacts with early stopping: the effective dataset is smaller, so the peak epoch shifts.

## Measured
- live_01:node_002 on [official FM]: primary 0.6019, single-seed Δ +0.0005 — rejected; 452 changed lines
- live_01:node_004 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6039, single-seed Δ +0.0003 — rejected; 127 changed lines
