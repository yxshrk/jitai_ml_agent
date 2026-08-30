# Information ceiling of the within-user ranking task — where the champion's error is, and what could still move it

Measured 2026-08-31 (research session) on valid against live_07 node_009 (BPR + 5-seed rank average; valid primary
0.6041, GAUC 0.6709). Scripts: `pairs.py` (pair-mass attribution), `oracles.py` (oracle bounds per signal family),
`knn.py` (collaborative signals), `bpr_sametab_mix.py` (same-tab negative sampler); shared loader `_common.py`.
Run from the repo root with `.venv/bin/python kb/data/screens/<script>.py`. Single-seed numbers unless stated;
the seed SD of a valid primary is ≈ 0.00035, so anything under ±0.0005 is zero.

## 1. Where the error is: pair-mass attribution (what GAUC actually counts)
GAUC = 1 − (weighted misordered positive–negative pairs). 210,482 pairs in valid, each user's pairs weighted so the
user counts its number of positives (the scorer's weighting). Weighted error of the champion: **0.329**.

| pair type | share of mass | error | contribution |
|---|---|---|---|
| both impressions tab 1 | **0.686** | 0.379 | **0.260** |
| any same tab | 0.735 | 0.381 | 0.280 |
| different tabs | 0.265 | 0.186 | 0.049 |
| positive tab 0, negative tab 1 | 0.005 | 0.985 | 0.005 (unwinnable: tab 0 positive rate 0.04) |
| different dates | 0.764 | 0.327 | 0.250 |
| more than a day apart | 0.618 | 0.326 | 0.202 |
| same date | 0.236 | 0.336 | 0.079 |
| within 10 minutes | **0.020** | 0.342 | 0.007 |
| positive is the shorter video | 0.513 | 0.297 | 0.153 |
| positive is the longer video | 0.485 | 0.364 | 0.176 |
| negative has duration 0 | 0.020 | 0.024 | 0.000 (already at the bottom) |
| same video / same author | 0.001 / 0.001 | — | ≈ 0 |

Reading: **two thirds of everything the metric still penalises is "two feed (tab 1) videos shown to the same user on
different days — which one did they long-view?"** Cross-tab ordering is essentially solved. Only 2 % of the pair mass
is within a session, which is why `features-exposure-session` can be worth +0.0009 and no more. The duration
calibration is right (champion's within-user rank percentile tracks P(long_view) by duration decile: 0.55 → 0.65 →
0.51 vs 0.26 → 0.36 → 0.27), so the "positive is longer" asymmetry is not a fixable bias.

## 2. Oracle bounds: signal families that cannot be rescued
Each family is measured with information a model could never legally have, blended additively on the champion at
its best weight (`oracles.py`). If the oracle adds nothing, no feature engineering in that family will.

| family | oracle | adds |
|---|---|---|
| video-side state in the valid week | leave-one-out video rate **from the valid labels themselves** (also × tab, author) | **+0.0003** |
| video-side, any period | the leaky whole-period `video_features_statistic` file (long_time_play_cnt / show_cnt) | **+0.0000** |
| user "mood" that day | user × date rate from the *other half* of the user's own valid rows (54 % coverage) | **+0.0000** |
| user × tab / duration / tag / video_type taste, same week | other-half rates | +0.0003 / −0.0004 / +0.0002 / 0 |
| user × author / user × music, same week | other-half rates, **3.3 % / 3.2 % coverage** | **+0.0021 / +0.0016** |
| random-exposure log (same dates as valid; 288 K rows, 78 % of valid rows have the user in it) | user-week, user-day, video-week rates | 0 / +0.0001 / 0 |
| train-history taste (user × tab/dur/tag/author/music/type rates) | — | ≤ +0.0001 each |
| collaborative signal beyond the FM | item–item kNN over train positives, negatives, watch fraction, co-exposure | 0 each |
| train → valid repeats | 1.6 % of rows; seen-positive P = 0.47, demoting/promoting them | +0.00002 |
| within-valid re-exposure (> 60 s later) | 827 rows, P = 0.20 vs 0.31; demoting them | 0 |
| day-level label-free context | rows that day / before / day index / first of day / same-hour count | ≤ +0.0002 |
| habits | user × weekday, user × hour, global weekday/hour | 0 |
| within-user time | −time_ms, −date, hourmin | 0 |

Conclusions: **item-side information is saturated** — even the valid week's own labels and the leaky month-long
statistics add +0.0003 or less, so every "video rate / drift / popularity / content" card is bounded there. **Day-level
user state does not exist** (knowing the user's other outcomes that day adds nothing). **The random log is worth
nothing** on this task (its exposures are random, positive rate 0.085, and the user-day signal it carries is nil) —
the rules question is moot. Taste exists at the **author / music** level (+0.002 where the same author recurs in the
week) but recurs in 3 % of rows and is invisible without labels.

## 3. Model-side ablations (single seed, the BPR script)
- FM without the `user_id` field: 0.5932 vs 0.6015 — **personalisation is worth +0.008**, and the champion's GAUC is
  flat across users with 1 or 800 training rows (0.665–0.676; users with no training rows 0.686), so the user field
  mostly learns user × tab and user × duration, not taste. On tab-1-only sublists the champion reaches GAUC 0.621
  against 0.603 for the train video rate alone.
- Adjacent-period data: train + the even half of valid, scored on the odd half: 0.5821 vs 0.5826 without. **Recency
  is not a lever; volume is** — training from 04-12 (51 % of rows) 0.6004, from 04-15 (22 %) 0.5949, full 0.6031.
  A train+valid refit at designation is worth ≤ +0.001 on test.
- Same-tab negatives in BPR (the 69 % of the mass): 100 % same-tab 0.5880 (loses the tab main effect), 30 % 0.6030,
  70 % 0.6024 — flat. Hard negatives and 5 % matched pairs were already dead.
- Seeds are saturated: 20 members (four 5-seed champions) 0.6044 vs 0.6041–0.6046 for each; adding the BPR+session
  lineage (node_010 × 4) 0.6047.
- Duplicate rows: 77 % of repeated (user, video) pairs in valid are at the same `time_ms` with the same label
  (P = 0.985 / 0.003) — copies of one event, 2.2 % of rows; harmless (same label → no scored pair). A genuine
  re-exposure more than an hour after a long-view is never long-viewed again (n = 45), after a non-long-view 0.25–0.28.

## 4. What this means for the agent
1. **The ceiling on valid with legal features is ≈ 0.605–0.607.** Four runs stopped at 0.604 because the data
   stops there, not the search. 0.62 on valid is not reachable from this file: the residual 0.33 pair error is the
   user's momentary interest in a specific feed video on a specific day, and nothing observable at show time — not
   even the leaky sources — predicts it.
2. **Stop spending slots on** item-side features (rates, drift, content, statistics), recency weighting, day/mood
   state, history aggregates and kNN/CF add-ons, same-tab pair sampling, and more seeds. Their bounds are above.
   Fold this section into `facts.md` §11 after live_07 ends (facts is in the cached prompt prefix) and mark the
   cards `bounded ≤ +0.0003 by oracle`.
3. **What is still open, in order of expected value on the judged (hidden-test) score:** (a) the bonus datasets —
   KuaiRand-1K / 27K earn explicit extra credit and have far longer per-user histories, where the author-level taste
   that is invisible here (§2) becomes observable; (b) a train+valid refit plus a two-lineage rank blend at
   designation (≤ +0.001 together); (c) the only untested modelling lever for the tab-1 cross-day pairs is a
   candidate-conditioned history model (`model-din-history-attention`) — its bound is the kNN result (0), so give it
   one slot, not a generation.
4. **Harness:** add the pair-mass attribution (§1) to the referee's per-node breakdown — "which pair types moved" is
   a sharper diagnostic than per-group GAUC (a group can move on label noise; the pair table says whether the node
   touched the 69 % that matters). Give the Selector each card's family bound from §2 so bounded families are
   deprioritised, and let the run declare convergence when the champion is within 0.002 of the bounded ceiling
   instead of waiting three flat generations (~$7 per run).
