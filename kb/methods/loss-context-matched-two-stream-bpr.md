---
id: loss-context-matched-two-stream-bpr
family: ranking-loss
target_component: loss
source: kb/data/facts.md §4 (tab positive rates 0.04–0.49); kb/literature/losses/1205.2618_bpr.pdf
applies_when:
  - same-user logistic BPR is already implemented
  - tab varies within users and strongly separates labels
  - training users have positive and negative rows within the same tab
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0003 over 1 measurement(s), so the promise is capped at the record; was: the isolated additive stream lost 0.00034 primary on one seed over standard FM+BPR;
  no positive seed-mean evidence exists, so no attributable gain is claimed
cost: 16 changed lines; runtime remained ~1x (16 s measured); numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, model-dcn-cross-head, regularization-embedding-dropout-l2]
conflicts_with: [loss-bpr-hard-negatives, loss-lambdarank-pairs, loss-listwise-softmax-within-user, loss-ranksvm-margin-pairs]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0003)]
evidence: [live_05:node_006, ceiling:oracle]
---
## Claim
Retain one ordinary same-user BPR pair per eligible positive and add a second pair for positives having a
same-user, same-tab negative, directing extra updates toward ordering items after the dominant tab prior cancels.

## Mechanism (why it moves within-user ranking)
Both rows in the added pair share user and tab, so their additive user/tab terms cancel. The gradient must instead
use row-varying video, author, duration, and their interactions to separate the positive from the negative.
Eligible positives receive both an ordinary pair and a context-matched pair, so this is also an implicit reweighting.

## How to implement on node_000
1. First implement `loss-bpr-pairwise-within-user`, including its ordinary negative pools.
2. Define `user_tab_keys = train_users * dims[3] + (Xtr[:,3] - off[3])`.
3. Stable-sort negative rows by this key; build `negative_tab_count` and cumulative `negative_tab_start`.
4. Keep positives whose user-tab key has a negative and cache their matching keys.
5. Concatenate ordinary positives with these matching positives as `two_stream_positive_rows`.
6. Each epoch, sample one negative from each matching user-tab segment.
7. Concatenate ordinary and same-tab negatives, jointly permute both streams, and call the unchanged BPR step.

## Risks / failure modes
- The measured implementation reduced primary by 0.00034 and peaked early at epoch 3 before declining.
- Extra pairs overweight positives from tabs containing both classes rather than isolating only context matching.
- The underlying logistic BPR mechanism was already present and proven; only the additive same-tab stream belongs
  to this card.
- Matching on a strong context can remove useful easy comparisons and amplify noisy within-tab labels.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0003)
- live_05:node_006 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6033, single-seed Δ -0.0003 — rejected; 16 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0003 for the signal family 'same-tab-pairs' — facts §11 §3: same-tab BPR negatives at 30 / 70 / 100 % score 0.6030 / 0.6024 / 0.5880; cross-tab pairs are already solved (error 0.186) (facts §11, kb/data/screens/CEILING.md)
