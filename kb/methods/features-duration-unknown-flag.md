---
id: features-duration-unknown-flag
family: features
target_component: features
source: kb/data/facts.md §3 (duration_ms = 0 rows are always long_view = 0; 1.9 % of rows); label_check.py
applies_when:
  - duration_ms = 0 rows exist in valid/test features (they do: same log, same column)
  - the current encoding sends duration 0 into the shortest bucket together with genuinely short videos (node_000: searchsorted on quantile edges)
expected_delta: [0.001, 0.004]
expected_delta_basis: mechanism-backed but small — it only reorders rows for users who were shown an unknown-length
  video; those rows are always negative, so pushing them down can only help nDCG for such users
cost: 2 lines in raw(); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, loss-watchtime-censored, data-weighting-recency, aux-targets-is-click, history-user-aggregates, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM x2 (best Δ -0.0003); official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0002); official FM + field-aware FM embeddings x1 (best Δ -0.0002)]
evidence: [live_01:node_003, live_02:node_003, live_02:node_007, live_04:node_018]
---
## Claim
Give unknown-duration rows their own categorical value so the model can learn they are never long views, instead of
mixing them into the "shortest videos" bucket whose positive rate is 0.28.

## Mechanism (why it moves within-user ranking)
Within a user's list, a duration-0 row currently gets the bucket-0 embedding and can rank above genuine
candidates; a dedicated value lets its bias and interactions go strongly negative. This is pure information the
model is currently denied.

## How to implement on node_000
1. In `raw()`: `bucket = 'UNK0' if float(dur) == 0 else str(int(np.searchsorted(edges, float(dur))))`.
2. Compute the quantile edges on rows with duration > 0 only, so bucket 0 is not distorted by the zeros.
3. Nothing else changes; the vocabulary picks the new value up automatically.

## Risks / failure modes
- Tiny effect on GAUC (those users are few); confirm with the grey-zone multi-seed rule rather than discard.
- Combine with features-fine-duration-and-tab-cross in a later node; keep this one isolated to measure it.

## Measured
_Verdict:_ never accepted in 4 measurements on 3 stack(s); official FM x2 (best Δ -0.0003); official FM + loss-bpr-pairwise-within-user x1 (best Δ +0.0002); official FM + field-aware FM embeddings x1 (best Δ -0.0002)
- live_01:node_003 on [official FM]: primary 0.6012, single-seed Δ -0.0003 — rejected; 275 changed lines
- live_02:node_003 on [official FM]: primary 0.6012, single-seed Δ -0.0003 — rejected; 4 changed lines
- live_02:node_007 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6031, single-seed Δ +0.0000, seed-mean Δ +0.0002 (t 1.11) — rejected; 4 changed lines
- live_04:node_018 on [official FM + field-aware FM embeddings]: primary 0.6028, single-seed Δ -0.0002 — rejected; 5 changed lines
