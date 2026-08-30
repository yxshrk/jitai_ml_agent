---
id: loss-tab4-long-duration-bpr
family: ranking-loss
target_component: loss
source: kb/data/facts.md §3–4; live_06:node_019
applies_when:
  - same-user BPR and per-user positive/negative pair sampling are already implemented
  - duration_ms and tab are available as legal show-time features
  - tab 4 impressions longer than 180 seconds contain both labels for enough training users
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated five-seed FM-BPR probe had fresh-seed mean Δ -0.00002 despite seed-0
  Δ +0.00018, so this fixed 2.5% replacement stream has no attributable positive expected gain
cost: 18 changed lines; unchanged pair count and approximately 1x parent runtime (43 s measured); numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-dcn-cross-head,
  model-field-aware-fm-embeddings]
conflicts_with: [loss-duration-cohort-weighted-bpr, loss-long-duration-matched-bpr,
  loss-same-tab-long-duration-bpr]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0000)]
evidence: [live_06:node_019]
---
## Claim
Replace 2.5% of ordinary same-user BPR samples with same-user positive-negative pairs for which both impressions
are in tab 4 and have duration above 180 seconds.

## Mechanism (why it moves within-user ranking)
Matching both sides on user, tab 4, and the coarse long-duration cohort cancels the dominant user, tab, and cohort
differences. The replacement updates therefore focus on ordering videos within the targeted weak context while
retaining ordinary BPR for 97.5% of samples.

## How to implement on node_000
1. First implement `loss-bpr-pairwise-within-user`; for the measured recipe also use `ensembling-seed-average`.
2. Build `long_mask = duration_ms > 180000` and `tab4_mask = tab == '4'` over training rows.
3. Select negative rows satisfying both masks and group them by encoded user with `bincount`, cumulative starts,
   and a stable user sort.
4. Select positive rows satisfying both masks whose user has at least one eligible negative.
5. Each member epoch, sample its ordinary negative array and copy `pair_pos` into `train_pos`.
6. Draw `len(pair_pos)//40` replacement locations without replacement.
7. Fill those locations with sampled eligible positives and uniformly sampled eligible negatives for their users.
8. Shuffle and train on `train_pos` and the modified negative array; preserve pair count and all stopping logic.

## Risks / failure modes
- The targeted pool may be too small, causing repeated examples and context-specific overfitting.
- Replacing broad pairs can weaken global ordering even if the tab-4 subgroup improves.
- The parent already contains BPR and five-seed rank averaging; only the 2.5% matched replacement is attributable
  to this card.
- Do not form cross-user pairs or include rows outside tab 4 or at/below 180 seconds.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0000)
- live_06:node_019 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6041, single-seed Δ +0.0002, seed-mean Δ -0.0000 (z -0.04) — rejected; 18 changed lines
