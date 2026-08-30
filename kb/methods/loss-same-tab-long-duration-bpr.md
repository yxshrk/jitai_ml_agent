---
id: loss-same-tab-long-duration-bpr
family: ranking-loss
target_component: loss
source: kb/data/facts.md §3–4 (duration and tab effects); kb/literature/losses/1205.2618_bpr.pdf
applies_when:
  - same-user BPR is already implemented
  - duration_ms and tab are available as legal show-time features
  - enough users have both positive and negative >180 s impressions within tab 1 or tab 4
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated five-seed FM-BPR probe lost 0.00019 primary and received no fresh-seed
  confirmation, so this fixed 5% replacement recipe has no attributable positive gain
cost: 19 changed lines; unchanged pair count; measured total runtime 59 s on a five-member ensemble; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-field-aware-fm-embeddings]
conflicts_with: [loss-long-duration-matched-bpr, loss-duration-cohort-weighted-bpr, loss-context-matched-two-stream-bpr]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0002)]
evidence: [live_06:node_016]
---
## Claim
Replace 5% of ordinary same-user BPR pairs with same-user, same-tab positive-negative pairs where both impressions
exceed 180 seconds and the shared tab is 1 or 4.

## Mechanism (why it moves within-user ranking)
Matching both duration cohort and tab cancels easy cross-tab and coarse-duration comparisons, directing a small
fraction of updates toward within-context ordering among long videos while preserving 95% of the broad BPR sample.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; retain its ordinary pair arrays and sampler.
2. Build `long_mask = duration_ms > 180000` and `tab14_mask = tab in {'1','4'}`.
3. Encode `(user, tab)` as `long_tab_key = user_index * 2 + (tab == '4')`.
4. Precompute sorted negative indices, counts, and starts by `long_tab_key`, restricted to long tab-1/4 rows.
5. Keep positive long rows only when their key has at least one eligible negative.
6. Each epoch, copy `pair_pos` and choose `len(pair_pos)//20` replacement positions without replacement.
7. Sample replacement positives from the eligible long-positive pool and negatives from the matching key.
8. Train on the resulting arrays with the original permutation, batch size, BPR step, and pair count.

## Risks / failure modes
- This is an auxiliary sampler on top of proven BPR; neither BPR nor seed averaging is attributable to this card.
- The fixed 5% stream oversamples a narrow cohort and can reduce broad GAUC even if tab-4 diagnostics improve.
- Sparse eligible keys cause repeated sampling of the same noisy pairs; empty pools must fall back to ordinary BPR.
- It overlaps with the broader long-duration matched stream and should not be enabled simultaneously.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0002)
- live_06:node_016 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6038, single-seed Δ -0.0002 — rejected; 19 changed lines
