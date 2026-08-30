---
id: loss-long-duration-matched-bpr
family: ranking-loss
target_component: loss
source: kb/data/facts.md §3 (duration cohorts); kb/literature/losses/1205.2618_bpr.pdf
applies_when:
  - same-user BPR training is already implemented
  - duration_ms is available as a legal show-time feature
  - users have both positive and negative training impressions longer than 180 seconds
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated five-seed FM-BPR ensemble probe changed primary by −0.00002 on seed 0 and
  received no fresh-seed confirmation, so this fixed 10% matched-pair recipe has no attributable positive gain
cost: 16 changed lines; preserves pair count and approximately 1x training cost; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-field-aware-fm-embeddings]
conflicts_with: [loss-duration-cohort-weighted-bpr, loss-listwise-softmax-within-user, loss-lambdarank-pairs]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0000)]
evidence: [live_06:node_011]
---
## Claim
Replace 10% of ordinary same-user BPR samples with positive-negative pairs for which both impressions have
`duration_ms > 180000`, while preserving the total number of pairs per epoch.

## Mechanism (why it moves within-user ranking)
Matching both sides on the weak long-duration cohort cancels coarse duration separation and directs BPR updates
toward ordering long videos against one another. The remaining 90% of ordinary pairs retains broad within-user
ranking supervision.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user`; optionally apply `ensembling-seed-average`.
2. Build `long_mask = duration_ms > 180000` on training rows.
3. Collect long negative indices and compute per-user `long_neg_count`, `long_neg_start`, and `long_neg_sorted`.
4. Define `long_pair_pos` as positive long rows whose user has at least one long negative.
5. Each epoch, sample the ordinary negative array exactly as in BPR and copy `pair_pos` to `train_pos`.
6. Select `len(pair_pos)//10` replacement positions without replacement.
7. Sample replacement positives uniformly from `long_pair_pos`.
8. For each replacement positive, sample a negative from that user's long-negative segment.
9. Train on `train_pos` and the modified negative array with the unchanged BPR step and epoch permutation.

## Risks / failure modes
- Long-only positives are sampled globally, so prolific eligible users may be overrepresented.
- Replacing broad pairs can weaken cross-duration ordering even if the long-duration subgroup improves.
- This card does not claim the gains of BPR or seed averaging already present in the measured parent.
- Empty or very small `long_pair_pos` arrays require a guard before sampling.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0000)
- live_06:node_011 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6039, single-seed Δ -0.0000 — rejected; 16 changed lines
