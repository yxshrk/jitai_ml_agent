---
id: loss-duration-cohort-weighted-bpr
family: ranking-loss
target_component: loss
source: kb/data/facts.md §3; live_05:node_011
applies_when:
  - same-user logistic BPR is already implemented
  - duration_ms is available as a legal show-time feature
  - short (<18 s) and very long (>180 s) videos are weak ranking cohorts
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0020 over 1 measurement(s), so the promise is capped at the record; was: the isolated single-seed probe lost 0.00197 primary on FM+BPR and had no seed confirmation;
  therefore this fixed 25%-of-cohort auxiliary sampling recipe has no attributable positive gain
cost: ~16 changed lines; adds at most 25% of eligible cohort-positive pairs per epoch; measured runtime 13 s; numpy only
composes_with: [loss-bpr-pairwise-within-user, model-field-aware-fm-embeddings, regularization-embedding-dropout-l2]
conflicts_with: [loss-listwise-softmax-within-user, loss-lambdarank-pairs, loss-ranksvm-margin-pairs]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0020)]
evidence: [live_05:node_011]
---
## Claim
Add an auxiliary BPR stream sampled from positive rows whose duration is below 18 seconds or above 180 seconds,
pairing them with ordinary uniform same-user negatives to upweight these duration cohorts.

## Mechanism (why it moves within-user ranking)
The extra pairs increase gradient frequency for positives in two weak duration cohorts while preserving same-user
differencing. Unlike context-matched negative sampling, the negative remains uniformly drawn from all negatives
of that user; only the positive-row sampling distribution changes.

## How to implement on node_000
1. Apply `loss-bpr-pairwise-within-user` and retain its positive rows and per-user negative pools.
2. Parse training `duration_ms` and select eligible positive rows with duration `<18000` or `>180000`.
3. Each epoch, sample without replacement `len(eligible)//4` eligible positives using the seeded RNG.
4. Sample one negative for each auxiliary positive from its existing uniform same-user negative pool.
5. Concatenate auxiliary and ordinary positive/negative row arrays, permute once, and call the unchanged BPR step.
6. Preserve validation-primary early stopping, checkpointing, inference, and score-extra behavior.

## Risks / failure modes
- The measured recipe substantially degraded both GAUC and nDCG@5; cohort upweighting can distort the global ranking.
- The underlying BPR gain belongs to `loss-bpr-pairwise-within-user`; this card can claim only the incremental stream.
- “25%” means one quarter of eligible cohort positives, not one quarter of all ordinary BPR pairs.
- Duration-zero positive rows do not exist, but the `<18 s` predicate technically includes that duration range.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0020)
- live_05:node_011 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6017, single-seed Δ -0.0020 — rejected; 16 changed lines
