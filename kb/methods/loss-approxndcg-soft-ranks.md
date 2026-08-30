---
id: loss-approxndcg-soft-ranks
family: ranking-loss
target_component: loss
source: Qin, Liu, and Li, "A General Approximation Framework for Direct Optimization of Information Retrieval Measures," Information Retrieval 2010; https://doi.org/10.1007/s10791-009-9124-x ([doi.org](https://doi.org/10.1007/s10791-009-9124-x?utm_source=openai))
applies_when:
  - nDCG@5 is exactly half the primary metric, but the BPR champion's improvements remain larger on GAUC than nDCG (journal nodes 003 and 009)
  - evaluation lists are short (Foundations §1: about 7 impressions per valid user), making quadratic soft-rank computation cheap
  - LambdaRank pairs and ListNet failed, but neither directly differentiated an approximation of the scored nDCG@5 formula
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0005 over 1 measurement(s), so the promise is capped at the record; was: direct metric alignment is attractive, but two related ranking losses already failed and no
  new information is added; expect at most an acceptance-scale nDCG improvement
cost: ~80 lines; one sampled listwise pass per epoch; approximately 1.5–2x single-model BPR runtime; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-dcn-cross-head, features-exposure-session]
conflicts_with: [loss-lambdarank-pairs, loss-listwise-softmax-within-user]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)]
evidence: [live_07:node_022]
---
## Claim
Add a weak ApproxNDCG auxiliary loss whose differentiable soft ranks and cutoff discounts directly approximate
each sampled user's binary nDCG@5, while retaining BPR as the stable main objective.

## Mechanism (why it moves within-user ranking)
For each user, replace a row's discrete rank by
`r_i = 1 + sum_{j!=i} sigmoid((s_j-s_i)/tau)` and evaluate a smooth top-five discount at that rank.
The resulting gradient jointly moves the entire user list according to approximate nDCG@5 rather than weighting
independently sampled swaps. User-constant score terms cancel from every score difference.

## How to implement on node_000
1. Apply the existing same-user BPR edit and expose a helper that updates the FM from arbitrary per-row score gradients.
2. Precompute mixed-label users and sample one fixed-width list of eight rows per user each epoch, forcing both labels into each list.
3. Compute the pairwise score-difference tensor and soft ranks with `tau=1.0`.
4. Use binary gains, `1/log2(1+r)`, and a sigmoid gate around rank 5.5; divide by each list's exact IDCG@5.
5. Differentiate the soft-rank expression in numpy and flatten its score gradients into the existing FM scatter update.
6. Run one auxiliary pass after ordinary BPR with initial weight 0.1; early-stop only on official validation primary.
7. Seed list sampling from `--seed`; cap epochs normally under `SMOKE_EPOCHS` and keep inference unchanged.

## Risks / failure modes
- Smooth ranks can have weak gradients once scores separate; temperature below 0.5 can instead make them unstable.
- Sampled eight-row lists only approximate full training-user lists, though they better match evaluation list length.
- A large auxiliary weight can sacrifice GAUC for noisy top-five changes; begin at 0.1 rather than replacing BPR.
- Users with no positives have undefined training NDCG and should not enter the auxiliary pass.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)
- live_07:node_022 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6036, single-seed Δ -0.0005 — rejected; 58 changed lines
