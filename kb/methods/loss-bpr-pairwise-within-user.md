---
id: loss-bpr-pairwise-within-user
family: ranking-loss
target_component: loss
source: kb/literature/losses/1205.2618_bpr.pdf §3–4 (BPR-Opt, LearnBPR); organizer README "unexplored" #1
applies_when:
  - the metric is within-user ranking (task.md) — GAUC is literally the fraction of correctly ordered (pos, neg) pairs
  - most training users have both positives and negatives (facts §7: 92.7 % of train users are discriminative)
  - the model produces a per-row score that can be differenced (any FM/DCN head)
expected_delta: [0.0, 0.0022]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0022 over 13 measurement(s), so the promise is capped at the record; was: organizers' lead #1; aligns the objective with the scored metric; the pointwise FM already
  captures the user x video signal, so the gain is in ordering not calibration — do not expect more than 0.01
cost: ~60 lines in FM.step + a pair sampler; runtime ~1x (pairs ~ number of positives per epoch); numpy only
composes_with: [features-duration-unknown-flag, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head]
conflicts_with: [loss-listwise-softmax-within-user, loss-lambdarank-pairs]
status: proven — accepted on [official FM], [official FM + loss-bpr-pairwise-within-user]
evidence: [live_01:node_001, live_02:node_001, live_03:node_001, live_04:node_002, live_04:node_006, live_05:node_002, live_06:node_002, live_07:node_003, live_06:node_011, live_06:node_016, live_06:node_019, live_08:node_001, live_08:node_007]
---
## Claim
Training on within-user (positive, negative) pairs with loss −log σ(s_pos − s_neg) optimises the pairwise ordering
that GAUC measures, instead of per-row probabilities that only matter through their order.

## Mechanism (why it moves within-user ranking)
Logloss spends capacity on calibration across users (a user-constant bias term is a free win for logloss but worth
exactly zero to the metric, task.md). BPR's gradient is the *difference* of two rows of the same user, so every
user-constant term cancels and all capacity goes to the ordering the metric scores. AUC = P(s_pos > s_neg) within a
user; BPR maximises a smooth lower bound of exactly that quantity (paper §3.1).

## How to implement on node_000
1. After encoding, group train row indices by user; keep users with at least one positive and one negative.
2. Each epoch, build pairs: for every positive row of a user, sample one negative row of the same user (uniform).
   ~380 K pairs per epoch (facts: 1.14 M rows, positive rate 0.337, most users mixed).
3. In `FM.step`, compute logits for the positive batch and the negative batch, `d = z_pos − z_neg`,
   `g = −σ(−d)` (d(−log σ(d))/dd); accumulate gradients as the existing code does, once with +g on the positive
   rows' features and once with −g on the negative rows' features (the existing `np.add.at` pattern).
4. Keep early stopping on validation primary unchanged; keep the logits as the prediction score.
5. Hybrid variant (safer first try): loss = 0.5·BPR + 0.5·logloss on the same batch — keeps the pointwise signal
   for users with a single class in the batch.

## Risks / failure modes
- Pair sampling in pure Python per epoch is slow — vectorise: precompute per-user positive/negative index arrays
  once, sample negatives with `rng.integers` over per-user counts.
- Users with only one class contribute no pairs; the hybrid variant keeps their rows in play.
- Ties: BPR can leave scores of rows never paired together nearly equal; evaluate.py breaks ties by file order —
  add a tiny logloss term or a small L2 to avoid exact ties.

## Measured
_Verdict:_ ACCEPTED 9x (live_01:node_001 on [official FM] Δ +0.0022; live_02:node_001 on [official FM] Δ +0.0016; live_03:node_001 on [official FM] Δ +0.0017; live_04:node_002 on [official FM] Δ +0.0011; live_05:node_002 on [official FM] Δ +0.0017; live_06:node_002 on [official FM] Δ +0.0013; live_07:node_003 on [official FM] Δ +0.0016; live_08:node_001 on [official FM] Δ +0.0010; live_08:node_007 on [official FM + loss-bpr-pairwise-within-user] Δ +0.0011)
- live_01:node_001 on [official FM]: primary 0.6036, single-seed Δ +0.0022 — ACCEPTED; 433 changed lines
- live_02:node_001 on [official FM]: primary 0.6031, single-seed Δ +0.0016, seed-mean Δ +0.0016 (t 8.22) — ACCEPTED; 34 changed lines
- live_03:node_001 on [official FM]: primary 0.6036, single-seed Δ +0.0022, seed-mean Δ +0.0017 (t 6.39) — ACCEPTED; 37 changed lines
- live_04:node_002 on [official FM]: primary 0.6028, single-seed Δ +0.0014, seed-mean Δ +0.0011 (t 3.83) — ACCEPTED; 37 changed lines
- live_04:node_006 on [official FM + field-aware FM embeddings]: primary 0.6030, single-seed Δ +0.0000 — rejected; 3 changed lines
- live_05:node_002 on [official FM]: primary 0.6036, single-seed Δ +0.0022, seed-mean Δ +0.0017 (z 6.49) — ACCEPTED; 34 changed lines
- live_06:node_002 on [official FM]: primary 0.6028, single-seed Δ +0.0014, seed-mean Δ +0.0013 (z 3.59) — ACCEPTED; 36 changed lines
- live_07:node_003 on [official FM]: primary 0.6031, single-seed Δ +0.0016, seed-mean Δ +0.0016 (z 5.87) — ACCEPTED; 44 changed lines
- live_06:node_011 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average] (variant: loss-bpr-pairwise-within-user — Replace 10% of ordinary same-user BPR samples with positive-negative pairs for which both impressions have): primary 0.6039, single-seed Δ -0.0000 — rejected; 16 changed lines
- live_06:node_016 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average] (variant: loss-bpr-pairwise-within-user — Replace 5% of ordinary same-user BPR pairs with same-user, same-tab positive-negative pairs where both impress): primary 0.6038, single-seed Δ -0.0002 — rejected; 19 changed lines
- live_06:node_019 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average] (variant: loss-bpr-pairwise-within-user — Replace 2.5% of ordinary same-user BPR samples with same-user positive-negative pairs for which both impressio): primary 0.6041, single-seed Δ +0.0002, seed-mean Δ -0.0000 (z -0.04) — rejected; 18 changed lines
- live_08:node_001 on [official FM]: primary 0.6030, single-seed Δ +0.0016, seed-mean Δ +0.0010 (z 3.73) — ACCEPTED; 33 changed lines
- live_08:node_007 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6038, single-seed Δ +0.0008, seed-mean Δ +0.0011 (z 4.11) — ACCEPTED; 73 changed lines
