---
id: training-schedule-lr-decay-early-stop
family: regularization
target_component: training-schedule
source: LOG.md baseline curve (0.5869 -> 0.6015 at epoch 7 -> 0.5990 at epoch 11); node_000 uses constant lr 1e-3, batch 8192, patience 4
applies_when:
  - the validation curve is peaked (rises then falls) — a decaying learning rate can hold the peak instead of overshooting it
  - epochs are cheap (~1 s each), so finer checkpoints cost nothing
expected_delta: [0.0, 0.0004]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0004 over 5 measurement(s), so the promise is capped at the record; was: schedule changes usually land inside the 0.002 noise floor on their own; they pay off bundled
  with regularisation or a new loss — treat this card as a booster, not a lead
cost: ~10 lines; runtime 1x; numpy only
composes_with: [regularization-embedding-dropout-l2, loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM x1 (best Δ -0.0004); official FM + field-aware FM embeddings x1 (best Δ +0.0000); official FM + loss-bpr-pairwise-within-user x2 (best Δ +0.0004); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0002)]
evidence: [live_03:node_002, live_04:node_008, live_05:node_009, live_05:node_015, live_06:node_013]
---
## Claim
Decay the learning rate once validation stops improving (x0.5 per stalled epoch), evaluate every half epoch, and
keep the best half-epoch checkpoint.

## Mechanism (why it moves within-user ranking)
With a constant lr the optimiser keeps moving past the validation optimum; decaying lets it settle near the peak.
Half-epoch checkpoints reduce the chance that the best state falls between two evaluations.

## How to implement on node_000
1. After an epoch with no improvement: `m.lr *= 0.5` (Adam step size), continue; stop when lr < 1e-5 or patience.
2. Split each epoch's permutation in two halves; run `evaluate` after each half; keep best_state at half-epoch granularity.
3. Optional: batch 4096 with lr 7e-4 (smaller steps, more updates).

## Risks / failure modes
- Gains within noise: never accept on one seed — rely on the grey-zone confirmation.
- Halving lr too early (patience 0) freezes training before the true peak — decay only after a stalled epoch.

## Measured
_Verdict:_ never accepted in 5 measurements on 4 stack(s); official FM x1 (best Δ -0.0004); official FM + field-aware FM embeddings x1 (best Δ +0.0000); official FM + loss-bpr-pairwise-within-user x2 (best Δ +0.0004); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0002)
- live_03:node_002 on [official FM]: primary 0.6011, single-seed Δ -0.0004 — rejected; 1 changed lines
- live_04:node_008 on [official FM + field-aware FM embeddings]: primary 0.6030, single-seed Δ +0.0000 — rejected; 4 changed lines
- live_05:node_009 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6040, single-seed Δ +0.0004, seed-mean Δ +0.0003 (z 1.1) — rejected; 2 changed lines
- live_05:node_015 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6041, single-seed Δ +0.0005, seed-mean Δ +0.0004 (z 1.39) — rejected; 2 changed lines
- live_06:node_013 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6040, single-seed Δ +0.0000, seed-mean Δ -0.0002 (z -0.38) — rejected; 7 changed lines
