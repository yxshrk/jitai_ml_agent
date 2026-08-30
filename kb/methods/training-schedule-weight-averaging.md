---
id: training-schedule-weight-averaging
family: regularization
target_component: training-schedule
source: Izmailov et al. 2018, Stochastic Weight Averaging (arXiv 1803.05407) — not in kb/literature; standard practice; live_01 curves peak then fall (node_001 peak epoch 8, −0.003 by epoch 12)
applies_when:
  - the validation curve peaks and then declines while training loss keeps falling (true for every node so far)
  - epochs are cheap enough to keep several checkpoints (FM: ~1 s per epoch)
expected_delta: [0.0, 0.0001]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0002 over 2 measurement(s), so the promise is capped at the record; was: averaging the last few checkpoints (or an EMA of weights) reduces the seed/epoch variance
  that the seed re-runs exposed (std 0.0004–0.0005 per node); a variance reducer, not new signal
cost: ~15 lines (keep an EMA of V, W, b; evaluate the EMA copy); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-bpr-hard-negatives, regularization-embedding-dropout-l2, data-weighting-recency, model-dcn-cross-head]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user x2 (best Δ +0.0001)]
evidence: [live_02:node_010, live_02:node_014]
---
## Claim
Keep an exponential moving average of the parameters (decay 0.99 per step, or the mean of the checkpoints around
the validation peak) and score validation with the averaged weights.

## Mechanism (why it moves within-user ranking)
Adam's last steps jitter around the optimum; the averaged weights sit closer to the centre of the good region, which
generalises better than any single step and is less sensitive to the seed — exactly the noise that makes single-seed
deltas unreliable here.

## How to implement on node_000
1. After `FM.__init__`, allocate `self.Va, self.Wa, self.ba` copies; after each optimizer step:
   `Va = decay*Va + (1-decay)*V` (same for W, b), decay 0.99.
2. Evaluate validation each epoch with the averaged copy; keep best_state from the averaged parameters.
3. Alternative: store the parameter copies of the best 3 epochs and average them at the end.

## Risks / failure modes
- Too slow a decay lags behind training early on — start averaging after epoch 2.
- Gains are small by construction; confirm with seeds rather than reject on one.

## Measured
_Verdict:_ never accepted in 2 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x2 (best Δ +0.0001)
- live_02:node_010 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6003, single-seed Δ -0.0027 — rejected; 29 changed lines
- live_02:node_014 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6033, single-seed Δ +0.0002, seed-mean Δ +0.0001 (t 0.86) — rejected; 13 changed lines
