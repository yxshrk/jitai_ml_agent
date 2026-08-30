---
id: regularization-embedding-dropout-l2
family: regularization
target_component: regularization
source: kb/literature/models/1706.06978_din.pdf (mini-batch-aware regularisation for sparse ids); LOG.md baseline curve (valid peaks at epoch 7, train loss keeps falling)
applies_when:
  - the champion's learning curve shows validation peaking early then falling while training loss keeps dropping (node_000: peak epoch 7 of 11)
  - most parameters are per-id embeddings (27 K users x 16 + 7.6 K videos x 16), the classic overfitting site
expected_delta: [0.0, 0.0010]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0010 over 10 measurement(s), so the promise is capped at the record; was: overfitting is measured, so there is something to regularise, but the user x video memorisation
  is also where the signal lives — the sweet spot is narrow; expect small gains, confirm with seeds
cost: ~15 lines (L2 sweep is a flag; field dropout ~10 lines); runtime 1x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, model-dcn-cross-head, training-schedule-lr-decay-early-stop, data-weighting-recency]
conflicts_with: []
status: proven — accepted on [official FM]
evidence: [live_02:node_006, live_03:node_003, live_04:node_004, live_04:node_007, live_05:node_003, live_05:node_007, live_06:node_003, live_07:node_004, live_07:node_008, live_06:node_009, live_06:node_017]
---
## Claim
Stronger, better-targeted regularisation of the embeddings — L2 raised from 1e-6 toward 1e-5/1e-4, or dropout of
whole field vectors during training — lets training run past epoch 7 without the validation drop.

## Mechanism (why it moves within-user ranking)
Rare ids (users with 6 rows, videos with few viewers) get embeddings fitted to noise; shrinking them toward zero
leaves the ordering to the well-estimated parts (popular videos, active users, tab, duration). DIN's mini-batch-aware
L2 applies the penalty only to ids present in the batch, scaled by their frequency — cheap and effective for sparse ids.

## How to implement on node_000
1. L2 sweep: `FM(dim, k, lr, l2=1e-5)` and 1e-4 — one flag; measure each as its own node or as a two-level probe.
2. Field dropout: in `step`, mask = rng.random((B, F, 1)) > p; E = E * mask / (1 − p) before computing S and inter
   (recompute logits from the masked E); p in {0.1, 0.2}. No masking at prediction time.
3. Frequency-aware L2 (DIN): gV += l2 * V[X] / freq[X] accumulated only for ids in the batch.

## Risks / failure modes
- Over-regularising removes the user x video signal that beats popularity (+0.023) — if GAUC falls, back off.
- Dropout changes the effective learning rate; keep early stopping on primary.

## Measured
_Verdict:_ ACCEPTED 3x (live_04:node_004 on [official FM] Δ +0.0009; live_05:node_003 on [official FM] Δ +0.0010; live_07:node_004 on [official FM] Δ +0.0010); implementation failed in live_03:node_003
- live_02:node_006 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6036, single-seed Δ +0.0005, seed-mean Δ +0.0002 (t 1.15) — rejected; 4 changed lines
- live_03:node_003 on [official FM]: FAILED at implement — no runnable script produced (recovery: None)
- live_04:node_004 on [official FM]: primary 0.6025, single-seed Δ +0.0010, seed-mean Δ +0.0009 (t 2.86) — ACCEPTED; 2 changed lines
- live_04:node_007 on [official FM + field-aware FM embeddings]: primary 0.6032, single-seed Δ +0.0002, seed-mean Δ +0.0001 (t 0.43) — rejected; 2 changed lines
- live_05:node_003 on [official FM]: primary 0.6025, single-seed Δ +0.0010, seed-mean Δ +0.0010 (z 3.47) — ACCEPTED; 2 changed lines
- live_05:node_007 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6037, single-seed Δ +0.0000, seed-mean Δ +0.0003 (z 1.04) — rejected; 2 changed lines
- live_06:node_003 on [official FM]: primary 0.6025, single-seed Δ +0.0010, seed-mean Δ +0.0008 (z 1.97) — rejected; 2 changed lines
- live_07:node_004 on [official FM]: primary 0.6025, single-seed Δ +0.0010, seed-mean Δ +0.0010 (z 3.41) — ACCEPTED; 2 changed lines
- live_07:node_008 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6036, single-seed Δ +0.0005, seed-mean Δ +0.0001 (z 0.34) — rejected; 2 changed lines
- live_06:node_009 on [official FM + loss-bpr-pairwise-within-user] (variant: regularization-embedding-dropout-l2 — Regularize latent FM embeddings at 1e-5 while retaining a 1e-6 penalty on linear feature weights, decoupling): primary 0.6026, single-seed Δ -0.0002 — rejected; 6 changed lines
- live_06:node_017 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average] (variant: regularization-embedding-dropout-l2 — Apply L2=3e-6 to user, video, and author latent embeddings while retaining L2=1e-6 for tab and duration): primary 0.6042, single-seed Δ +0.0003, seed-mean Δ +0.0001 (z 0.35) — rejected; 10 changed lines
