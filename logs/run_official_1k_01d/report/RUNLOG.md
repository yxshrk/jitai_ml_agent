# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6208], mean 0.6208, sigma 0.0015 | 0.6208 | - | - | yes | - |
| 1 | draft | The parent is severely overfitting because validation primary falls from 0.6208 at epoch 0.5 to 0.5690 at epoch 2.5; an aggressive coherent package of MLP dropout 0.30, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch LR decay with GAUC checkpointing will improve validation primary by about 0.003. | The parent is severely overfitting because validation primary falls from 0.6208 at epoch 0.5 to 0.5690 at epoch 2.5; an aggressive coherent package of MLP dropout 0.30, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch LR decay with GAUC checkpointing will improve validation primary by about 0.003. | 0.0000 | +0.003000 | - | no | timeout after 120s -> reverted |
| 2 | debug | An aggressive package of MLP dropout 0.30, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch learning-rate decay with validation-GAUC checkpointing will reduce severe early overfitting and improve validation primary by approximately 0.003, while larger vectorized batches will allow the same approach to finish within the timeout. | An aggressive package of MLP dropout 0.30, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch learning-rate decay with validation-GAUC checkpointing will reduce severe early overfitting and improve validation primary by approximately 0.003, while larger vectorized batches will allow the same approach to finish within the timeout. | 0.6019 | +0.003000 | -0.018926 | no | - |
| 3 | draft | Because the parent peaks at epoch 0.5 and then collapses while training loss continues falling, it is overfitting sparse static IDs; adding strictly causal session-gap/session-position fields plus fine hour/day context crosses will supply shared transient-intent signal and improve validation primary by approximately 0.0025. | Because the parent peaks at epoch 0.5 and then collapses while training loss continues falling, it is overfitting sparse static IDs; adding strictly causal session-gap/session-position fields plus fine hour/day context crosses will supply shared transient-intent signal and improve validation primary by approximately 0.0025. | 0.0000 | +0.002500 | - | no | timeout after 120s -> reverted |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 1: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Training loss falls continuously while validation primary peaks at epoch 0.5 (0.6208) and collapses to 0.5690 by epoch 2.5, showing severe early overfitting. The compound dropout, row-L2, weight-decay, and LR-decay package directly targets this curve shape and has the highest honest expected gain among eligible overfit treatments, with 19 iterations remaining to evaluate it cleanly.
- Rejected alternative: embedding-dim-down — Also targets overfitting, but its expected gain is smaller (<=0.003) and a capacity-only change is less likely than the compound schedule to arrest the sharp post-peak collapse.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Training loss falls continuously while validation primary peaks at epoch 0.5 (0.6208) and collapses to 0.5690 by epoch 2.5, showing severe early overfitting. The compound dropout, row-L2, weight-decay, and LR-decay package directly targets this curve shape and has the highest honest expected gain among eligible overfit treatments, with 19 iterations remaining to evaluate it cleanly.
- Rejected alternative: embedding-dim-down — Also targets overfitting, but its expected gain is smaller (<=0.003) and a capacity-only change is less likely than the compound schedule to arrest the sharp post-peak collapse.

### Iteration 3: session-time-features

- Diagnosis: overfit
- Selected card: session-time-features — KuaiRand schema/research notes; MENU temporal-context item 9
- Why: Validation peaks at epoch 0.5 and then collapses from 0.6208 to 0.5690 while training loss keeps falling, confirming severe overfit. The direct regularization opening has already failed twice and the overfit family is portfolio-excluded. Causal session position, gap, and fine-time context provide shared transient-intent signal that may generalize better than memorized sparse IDs, with an honest estimated gain of +0.001 to +0.005 and 17 iterations remaining.
- Rejected alternative: regularization-schedule — It directly matches the diagnosis but its aggressive measured opening failed and the overfit family is explicitly excluded.
