# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | Because validation GAUC peaks around epoch 3.5 and then declines while training loss continues falling, the model is mildly overfitting; an aggressive coherent package of MLP dropout 0.3, accessed-row embedding L2, dense-only AdamW weight decay 1e-3, and rapid stepwise LR decay will improve validation primary by approximately 0.0025 over node_001 by preserving ranking quality beyond the early peak. | Because validation GAUC peaks around epoch 3.5 and then declines while training loss continues falling, the model is mildly overfitting; an aggressive coherent package of MLP dropout 0.3, accessed-row embedding L2, dense-only AdamW weight decay 1e-3, and rapid stepwise LR decay will improve validation primary by approximately 0.0025 over node_001 by preserving ranking quality beyond the early peak. | 0.6030 | +0.002500 | -0.002004 | no | - |
| 3 | draft | Diagnosis: validation primary peaks near epoch 3.5 and then declines, indicating mild overfit; adding zero-initialized, L2-regularized short- and long-duration residual prediction heads at the 18-second censoring boundary while retaining a shared DCN-lite trunk will improve validation primary by 0.0030 over node_001 because the two duration regimes can learn distinct ranking corrections without duplicating the high-capacity trunk. | Diagnosis: validation primary peaks near epoch 3.5 and then declines, indicating mild overfit; adding zero-initialized, L2-regularized short- and long-duration residual prediction heads at the 18-second censoring boundary while retaining a shared DCN-lite trunk will improve validation primary by 0.0030 over node_001 because the two duration regimes can learn distinct ranking corrections without duplicating the high-capacity trunk. | 0.5986 | +0.003000 | -0.006431 | no | - |
| 4 | improve | Diagnosis: validation primary peaks at epoch 3.5 and then declines, indicating mild overfit; reducing only the embedding dimension from k=16 to k=8 while keeping the regularized DCN-lite head and loss unchanged will reduce sparse-ID memorization and improve validation primary by 0.0020 over node_001. | Diagnosis: validation primary peaks at epoch 3.5 and then declines, indicating mild overfit; reducing only the embedding dimension from k=16 to k=8 while keeping the regularized DCN-lite head and loss unchanged will reduce sparse-ID memorization and improve validation primary by 0.0020 over node_001. | 0.6045 | +0.002000 | -0.000460 | no | - |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5, then declines while training loss continues to edge downward, indicating mild overfit. With 23 iterations left, the coherent dropout, row-L2, AdamW, and LR-decay package is the highest-expected-gain eligible treatment (+0.002-0.008) and is materially different from the measured-dead single-dose variants.
- Rejected alternative: swa-ema — The post-peak checkpoints are progressively weaker, and averaging offers only an estimated +0.000-0.003, below the regularization package's expected gain.

### Iteration 3: duration-regime-heads

- Diagnosis: overfit
- Selected card: duration-regime-heads — D2Q duration debiasing, KDD 2022 (arXiv:2206.06003); KuaiRand research notes
- Why: Validation primary peaks near epoch 3.5 and then declines, confirming mild overfit. However, the prior aggressive regularization package fell from 0.6050 to 0.6030, and overfit-family methods are portfolio-excluded. Duration-regime heads are an eligible, untried mechanism with a comparatively strong honest expected gain of +0.002 to +0.006; shared-trunk regularized heads may capture the distinct ranking behavior induced by the 18-second label regime without repeating failed global regularization.
- Rejected alternative: regularization-schedule — The coherent aggressive package was already tried in node_002 and rejected at 0.6030; it also belongs to the excluded overfit family.

### Iteration 4: embedding-dim-down

- Diagnosis: overfit
- Selected card: embedding-dim-down — `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss generally continues downward, indicating mild overfit. The coherent regularization package already failed at node_002, so reducing embedding capacity from k=16 to k=8 is the strongest untried, low-cost overfit treatment. It directly targets sparse ID memorization and has an honest expected gain of at most 0.003.
- Rejected alternative: regularization-schedule — The journal already tested the card's aggressive compound package at node_002; it fell from 0.6050 to 0.6030, so repeating it would disregard measured evidence.
