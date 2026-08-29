# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive regularization schedule combining MLP dropout 0.3, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch LR decay will delay overfitting and improve validation primary by about 0.003. | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive regularization schedule combining MLP dropout 0.3, accessed-row embedding L2, dense AdamW decay 1e-3, and 0.5-per-epoch LR decay will delay overfitting and improve validation primary by about 0.003. | 0.6019 | +0.003000 | -0.003095 | no | - |
| 3 | draft | Because validation primary peaks at epoch 3.5 and then falls, indicating mild overfit, adding quadratically shrunk short/long-duration residual heads around the shared DCN prediction will capture the 18-second censoring discontinuity without materially increasing effective capacity and improve validation primary by 0.0030. | Because validation primary peaks at epoch 3.5 and then falls, indicating mild overfit, adding quadratically shrunk short/long-duration residual heads around the shared DCN prediction will capture the 18-second censoring discontinuity without materially increasing effective capacity and improve validation primary by 0.0030. | 0.6046 | +0.003000 | -0.000407 | no | - |
| 4 | improve | Because validation primary peaks early at epoch 3.5 and then mildly declines, indicating overfit, an EMA with decay 0.998 over training weights will average the useful near-peak trajectory and improve validation primary by approximately 0.0010. | Because validation primary peaks early at epoch 3.5 and then mildly declines, indicating overfit, an EMA with decay 0.998 over training weights will average the useful near-peak trajectory and improve validation primary by approximately 0.0010. | 0.6039 | +0.001000 | -0.001121 | no | - |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5, then declines while training loss continues to edge downward, indicating mild overfitting. With 23 iterations left, the coherent dropout, row-L2, AdamW, and LR-decay package has the highest honest expected gain among eligible overfit treatments and can extend useful validation learning beyond the current peak; this is materially different from the measured-dead single-dose variants.
- Rejected alternative: swa-ema — The post-peak checkpoints are already degrading, and its expected gain of at most 0.003 is lower than the compound regularization package.

### Iteration 3: duration-regime-heads

- Diagnosis: overfit
- Selected card: duration-regime-heads — D2Q duration debiasing, KDD 2022 (arXiv:2206.06003); KuaiRand research notes
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss trends downward, indicating mild overfit. However, the overfit family is portfolio-excluded and the prior aggressive regularization package was rejected at 0.6019 versus the 0.6050 parent. Duration-regime heads are an untried, non-dead diversification with the strongest credible eligible upside (+0.002-0.006); regularizing both heads toward the shared parent limits added-capacity risk while addressing the label-regime discontinuity at 18 seconds.
- Rejected alternative: regularization-schedule — The relevant aggressive compound package was already rejected at 0.6019, and the overfit family is excluded from this portfolio step.

### Iteration 4: swa-ema

- Diagnosis: overfit
- Selected card: swa-ema — SWA (Izmailov et al.); EMA training practice; `research/models-losses-hparams.md` seed-variance guidance
- Why: Validation primary peaks at epoch 3.5 and then trends downward while training loss continues to improve, indicating mild overfit. Several late checkpoints remain close to the peak rather than collapsing, so SWA/EMA can average this useful near-peak region and reduce optimizer variance at low cost. Honest expected gain is small, about 0.000-0.003, but this is the best-fitting untried overfit intervention after the aggressive regularization package failed.
- Rejected alternative: regularization-schedule — The prescribed compound package was already tested in node_002 and fell from 0.6050 to 0.6019, so repeating it would ignore the journal outcome.
