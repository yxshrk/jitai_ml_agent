# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | The parent curve peaks at epoch 3.5 and then declines, diagnosing overfit; an aggressive package of 0.30 MLP dropout, accessed-row embedding L2, dense-only AdamW decay 1e-3, and 0.5-per-epoch LR decay with GAUC selection will improve validation primary by about 0.0025. | The parent curve peaks at epoch 3.5 and then declines, diagnosing overfit; an aggressive package of 0.30 MLP dropout, accessed-row embedding L2, dense-only AdamW decay 1e-3, and 0.5-per-epoch LR decay with GAUC selection will improve validation primary by about 0.0025. | 0.6023 | +0.002500 | -0.002671 | no | - |
| 3 | draft | Because the parent validation primary peaks near epoch 3.5 and then only slightly declines, indicating mild overfit rather than severe collapse, adding regularized residual prediction heads routed by the impression-known 18-second duration regime will improve validation primary by 0.0030 by modeling the threshold-dependent ranking functions without changing the shared trunk or objective. | Because the parent validation primary peaks near epoch 3.5 and then only slightly declines, indicating mild overfit rather than severe collapse, adding regularized residual prediction heads routed by the impression-known 18-second duration regime will improve validation primary by 0.0030 by modeling the threshold-dependent ranking functions without changing the shared trunk or objective. | 0.6033 | +0.003000 | -0.001657 | no | - |
| 4 | improve | Because validation primary peaks at epoch 3.5 and then declines despite falling training loss, diagnosing mild overfit, stochastic weight averaging across epochs 2-5 of the unchanged regularized DCN-lite hybrid model will improve validation primary by approximately 0.0010 by stabilizing the broad near-peak checkpoint region. | Because validation primary peaks at epoch 3.5 and then declines despite falling training loss, diagnosing mild overfit, stochastic weight averaging across epochs 2-5 of the unchanged regularized DCN-lite hybrid model will improve validation primary by approximately 0.0010 by stabilizing the broad near-peak checkpoint region. | 0.6034 | +0.001000 | -0.001642 | no | - |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks near epoch 3.5 at 0.604998, then declines despite training loss continuing downward, indicating mild overfit. The compound regularization and LR schedule directly targets this shape and has the largest honest expected gain among eligible overfit cards (+0.002-0.008), with 18 iterations remaining for a coherent package rather than a previously dead single-dose variant.
- Rejected alternative: embedding-dim-down — Eligible capacity regularization, but its expected recovery is only <=0.003 and is less likely than the compound schedule to improve the peak epoch.

### Iteration 3: duration-regime-heads

- Diagnosis: overfit
- Selected card: duration-regime-heads — D2Q duration debiasing, KDD 2022 (arXiv:2206.06003); KuaiRand research notes
- Why: Validation peaks at epoch 3.5 and then declines, but the decline is small relative to run noise, while the prior aggressive overfit package fell from 0.6050 to 0.6023. Because the overfit family is portfolio-excluded, the best eligible untried pivot is duration-specific heads, with an honest estimated gain of +0.002 to +0.006 if the 18-second label regime induces materially different ranking functions.
- Rejected alternative: regularization-schedule — The coherent aggressive package was already tried and rejected at 0.6023, and the overfit family is excluded.

### Iteration 4: swa-ema

- Diagnosis: overfit
- Selected card: swa-ema — SWA (Izmailov et al.); EMA training practice; `research/models-losses-hparams.md` seed-variance guidance
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss continues edging down, indicating mild overfit. Several checkpoints from epochs 2.5-4.5 remain near the peak, satisfying the averaging precondition. SWA/EMA is an untried, low-cost way to stabilize this broad near-peak region after the aggressive regularization package failed; the honest expected gain is modest, about 0.000-0.003.
- Rejected alternative: regularization-schedule — The coherent aggressive package specified by this card was already tested in node_002 and fell to 0.6023, so repeating it is not justified.
