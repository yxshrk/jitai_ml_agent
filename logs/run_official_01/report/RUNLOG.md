# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive coherent regularization schedule combining MLP dropout 0.3, accessed-row embedding L2, AdamW dense-weight decay 1e-3, and 0.5-per-epoch learning-rate decay will reduce overfitting and improve validation primary by about 0.0025. | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive coherent regularization schedule combining MLP dropout 0.3, accessed-row embedding L2, AdamW dense-weight decay 1e-3, and 0.5-per-epoch learning-rate decay will reduce overfitting and improve validation primary by about 0.0025. | 0.6032 | +0.002500 | -0.001787 | no | - |
| 3 | draft | The early validation peak followed by decline diagnoses mild overfit, and adding zero-initialized, L2-regularized short/long-duration residual heads around a shared DCN-lite parent will model the 18-second censoring discontinuity without materially increasing effective capacity, improving validation primary by about 0.0030. | The early validation peak followed by decline diagnoses mild overfit, and adding zero-initialized, L2-regularized short/long-duration residual heads around a shared DCN-lite parent will model the 18-second censoring discontinuity without materially increasing effective capacity, improving validation primary by about 0.0030. | 0.6028 | +0.003000 | -0.002202 | no | - |
| 4 | improve | Because validation primary peaks at epoch 3.5 and then declines, diagnosing mild capacity-driven overfit, halving the embedding dimension from 16 to 8 while keeping the loss, head, and training schedule unchanged will improve validation primary by approximately 0.0015. | Because validation primary peaks at epoch 3.5 and then declines, diagnosing mild capacity-driven overfit, halving the embedding dimension from 16 to 8 while keeping the loss, head, and training schedule unchanged will improve validation primary by approximately 0.0015. | 0.6050 | +0.001500 | +0.000000 | no | -> patched |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5, then declines while training loss continues trending downward, indicating mild overfit. With 23 iterations left, the coherent dropout/L2/AdamW/LR-decay package has the highest honest expected gain (+0.002-0.008) among eligible overfit treatments and is materially different from the measured-dead single-dose variants.
- Rejected alternative: embedding-dim-down — Eligible for overfit, but expected recovery is only <=0.003 and the curve shows mild late-epoch degradation better addressed by a full regularization schedule.

### Iteration 3: duration-regime-heads

- Diagnosis: overfit
- Selected card: duration-regime-heads — D2Q duration debiasing, KDD 2022 (arXiv:2206.06003); KuaiRand research notes
- Why: Validation peaks at epoch 3.5 and then declines while training loss continues edging down, indicating mild overfit. However, the overfit family is portfolio-excluded and the prior aggressive regularization package already reduced primary from 0.6050 to 0.6032. Duration-regime heads are an untried, low-medium-cost orthogonal mechanism with the strongest eligible expected gain (+0.002-0.006); they can address the known 18-second label-regime discontinuity without repeating failed regularization.
- Rejected alternative: regularization-schedule — Its specified aggressive compound package was already tried in node_002 and fell to 0.6032; the overfit family is also excluded.

### Iteration 4: embedding-dim-down

- Diagnosis: overfit
- Selected card: embedding-dim-down — `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- Why: Validation primary peaks at epoch 3.5 and declines while training loss continues to decrease, indicating mild capacity-driven overfit. Reducing embeddings from k=16 to k=8 is the strongest clean, untried overfit intervention remaining; k=32 was already worse, and the previously tested aggressive regularization package materially hurt primary.
- Rejected alternative: regularization-schedule — The coherent aggressive package was already tested in node_002 and rejected at 0.6032 versus the 0.6050 parent; repeating it would ignore measured evidence.
