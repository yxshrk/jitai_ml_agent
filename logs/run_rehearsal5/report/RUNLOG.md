# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | Because validation GAUC peaks near epoch 3.5 while training loss continues falling, the model is mildly overfitting; an aggressive package of 0.30 MLP dropout, accessed-row embedding L2, dense-only AdamW weight decay, and rapid epoch-wise LR decay will delay that overfit and improve validation primary by about 0.0025. | Because validation GAUC peaks near epoch 3.5 while training loss continues falling, the model is mildly overfitting; an aggressive package of 0.30 MLP dropout, accessed-row embedding L2, dense-only AdamW weight decay, and rapid epoch-wise LR decay will delay that overfit and improve validation primary by about 0.0025. | 0.6031 | +0.002500 | -0.001880 | no | - |
| 3 | draft | The validation curve peaking around epoch 3.5 and then declining diagnoses mild capacity-driven overfit; reducing only the embedding dimension from k=16 to k=8 while preserving the regularized DCN-lite head, hybrid loss, seed, and early stopping will improve validation primary by approximately 0.0015. | The validation curve peaking around epoch 3.5 and then declining diagnoses mild capacity-driven overfit; reducing only the embedding dimension from k=16 to k=8 while preserving the regularized DCN-lite head, hybrid loss, seed, and early stopping will improve validation primary by approximately 0.0015. | 0.0000 | +0.001500 | - | no | invalid metric gauc=None -> reverted |
| 4 | debug | Reducing only the embedding dimension from k=16 to k=8 while preserving the regularized DCN-lite head, 0.5 BPR plus 0.5 logloss objective, and validation-GAUC early stopping will reduce capacity-driven overfitting and improve validation primary by approximately 0.0015. | Reducing only the embedding dimension from k=16 to k=8 while preserving the regularized DCN-lite head, 0.5 BPR plus 0.5 logloss objective, and validation-GAUC early stopping will reduce capacity-driven overfitting and improve validation primary by approximately 0.0015. | 0.6011 | +0.001500 | -0.003879 | no | - |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5 and then declines while training loss continues edging down, indicating mild overfit. With 13 iterations left, the coherent dropout 0.3, row-L2, AdamW, and LR-decay package has the highest honest expected gain among applicable untried cards (+0.002 to +0.008) and directly targets preservation of post-peak GAUC.
- Rejected alternative: embedding-dim-down — Also targets overfit, but its expected recovery is smaller (<=0.003) and only tests capacity reduction rather than the broader optimization and regularization failure.

### Iteration 3: embedding-dim-down

- Diagnosis: overfit
- Selected card: embedding-dim-down — `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss continues trending downward, indicating mild capacity-driven overfit. The compound regularization schedule already reduced primary from 0.6050 to 0.6031, so repeating it is not justified. Reducing embeddings from k=16 to k=8 is a clean, untried capacity intervention supported by k=32 underperforming k=16, with a plausible gain of up to 0.003 at low cost.
- Rejected alternative: regularization-schedule — The aggressive compound package was just measured at 0.6031 and rejected versus the 0.6050 parent; repeating the same mechanism is not honest.
