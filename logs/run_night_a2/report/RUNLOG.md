# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive regularization schedule combining MLP dropout 0.30, accessed-row embedding L2, dense-only AdamW decay 1e-3, and 0.5-per-epoch learning-rate decay will reduce overfit and improve validation primary by about 0.0025. | Because validation primary peaks at epoch 3.5 and then declines while training loss continues falling, an aggressive regularization schedule combining MLP dropout 0.30, accessed-row embedding L2, dense-only AdamW decay 1e-3, and 0.5-per-epoch learning-rate decay will reduce overfit and improve validation primary by about 0.0025. | 0.6013 | +0.002500 | -0.003702 | no | - |
| 3 | draft | Although the parent shows mild overfit after its epoch-3.5 validation-primary peak, replacing its unweighted within-user BPR term with delta-nDCG@5-weighted BPR on complete five-impression user groups will better align gradients with the official top-5 metric and improve validation primary by approximately 0.0015. | Although the parent shows mild overfit after its epoch-3.5 validation-primary peak, replacing its unweighted within-user BPR term with delta-nDCG@5-weighted BPR on complete five-impression user groups will better align gradients with the official top-5 metric and improve validation primary by approximately 0.0015. | 0.6014 | +0.001500 | -0.003598 | no | - |
| 4 | improve | Because validation primary peaks at epoch 3.5 and then declines, indicating mild overfit, reducing only the embedding dimension from k=16 to k=8 while keeping the head, loss, regularization, and early stopping unchanged will reduce sparse-ID memorization and improve validation primary by approximately 0.0015. | Because validation primary peaks at epoch 3.5 and then declines, indicating mild overfit, reducing only the embedding dimension from k=16 to k=8 while keeping the head, loss, regularization, and early stopping unchanged will reduce sparse-ID memorization and improve validation primary by approximately 0.0015. | 0.6050 | +0.001500 | +0.000000 | no | -> patched |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5 and then declines while training loss continues edging downward, indicating mild overfit. With 18 iterations left, the coherent dropout, row-L2, AdamW, and LR-decay package has the highest honest expected gain among eligible overfit treatments (+0.002-0.008) and can extend useful validation learning beyond the current early peak without repeating measured-dead single-dose variants.
- Rejected alternative: swa-ema — The post-peak decline is sharper evidence for controlling training dynamics directly; checkpoint averaging offers only an estimated +0.000-0.003 and requires several consistently useful near-peak checkpoints.

### Iteration 3: dndcg-lambda

- Diagnosis: overfit
- Selected card: dndcg-lambda — LambdaRank/LambdaLoss literature; `research/models-losses-hparams.md` section 2
- Why: Validation peaks at epoch 3.5 and then declines while training loss edges downward, indicating mild overfit. However, the decline is only about 0.0004, and the prior aggressive regularization package severely reduced primary to 0.6013. Given the excluded overfit family, delta-nDCG weighting is the best eligible untried low-to-medium-cost option: it preserves the grouped ranking setup while aligning gradients more directly with the top-5 evaluator, with an honest expected gain of 0.000-0.003.
- Rejected alternative: regularization-schedule — Its aggressive compound implementation was already tried and rejected at 0.6013, and its overfit treatment family is excluded.

### Iteration 4: embedding-dim-down

- Diagnosis: overfit
- Selected card: embedding-dim-down — `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss continues trending downward, indicating mild overfit. The aggressive compound regularization package already failed badly in node_002, so the highest-value untried overfit intervention is a clean capacity reduction from k=16 to k=8. Prior evidence that k=32 underperforms k=16 supports testing lower embedding capacity, with an honest expected gain of at most about 0.003.
- Rejected alternative: regularization-schedule — The specified aggressive compound package was already tested in node_002 and fell from 0.6050 to 0.6013; repeating it is unsupported.
