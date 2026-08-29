# Run log (per iteration)

| n | action | hypothesis | change | primary | expected delta | realized delta | accepted | error / recovery |
|---|---|---|---|---|---|---|---|---|
| 0 | reproduce_baseline | reproduce official FM baseline and calibrate seed noise | baseline seeds [42]: primaries [0.6018], mean 0.6018, sigma 0.0008 | 0.6018 | - | - | yes | - |
| 1 | draft | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | team-provided reference implementation: frozen_stack.py (from MENU frozen stack) | 0.6050 | - | +0.003161 | yes | - |
| 2 | draft | The validation curve peaks near epoch 3.5 and then declines, diagnosing overfit; an aggressive package of 0.3 MLP dropout, accessed-row embedding L2, AdamW dense weight decay 1e-3, and 0.5-per-epoch LR decay will delay that decline and improve validation primary by at least 0.002. | The validation curve peaks near epoch 3.5 and then declines, diagnosing overfit; an aggressive package of 0.3 MLP dropout, accessed-row embedding L2, AdamW dense weight decay 1e-3, and 0.5-per-epoch LR decay will delay that decline and improve validation primary by at least 0.002. | 0.6013 | +0.002500 | -0.003652 | no | - |
| 3 | draft | Although the parent validation curve diagnoses mild overfit by peaking at epoch 3.5, replacing its uniform within-user BPR term with normalized absolute delta-nDCG@5 weighting on complete five-impression user groups will improve validation primary by approximately 0.0010 through better top-5 metric alignment. | Although the parent validation curve diagnoses mild overfit by peaking at epoch 3.5, replacing its uniform within-user BPR term with normalized absolute delta-nDCG@5 weighting on complete five-impression user groups will improve validation primary by approximately 0.0010 through better top-5 metric alignment. | 0.6040 | +0.001000 | -0.000965 | no | - |
| 4 | improve | The validation curve peaks at epoch 3.5 and then declines, diagnosing mild capacity overfit; reducing only the embedding dimension from k=16 to k=8 while keeping the regularized DCN-lite head and hybrid loss unchanged will improve validation primary by approximately 0.0015. | The validation curve peaks at epoch 3.5 and then declines, diagnosing mild capacity overfit; reducing only the embedding dimension from k=16 to k=8 while keeping the regularized DCN-lite head and hybrid loss unchanged will improve validation primary by approximately 0.0015. | 0.6050 | +0.001500 | +0.000000 | no | -> patched |

## Diagnose → select evidence

Each draft/improve decision records the selected method and a considered alternative.

### Iteration 2: regularization-schedule

- Diagnosis: overfit
- Selected card: regularization-schedule — `research/models-losses-hparams.md` section 4; MENU CURRENT DIRECTIVE
- Why: Validation primary peaks at 0.604998 around epoch 3.5 and then declines while training loss continues trending downward, indicating mild overfit. The coherent dropout, row-L2, AdamW, and LR-decay package directly targets this shape and has the strongest honest expected gain among eligible overfit cards (+0.002 to +0.008) at low cost. This is materially different from the measured-dead single-dose variants.
- Rejected alternative: swa-ema — Expected gain is smaller (+0.000 to +0.003), and checkpoint averaging is mainly variance reduction rather than directly preventing the post-peak decline.

### Iteration 3: dndcg-lambda

- Diagnosis: overfit
- Selected card: dndcg-lambda — LambdaRank/LambdaLoss literature; `research/models-losses-hparams.md` section 2
- Why: Validation primary peaks at epoch 3.5 and then declines, so the run is mildly overfit. However, the overfit family is portfolio-excluded and the prior aggressive regularization package sharply regressed to 0.6013. Delta-nDCG weighting is an eligible, low-medium-cost change that reuses the frozen stack's grouped BPR implementation while aligning gradients with top-5 evaluation; its honest expected gain is modest, about 0.000-0.003.
- Rejected alternative: regularization-schedule — The family is excluded, and the immediately prior aggressive compound schedule was rejected at 0.6013.

### Iteration 4: embedding-dim-down

- Diagnosis: overfit
- Selected card: embedding-dim-down — `research/models-losses-hparams.md` section 4; MENU item 6 and E2
- Why: Validation primary peaks at epoch 3.5 and then declines while training loss continues edging down, indicating mild capacity overfit. The aggressive regularization package already failed at 0.6013, whereas reducing embeddings from k=16 to k=8 is an untried, clean capacity intervention supported by k=32 underperforming k=16. Expected gain is honestly small, at most about 0.003, but it is among the highest-gain eligible untried overfit treatments.
- Rejected alternative: regularization-schedule — The journal already tested the card's coherent aggressive dropout/L2/AdamW/LR-decay package and it regressed to 0.6013.
