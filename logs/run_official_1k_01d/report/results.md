# Run results

| metric | value |
|---|---|
| final valid best GAUC | 0.6591 |
| final valid best nDCG@5 | 0.5826 |
| final valid best primary | 0.6208 |
| official FM baseline (valid primary) | 0.6016 |
| delta vs baseline | +0.0192 |
| best node | node_000 (iteration 0) |

## Run accounting

| | |
|---|---|
| iterations used | 3 / 50 (official cap) |
| GPU-hours | 0 (CPU-only) |
| baseline reproduction | FAIL: seed primaries [0.620817], mean 0.6208 vs published 0.6016 |
| accepted | 1 |
| rejected | 1 |
| errors | 2 |
| human interventions | 0 |
| tokens in | 14,115 |
| tokens out | 12,055 |
| total wall-clock | 875 s (0.24 h) |

## Prediction calibration

| iteration | node | expected delta | realized delta | absolute error |
|---|---|---:|---:|---:|
| 1 | node_001 | +0.003000 | - | - |
| 2 | node_002 | +0.003000 | -0.018926 | 0.021926 |
| 3 | node_003 | +0.002500 | - | - |

Mean absolute calibration error: **0.021926**
