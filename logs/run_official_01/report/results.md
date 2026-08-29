# Run results

| metric | value |
|---|---|
| final valid best GAUC | 0.6719 |
| final valid best nDCG@5 | 0.5381 |
| final valid best primary | 0.6050 |
| official FM baseline (valid primary) | 0.6016 |
| delta vs baseline | +0.0034 |
| best node | node_001 (iteration 1) |

## Run accounting

| | |
|---|---|
| iterations used | 4 / 50 (official cap) |
| GPU-hours | 0 (CPU-only) |
| baseline reproduction | PASS: seed primaries [0.601838], mean 0.6018 vs published 0.6016 |
| accepted | 2 |
| rejected | 3 |
| errors | 0 |
| human interventions | 0 |
| tokens in | 9,758 |
| tokens out | 12,063 |
| total wall-clock | 521 s (0.14 h) |

## Prediction calibration

| iteration | node | expected delta | realized delta | absolute error |
|---|---|---:|---:|---:|
| 1 | node_001 | - | +0.003161 | - |
| 2 | node_002 | +0.002500 | -0.001787 | 0.004287 |
| 3 | node_003 | +0.003000 | -0.002202 | 0.005202 |
| 4 | node_004 | +0.001500 | +0.000000 | 0.001500 |

Mean absolute calibration error: **0.003663**
