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
| tokens in | 14,106 |
| tokens out | 17,378 |
| total wall-clock | 919 s (0.26 h) |

## Prediction calibration

| iteration | node | expected delta | realized delta | absolute error |
|---|---|---:|---:|---:|
| 1 | node_001 | - | +0.003161 | - |
| 2 | node_002 | +0.002500 | -0.002671 | 0.005171 |
| 3 | node_003 | +0.003000 | -0.001657 | 0.004657 |
| 4 | node_004 | +0.001000 | -0.001642 | 0.002642 |

Mean absolute calibration error: **0.004157**
