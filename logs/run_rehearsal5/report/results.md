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
| rejected | 2 |
| errors | 1 |
| human interventions | 0 |
| tokens in | 9,183 |
| tokens out | 10,181 |
| total wall-clock | 356 s (0.10 h) |

## Prediction calibration

| iteration | node | expected delta | realized delta | absolute error |
|---|---|---:|---:|---:|
| 1 | node_001 | - | +0.003161 | - |
| 2 | node_002 | +0.002500 | -0.001880 | 0.004380 |
| 3 | node_003 | +0.001500 | - | - |
| 4 | node_004 | +0.001500 | -0.003879 | 0.005379 |

Mean absolute calibration error: **0.004880**
