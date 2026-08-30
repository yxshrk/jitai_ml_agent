# Champion robustness audit (analysis only; predeclared: no re-selection)

Champion (node_006 ensemble) primary: **0.605575**; baseline (calib seed42): 0.601853; delta **+0.003722**.

## Paired per-user bootstrap (400 resamples, users resampled with replacement)
Delta 95% CI: **[+0.002015, +0.005472]** (excludes zero — the gain is not a user-sampling artifact).

## Temporal slices (per validation day)
| date | rows | delta |
|---|---:|---:|
| 20220422 | 22283 | +0.00288 |
| 20220423 | 26645 | +0.00359 |
| 20220424 | 18240 | -0.00006 |
| 20220425 | 14911 | +0.00418 |
| 20220426 | 14530 | +0.00403 |
| 20220427 | 14328 | +0.00301 |
| 20220428 | 13972 | +0.00427 |

## User-activity bins
| bin | rows | delta |
|---|---:|---:|
| <=3 impressions | 19240 | +0.00543 |
| 4-8 | 44265 | +0.00384 |
| >8 | 61404 | +0.00435 |

Interpretation notes: per-day deltas fluctuate (small slices); the audit asks whether the gain is broad-based rather than concentrated in one day or one user segment. Bootstrap keeps each user's rows together (cluster bootstrap).
