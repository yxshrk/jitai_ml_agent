# Final Pure marginal-method campaign

Run date: 2026-08-29. All commands used `uv run python`, `data/real_ws`, seed 42
for exploration, and the official scorer imported by `zoo/polish_stack.py`.
Only train dates were used for fitting. The loader's date guards restricted train
to 2022-04-08 through 2022-04-21 (the export actually begins on April 9) and
validation to April 22 through April 28. No test export was opened.

Promotion rule: a seed-42 primary of at least 0.605700 (0.604700 + 0.001000)
is required before seeds 43/44 may be run. No marginal cell reached that gate,
so no three-seed confirmations were run. Every training run finished below the
six-minute cap; no result is extrapolated.

## Harness control

Before any variant, the untouched `zoo/polish_stack.py` frozen defaults were
run at seed 42. The official primary was 0.604998355, exactly matching the prior
recorded seed-42 reproduction and within the known noise around 0.6047. This
confirmed the data split, scorer, and training harness.

## Complete results

| track | cell / config | seed(s) | val primary | delta vs 0.604700 | runtime | verdict |
|---|---|---:|---:|---:|---:|---|
| control | untouched frozen stack, 7-day recency | 42 | 0.604998355 | +0.000298355 | 33.4s | control reproduced |
| density ratio | hashed 5-field logistic ratio, cap 3, no recency | 42 | 0.604232013 | -0.000467987 | 38.9s | no-win |
| density ratio | hashed 5-field logistic ratio, cap 5, no recency | 42 | 0.603947448 | -0.000752552 | 40.4s | no-win |
| density ratio | hashed 5-field logistic ratio, cap 3 x 7-day recency | 42 | 0.603637892 | -0.001062108 | 41.7s | no-win |
| density ratio | hashed 5-field logistic ratio, cap 5 x 7-day recency | 42 | 0.603656202 | -0.001043798 | 46.6s | no-win |
| SAM | full SAM, rho 0.02, frozen 7-day weights | 42 | 0.604715650 | +0.000015650 | 91.6s | no-win |
| SAM | full SAM, rho 0.05, frozen 7-day weights | 42 | 0.604670328 | -0.000029672 | 83.6s | no-win |
| temporal windows | all available train rows (nominal 14-day window), frozen control reused | 42 | 0.604998355 | +0.000298355 | 33.4s | no-win |
| temporal windows | April 12-21 inclusive (last 10 calendar days), 7-day recency | 42 | 0.602125152 | -0.002574848 | 23.2s | no-win |
| temporal windows | April 15-21 inclusive (last 7 calendar days), 7-day recency | 42 | 0.595779388 | -0.008920612 | 13.7s | no-win |
| temporal windows | equal within-user rank average of all/10-day/7-day members | 42 | 0.603114863 | -0.001585137 | 0.3s | no-win |
| cyclic snapshots | epoch-2 snapshot, diagnostic member | 42 | 0.603292624 | -0.001407376 | 33.9s shared | no-win |
| cyclic snapshots | epoch-4 snapshot, diagnostic member | 42 | 0.604330228 | -0.000369772 | 33.9s shared | no-win |
| cyclic snapshots | epoch-6 snapshot, diagnostic member | 42 | 0.604361433 | -0.000338567 | 33.9s shared | no-win |
| cyclic snapshots | equal within-user rank average of epochs 2/4/6 | 42 | 0.604768839 | +0.000068839 | 33.9s | no-win |

The three cyclic member rows are diagnostics from one 33.9-second training run,
not three separately charged runs.

## Track notes and verdicts

### 1. Density-ratio weighting: no-win

The tiny additive logistic discriminator used 4,096 hash buckets per frozen
categorical field and classified April 19-21 rows against earlier train rows.
The observed late-class prior was 0.053733. Ratios were prior-corrected,
upper-clipped at 3 or 5, lower-bounded only for numerical safety, and normalized
to mean one. After normalization, the maximum weights were 2.6233 and 4.1906.
All four variants regressed, and adding recency made the regression larger.

### 2. SAM: no-win

Full two-pass SAM fit comfortably inside the time budget, so no ASAM or
final-epochs-only fallback was necessary. Rho 0.02 was effectively flat and rho
0.05 was slightly below the rounded baseline; both missed the promotion gate by
about 0.00098 or more.

### 3. Snapshot ensembles: no-win

Shorter windows lost substantial item/user coverage and regressed sharply. Their
rank average also regressed. The cheaper fixed cyclic schedule used cosine cycles
with 0.001 maximum LR, 0.0001 minimum LR, two-epoch periods, and snapshots at the
ends of epochs 2, 4, and 6. Its ensemble was only +0.000069 over 0.6047, far below
the +0.001 confirmation gate.

### 4. Best combination: no-win (eligibility gate closed)

The track was conditional on a winning survivor from tracks 1-3. None reached
0.605700 at seed 42, so combining one with the existing five-seed recipe would
have violated the stated promotion rule. No add-in blend was run and no metric is
invented. Consequently there was no candidate that could be tested against the
strict adoption threshold of greater than 0.606270 (0.605770 + 0.000500).

## Decision

Do not change the existing 0.60577 Pure submission recipe. Keep the documented
five frozen-stack seeds `{46, 74, 93, 91, 60}` and their equal per-user rank
average. None of the three literature-sweep families earned confirmation or an
add-in trial.
