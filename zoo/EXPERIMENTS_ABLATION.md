# Field ablation curve — 2026-08-28

## Protocol

This campaign resumed the already-committed `zoo/ablate_fields.py` and
`tests/test_ablation.py` from commit `d51b456`. There was no partial results
document, dirty worktree, or stash to recover. The implementation was audited
before execution and the existing focused tests passed.

All results below are validation-only official-scorer outputs. The runner loads
only the frozen 2022-04-08 through 2022-04-21 training window and the 2022-04-22
through 2022-04-28 validation window; it does not load a test split. The fixed
model is DCN-lite with one cross layer, a 128-unit MLP, and 16-dimensional
embeddings. Its objective is 0.5 within-user BPR plus 0.5 pointwise logloss,
with seven-day-half-life recency weights. Checkpoints are evaluated every half
epoch with `data/official/evaluate.py`, selected on official `primary`, and the
complete checkpoint history is written to each run's `metrics.json`.

The base configuration uses dropout 0.1, no embedding dropout, Adam, no weight
decay, learning rate 1e-3, batch size 8192, and seed 42. The strong package is
the best regularization setting (`r03`) from `EXPERIMENTS_SWEEP.md`: MLP dropout
0.2, embedding dropout 0.1, AdamW weight decay 1e-5, and a per-epoch 0.5 step
decay, while retaining the fixed one-cross-layer architecture. Runs use a
330-second alarm to preserve margin under the six-minute limit.

The cumulative levels are:

- L0: user, video, author, tab, 10-bin duration.
- L1: L0 plus hour and day of week.
- L2: L1 plus 50-bin duration, duration <=18 seconds, and duration-bin x tab.
- L3: L2 plus coarse user activity, follow/fans, registration-age, and
  video-author metadata.
- L4: L3 plus video/upload type, train-top-200 music ID, first tag, aspect
  ratio, and visibility.
- L5: L4 plus leakage-safe train-window smoothed item/author long-view rates
  and upload age.

## Seed-42 ablation curve

| level | fields | GAUC | nDCG@5 | primary | delta vs 0.6016 | best epoch | runtime | <=6m |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **L0** | 5 | 0.671209 | 0.537461 | **0.604335** | +0.002735 | 2.0 | 23.0s | yes |
| L1 | 7 | 0.670256 | 0.537054 | 0.603655 | +0.002055 | 3.0 | 29.4s | yes |
| L2 | 10 | 0.670207 | 0.537202 | 0.603704 | +0.002104 | 2.0 | 28.3s | yes |
| L3 | 15 | 0.670225 | 0.537004 | 0.603614 | +0.002014 | 3.5 | 52.0s | yes |
| L4 | 21 | 0.671200 | 0.536979 | 0.604089 | +0.002489 | 2.0 | 47.2s | yes |
| L5 | 24 | 0.667332 | 0.536149 | 0.601740 | +0.000140 | 1.0 | 40.2s | yes |

The requested seed-42 curve peaks at **L0, primary 0.604335**. L1 through L4
form a narrow plateau below it. The kitchen-sink L5 addition is sharply harmful,
losing 0.002349 versus L4 and 0.002595 versus L0.

## Strong-regularization reruns

The seed-42 base winner L0 and L5 were rerun with the `r03` package.

| level | package | GAUC | nDCG@5 | primary | delta vs 0.6016 | change vs base | best epoch | runtime | <=6m |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **L0** | strong | 0.671859 | 0.538138 | **0.604998** | +0.003398 | +0.000663 | 3.5 | 32.2s | yes |
| L5 | strong | 0.669510 | 0.536473 | 0.602991 | +0.001391 | +0.001251 | 2.0 | 66.1s | yes |

Regularization helps both configurations and rescues part of L5's collapse, but
does not change the story: regularized L5 remains 0.002007 behind regularized L0
and 0.001344 behind unregularized L0. Larger field sets do not win.

## Required three-seed confirmations

Every seed-42 configuration with primary at least 0.6036 (baseline +0.002) was
rerun at seeds 43 and 44. Thus base L0 through L4 and regularized L0 were
confirmed; neither L5 run qualified. The aggregate columns report population
mean +/- population standard deviation across seeds 42, 43, and 44.

| configuration | seed 42 | seed 43 | seed 44 | 3-seed GAUC | 3-seed nDCG@5 | 3-seed primary | mean delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 base | 0.604335 | 0.603399 | 0.603319 | 0.670388 +/- 0.000590 | 0.536980 +/- 0.000342 | 0.603684 +/- 0.000461 | +0.002084 |
| L1 base | 0.603655 | 0.602624 | 0.603187 | 0.669579 +/- 0.000549 | 0.536732 +/- 0.000297 | 0.603155 +/- 0.000422 | +0.001555 |
| L2 base | 0.603704 | 0.603585 | 0.604211 | 0.670431 +/- 0.000430 | 0.537235 +/- 0.000113 | 0.603833 +/- 0.000271 | +0.002233 |
| L3 base | 0.603614 | 0.603465 | 0.604452 | 0.670499 +/- 0.000561 | 0.537188 +/- 0.000308 | 0.603844 +/- 0.000434 | +0.002244 |
| **L4 base** | 0.604089 | 0.603772 | 0.604142 | 0.670904 +/- 0.000248 | 0.537099 +/- 0.000190 | **0.604001 +/- 0.000163** | **+0.002401** |
| **L0 strong** | 0.604998 | 0.604250 | 0.604730 | 0.671546 +/- 0.000268 | 0.537773 +/- 0.000352 | **0.604660 +/- 0.000309** | **+0.003060** |

All confirmation runtimes were compliant: base L0 20.5-24.6s, L1 22.7-24.7s,
L2 28.3-37.4s, L3 45.7-52.0s, L4 47.2-67.6s, and strong L0 32.2-49.5s.
The campaign-wide maximum was **67.6s**, so no over-budget result needed to be
discarded or rerun.

## Conclusions

On the mandated seed-42 ablation curve, **L0 is the peak**. The more robust
three-seed comparison adds nuance: among unregularized confirmed levels,
**L4 has the highest mean (0.604001 +/- 0.000163)**, although its advantage over
L2/L3 is small and the seed-42 winner does not persist. L5 is the clear failure;
its smoothed rates and upload-age additions cause immediate overfit despite being
constructed only from the legal training window.

Strong regularization improves the small and large models, but **does not let the
bigger field set win**. Strong L0 is the campaign's best confirmed configuration
at **0.604660 +/- 0.000309**, while strong L5 reaches only 0.602991 at seed 42 and
does not meet the confirmation threshold. The practical result is to prefer the
compact L0 model with strong regularization; if regularization is held fixed at
the base setting and richer semantics are desired, L4 is the best multi-seed
field level, but the kitchen-sink L5 additions should be excluded.

## Reproduction

Base level (replace `N` with 0 through 5):

```bash
uv run python zoo/ablate_fields.py --data-dir data/real_ws \
  --out-dir /tmp/field_ablation_lN_s42 --field-level N --seed 42
```

Strong regularization and confirmation use `--regularized` and/or `--seed 43`
or `--seed 44` with distinct output directories. Every output directory contains
the selected validation `predictions.csv` and the real `metrics.json`, including
the full half-epoch history.
