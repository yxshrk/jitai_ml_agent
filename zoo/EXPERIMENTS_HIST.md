# User-history and data-level feature campaign — validation only

Protocol: all runs use `data/real_ws/{train,val}.npz` and the aligned CSVs. The
runner opens no test files. Train dates are asserted to lie in 20220408–20220421;
validation dates are asserted to lie in 20220422–20220428. All history and target
aggregates are fit only on the train window. Target-derived features use leave-one-
out statistics for training rows and the complete train window for validation.
Scores are produced only by `data/official/evaluate.py`. Exploration uses seed 42.
A win requires delta >= 0.002 over the 0.6016 baseline and confirmation at seeds
43 and 44, reported as mean ± population standard deviation over all three seeds.

Assumption: “duration-decile” means ten quantile buckets whose edges are fitted on
train durations. Affinity smoothing strength is 10 and rates are quantile-bucketed
into up to 20 categorical bins.

## Results (recorded in required order)

### S0 — best-stack sanity check

Command config: base DCN-lite, k=16, two cross layers, hidden=128, dropout=0.1,
0.5 BPR + 0.5 logloss, click/effective-view auxiliaries at 0.1, GAUC early stop.

| config | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | verdict |
|---|---:|---:|---:|---:|---:|---|
| base reproduction | 42 | 0.6719687 | 0.5375318 | 0.6047503 | +0.0031503 | sanity passed; not a new win |

Runtime 31.9s. This reproduces the prior seed-42 best-stack level (~0.6048), so
subsequent feature comparisons use this implementation as their control.

### E1 — per-user train-window affinities

Features: Bayesian-smoothed per-user long-view rate by author, tab, and train-
quantile duration decile; user global train rate is the prior, strength 10. Training
rows use leave-one-out pair/global counts; validation uses full train counts. Each
rate is added as a 20-quantile categorical bucket to the sanity-checked stack.

| config | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | verdict |
|---|---:|---:|---:|---:|---:|---|
| base + three affinity fields | 42 | 0.6699759 | 0.5370244 | 0.6035001 | +0.0019001 | **no-win**; below epsilon and -0.0012501 vs seed-42 control |

Runtime 46.1s. The affinity fields sharply overfit after epoch 1. Because E1 did
not show a real win, it is not reseeded and E5 DIN-lite will not be run.

### E2 — user global statistics × item/context interactions

Features: log2 user train-impression-count bucket × duration decile, plus leave-
one-out user global long-view-rate bucket × author for training (full train-window
user rate for validation). Both are categorical crosses on the best stack.

| config | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | verdict |
|---|---:|---:|---:|---:|---:|---|
| count×duration + rate×author | 42 | 0.6566011 | 0.5308215 | 0.5937113 | -0.0078887 | **no-win**; below baseline |

Runtime 51.2s. The large sparse rate×author vocabulary overfits from epoch 1; this
branch is not reseeded.

### E3 — recency weighting

Training sample weights are `2^(-age_days / half_life)`, normalized to mean one;
the same per-impression weights apply to pointwise, auxiliary, and pairwise losses.
All three requested half-lives are tested without and with E1 features.

| affinity | half-life | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| no | 3d | 42 | 0.6706762 | 0.5370435 | 0.6038598 | +0.0022598 | promising by baseline threshold; sweep pending |
| no | 7d | 42 | 0.6732567 | 0.5382032 | 0.6057299 | +0.0041299 | promising; pending seeds 43/44 |
| no | 14d | 42 | 0.6724718 | 0.5378521 | 0.6051619 | +0.0035619 | promising; confirmation decision after sweep |
| yes | 3d | 42 | 0.6685455 | 0.5365872 | 0.6025664 | +0.0009664 | **no-win** |
| yes | 7d | 42 | 0.6699085 | 0.5371535 | 0.6035310 | +0.0019310 | **no-win**; below epsilon |
| yes | 14d | 42 | 0.6704306 | 0.5374086 | 0.6039196 | +0.0023196 | seed-42 promising only; worse than no-affinity variants |

The 7-day no-affinity configuration is the selected candidate. Other seed-42
rows above +0.002 are exploratory observations, not claimed wins; only the best
candidate is advanced to mandatory seeds 43/44 confirmation.

Runtime note: this run reported 1770.8s because the process experienced an
external execution stall between epochs 1 and 2; normal compute epochs remained
under 22s. The queued sweep was interrupted, and the interrupted half-life-7 run
is not recorded as a result.
