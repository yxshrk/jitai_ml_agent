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
