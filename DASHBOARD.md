# Dashboard — everything tried, generated. Do not edit.

## Measured experiment cells

| campaign file | cells | confirmed wins | last update |
|---|---|---|---|
| EXPERIMENTS.md | 27 | 2 | 28 Aug 01:38 |
| EXPERIMENTS_ABLATION.md | 16 | 0 | 28 Aug 13:38 |
| EXPERIMENTS_AUDIT.md | 16 | 0 | 28 Aug 10:47 |
| EXPERIMENTS_DIMS.md | 25 | 0 | 28 Aug 14:43 |
| EXPERIMENTS_FINAL.md | 19 | 3 | 28 Aug 10:43 |
| EXPERIMENTS_HIST.md | 26 | 2 | 28 Aug 09:48 |
| EXPERIMENTS_SWEEP.md | 54 | 9 | 28 Aug 13:24 |

**~183 table rows logged, 16 confirmed-win markers.**

## Autonomous runs

| run | stop | iters | best primary | wall min | tokens |
|---|---|---|---|---|---|
| run_20260828-011439-bffeac | max_iters | 2 | 0.6633 | 1.1 | 11257 |
| run_ab_compact | converged | 3 | 0.6034 | 59.2 | 12904 |
| run_ab_full | converged | 3 | 0.6018 | 2.8 | 0 |
| run_ab_full2 | converged | 3 | 0.6018 | 18.3 | 39607 |
| run_real_01 | converged | 5 | 0.6042 | 23.5 | 27649 |
| run_real_02 | converged | 5 | 0.604 | 14.2 | 30731 |
| run_real_03 | converged | 3 | 0.6033 | 11.6 | 20000 |
| run_real_04 | converged | 3 | 0.6018 | 15.6 | 19257 |
| run_real_05 | converged | 3 | 0.6033 | 12.8 | 20125 |

## Lever status (from LEVERS.md)

### DEAD (11)
- pure BPR / pure logloss
- per-user listwise softmax (full history)
- ordinal watch-ratio aux
- CWM censored watch-time
- FM (k16)
- item/author Bayesian aggregates
- video content features
- user-history affinities (author/tab/duration)
- user-stats x item crosses
- single-dose dropout 0.15 / AdamW 1e-4
- LightGBM lambdarank + blends

### ALIVE (3)
- within-user BPR hybrid
- DCN-lite
- 50 dur buckets + <=18s flag + dur x tab + hour + dow

### IN-FLIGHT (24)
- dnDCG lambda, top-5 sized groups
- user-uniform vs positive-weighted specialist mix
- FFM (k8), FinalMLP
- DIN-lite sequence attention
- duration-regime two-head
- hierarchical user-embedding shrinkage
- user-metadata (coarse) x item crosses
- session features from time_ms
- item freshness (upload age) + causal popularity velocity
- explicit sparse cross IDs w/ frequency backoff
- co-visitation SVD embedding init
- early stop on GAUC
- joint dropout x wd x lr-schedule grid, embedding-specific reg
- batch/lr scaling, SWA/EMA
- recency weighting of train days
- Optuna polish of winner
- BPR pair sampling strategy (negs/pos count, popularity-weighted, hard negatives)
- auxiliary-task set over the 11 unscored signals (singles, combos, weights)
- optimizer family for sparse embeddings (Adagrad, split emb/MLP optimizers)
- field ablation curve L0-L5 + kitchen sink + regularization rescue
- diverse-architecture rank ensemble
- specialist gating (GAUC-model + nDCG-model)
- method-selector call + convergence pressure + stagnation reflector
- context-mode ablation (compact vs full-history proposer context)

### PARKED (2)
- log_random rows (dates overlap val/test)
- train on train+val for final test model

### N/A (5)
- xDeepFM/AutoInt
- user-constant features alone
- evaluate.py zip-truncation quirk
- pseudo-labeling val
- LLM-as-ranker (GenRec, Netflix, arXiv:2608.10257)

### TODO/QUEUED (1)
- final test-submission script (outside agent world)

## Anything untried?

Open items above are the complete untried set. If an idea is not in LEVERS.md at all, it is UNRECORDED — add it there first, that is the rule.
