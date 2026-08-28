# Final remaining-lever campaign — validation only

Protocol: all cells read only `data/real_ws/train.{npz,csv}` and
`data/real_ws/val.{npz,csv}`. The runner asserts train dates are 20220408–20220421,
validation dates are 20220422–20220428, and rejects every date >= 20220429. Every
reported score is produced by `data/official/evaluate.py`. Exploration uses seed
42. A new lever is called a win only when its primary is at least 0.6036 (the
specified +0.002 over 0.6016) and it has been rerun at seeds 43 and 44. The valid
primary reference for controlled comparisons is 0.6041. Each process has a 350s
hard alarm and model selection uses official validation primary every half epoch.

Implementation assumption: “half-epoch” is the checkpoint after the first
`ceil(n_batches/2)` optimizer batches, followed by the ordinary end-of-epoch
checkpoint. The control uses exactly the five encoded NPZ fields, one DCN cross
layer, a 128-unit MLP, dropout 0.1, and exactly 0.5 within-user BPR + 0.5 logloss.

## S0 — DCN-lite control

Command: `uv run python zoo/final_control.py --data-dir real --out-dir
/tmp/final_control_s42 --seed 42`

| seed | GAUC | nDCG@5 | primary | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6712936 | 0.5376250 | 0.6044593 | 2.0 | 25.9s | control verified (~0.604) |

The reimplementation is 0.00036 above the 0.6041 primary reference and therefore
passes the required sanity check. It peaked at epoch 2 and subsequently overfit.

## E1 — hierarchical user-embedding shrinkage

Personal user embeddings are mixed with count-decile cohort embeddings. The gate
is `sigmoid(a * standardized_log1p(train_impression_count) + b)` with learned
parameters. Cohorts and count normalization are fit on train only. A first process
exposed and fixed an in-place-autograd implementation bug before producing any
score; the clean scored run used early-stop patience 4.

| seed | GAUC | nDCG@5 | primary | delta vs control | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6677281 | 0.5355633 | 0.6016457 | -0.0028136 | 1.0 | 20.3s | loss; no reseed |

The hierarchy overfits immediately and is essentially at the raw 0.6016 baseline,
so it is neither a controlled improvement nor an absolute-threshold candidate.

## E2 — sparse cross IDs with frequency backoff

Four train-vocabulary fields were appended: user×tab, user×duration-regime
(`duration_ms <= 18000`), author×tab, and tab×official-duration-bucket. Each cross
type has its own UNK token, and every train key with count below 20 maps to it in
both train and validation.

| seed | GAUC | nDCG@5 | primary | delta vs control | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6698072 | 0.5367193 | 0.6032632 | -0.0011961 | 1.0 | 32.1s | neutral/loss; no reseed |

The cell misses the absolute 0.6036 confirmation threshold and regresses against
the same-code control, so seeds 43/44 are not run.

## E3 — item freshness and causal popularity velocity

Features are log upload age at impression plus day-decayed exposure volume and
long-view rate per video. For every row, velocity uses only train days strictly
earlier than that row's impression date. Validation rows may use all earlier train
days; no validation outcomes enter a feature. Upload timestamps come only from
`video_features_basic_pure.csv`. Values enter as three dense inputs.

| half-life | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2d | 42 | 0.6697258 | 0.5373927 | 0.6035593 | +0.0019593 | 2.0 | 44.7s | neutral; below threshold |
| 5d | 42 | 0.6699754 | 0.5374443 | 0.6037098 | +0.0021098 | 2.0 | 51.0s | threshold candidate; confirmed |
| 5d | 43 | 0.6693119 | 0.5364796 | 0.6028957 | +0.0012957 | 1.5 | 39.3s | confirmation |
| 5d | 44 | 0.6701661 | 0.5374175 | 0.6037918 | +0.0021918 | 2.0 | 57.5s | confirmation |
| **5d mean ± population std** | **42–44** | **0.6698178 ± 0.0003654** | **0.5371138 ± 0.0004485** | **0.6034658 ± 0.0004025** | **+0.0018658** | — | — | **not a confirmed win** |

The 5-day seed-42 result triggered the mandatory confirmation, but its three-seed
mean misses +0.002 and is below the 0.6041 primary reference. The lever is closed
as neutral/loss rather than reported as a win.

## E4 — two-specialist gating

Specialist A uses the usual per-positive pair pool but raises BPR weight to 0.7.
Specialist B samples one pair per eligible user (user-uniform) and weights its
pointwise term inversely with the square root of train-history depth, an observable
proxy emphasizing short/top-critical lists. Predictions are converted to within-
user percentile ranks. The requested validation-fit gate searches 27 logistic
gates using standardized validation list size and train-history depth. This makes
the mixture an explicitly validation-selected exploratory result, not an unbiased
estimate; every seed repeats the whole gate fit.

| seed | specialist A | specialist B | gated primary | gate `(b,size,depth)` | runtime | verdict |
|---:|---:|---:|---:|---|---:|---|
| 42 | 0.6041870 | 0.5975702 | 0.6045852 | (1,0,0) | 44.1s | threshold candidate |
| 43 | 0.6025643 | 0.5970389 | 0.6029655 | (1,0,0) | 48.4s | confirmation |
| 44 | 0.6036790 | 0.5985334 | 0.6039694 | (1,0,1) | 56.1s | confirmation |
| **mean ± population std** | — | — | **0.6038400 ± 0.0006675** | — | — | **confirmed by specified threshold** |

Mean GAUC is 0.6705477 ± 0.0008787 and mean nDCG@5 is 0.5371323 ±
0.0004570. The +0.0022400 mean delta over 0.6016 meets the stated confirmation
rule, so this is honestly labeled a confirmed absolute-baseline win. It does not
beat the separate 0.6041 primary reference in three-seed mean, and its validation-
selected gate adds optimism; those caveats are retained for final selection.

## E5a — FFM-style k=8 closure

The field-aware head uses a separate k=8 embedding table for every target field,
all ten pairwise interactions among the five official fields, plus a linear term.

| seed | GAUC | nDCG@5 | primary | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6588201 | 0.5306045 | 0.5947123 | 4.0 | 145.1s | decisive loss; no reseed |

This is below both the baseline and control, despite remaining inside the six-
minute cap. The architecture is closed.

## E5b — FinalMLP closure

The FinalMLP-style head has two independent 128→64 ReLU towers over the flattened
five-field embeddings, concatenated into a 64-unit fusion head.

| seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | best step | runtime | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.6702986 | 0.5370700 | 0.6036843 | +0.0020843 | 1.5 | 70.5s | threshold candidate |
| 43 | 0.6695876 | 0.5371794 | 0.6033835 | +0.0017835 | 1.5 | 44.9s | confirmation |
| 44 | 0.6703754 | 0.5364352 | 0.6034053 | +0.0018053 | 1.5 | 45.6s | confirmation |
| **mean ± population std** | **0.6700872 ± 0.0003547** | **0.5368949 ± 0.0003281** | **0.6034910 ± 0.0001369** | **+0.0018910** | — | — | **not a confirmed win** |

The seed-42 threshold crossing does not survive confirmation, so FinalMLP is
closed as neutral/loss rather than a win.

## E6 — Optuna hyperparameter search

Pending.

## Final summary and conclusions

Pending completion of all cells.
