# KuaiRand-Pure Starter Kit

*(English translation of the official kit README. The original Chinese version is kept as `README.zh.md`.)*

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn needed.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration):

```bash
# Run inside the Starter Kit directory; extracting produces ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if the data lives elsewhere.

`--model` is one of `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for self-checking the evaluation code).
FM takes about 40 seconds end to end (CPU, single core).

## Task definition (the conventions are pinned — do not change them)

| | |
|---|---|
| Task | **Within-user ranking** — each user is ranked only over their own impressions in the evaluation split; no full-catalog retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = the mean of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Zero-positive users | nDCG is recorded as 0.0 and included in the average; GAUC only counts users with `0 < #positives < #impressions`, weighted by #positives |
| nDCG gain | `2^rel − 1` (identity for binary labels) |

See `evaluate.py` for the implementation; every convention is written in the file's header comment.

## Baseline ladder

Scores on the test split. **The row to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, self-check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metrics: nDCG@5's ceiling is 0.729, not 1.0

Of the 23,875 users in the test split:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of the user's impressions is a long_view) | **27.1%** | nDCG is always **0**; no model can change that; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Discriminative users | **63.7%** | the actual sample GAUC is computed on |

So even using the true labels as the prediction (the oracle, i.e. a perfect ranking) only reaches:

| | random | FM baseline | **oracle ceiling** | share of the range FM already captures |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Judge progress with the oracle as the denominator.** Seeing 0.5946 and thinking "still far from a perfect 1.0" is a misjudgement —
the baseline has already captured 30% of the attainable range, so the remaining headroom is 0.27, not 0.41.

FM's std over 5 random seeds is **0.0008** on every metric. From this, the convergence rule is **ε = 0.002 (≈2.5σ), N = 3**:
if the validation primary score improves by no more than 0.002 over 3 consecutive iterations, the run is judged converged.

> Self-check: if your evaluation code does not get primary ≈ 0.475 (±0.001) with `--model random`, your harness is broken — fix that first.

## Submission format

CSV with a header, one line per row of the evaluation split:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | 0-based, strictly increasing; matches the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, keep the original file order) |
| `user_id` / `video_id` | redundant fields, only used to verify alignment |
| `score` | your model's score for the row; any real number, only the relative order matters; NaN / Inf are not allowed |

> **Why `row_id` is required:** `(user_id, video_id)` is **not unique** in the evaluation split —
> 3.06% of test rows are repeated pairs, up to 12 times. So it cannot be the primary key.

Generating and validating:

```bash
python3 submit.py --make  --split test  submission.csv    # generate an example submission with the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (local valid split only)
```

`--check` rejects: a wrong header, a row-count mismatch, gaps in `row_id`, `user_id`/`video_id` misaligned with the evaluation split,
and a non-numeric or NaN/Inf `score`. **Run `--check` yourself before submitting.**

## Where to start changing things

The ordering below is **measured**, not guessed. Dead ends the organizers have already tried are marked explicitly — don't repeat them.

### Measured: these two give no gain — don't waste iterations on them

| Tried | Result |
|---|---|
| **Adding static features** — wiring in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** with 5 fields; no difference within noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887; essentially flat |

Reason: the `user_id × video_id` cross already captures most of the learnable signal. Coarse buckets like `follow_user_num_range`
are redundant next to `user_id`, and 1.14M rows cannot support more capacity. **The bottleneck is not features or capacity.**

⚠️ Also note: **first-order terms of pure user-side features contribute exactly 0 to the score.** Because ranking is done within each user,
any term that is constant within a user cannot change the order inside that group (measured: `item_pop × user_bias` and plain `item_pop`
give bit-identical scores). User-side features can only act through **cross terms with item-side features**.

### Unexplored: this is where the headroom should be

Ranked by our judgement of likelihood (**the organizers have not tested these — they are left for you**):

1. **Change the loss function.** Currently pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**.
   Switching to pairwise (BPR) or listwise (softmax over the user's impressions) aligns the objective with the evaluation —
   we consider this the most likely to work.
2. **User history sequences.** The current features **make no use of behaviour sequences at all**. In KuaiRand each user has
   hundreds to thousands of interactions in train; DIN / SIM-style interest modelling is a completely blank direction.
3. **Multi-objective.** The log also has `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `play_time_ms`;
   these can serve as auxiliary tasks supporting the `long_view` main task.
4. **Modelling watch time.** This is exactly [CWM](https://github.com/hyz20/CWM)'s contribution: it models watch time with
   **censored regression** (when a video is played to completion the true watch time is truncated, so a one-sided loss is used
   instead of squared error). This is a direction with research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity has been measured not to be the bottleneck, **prioritise this after 1–4**.
6. **Time features and distribution drift.** `hourmin`, `date`, and the drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is the random-exposure log (1.18M rows); it can be
   used as an additional unbiased validation set to check whether the model only overfits biased traffic.

## Using your own model (including CWM)

`evaluate.py` is fully decoupled from the model; it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id of every row in the evaluation split
- `labels`: that row's `long_view` (0/1)
- `scores`: your model's score for that row (any real number, only the relative order matters)

So you don't have to use `baseline.py` at all — swap in PyTorch, LightGBM, or [CWM](https://github.com/hyz20/CWM)'s xDeepFM,
as long as you hand `scores` to `evaluate()` at the end. **The scoring convention is determined solely by `evaluate.py`.**

> Note on CWM: it depends on `torch==1.6.0` (a 2020 release that probably won't install on new GPUs),
> its loss optimises counterfactual watch time, and its evaluation label is its own rebuilt `long_view2`.
> It is the research code of a watch-time debiasing paper — an **advanced reference**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + every scoring convention. **Do not modify.** |
| `data.py` | Data loading, official split, feature encoding. Add features here. |
| `baseline.py` | The three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiment; reproduces the "adding features gives no gain" numbers. |
