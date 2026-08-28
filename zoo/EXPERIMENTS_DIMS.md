# Under-swept dimensions campaign — validation only

## Protocol

- Control: the confirmed strong-regularized L0 from `ablate_fields.py`: exactly
  five offset-encoded NPZ fields; DCN-lite with one cross layer, MLP width 128,
  MLP dropout 0.2, and embedding dropout 0.1; `0.5 * within-user BPR + 0.5 *
  logloss`; no auxiliary task; AdamW at `1e-3` with weight decay `1e-5` and
  per-epoch 0.5 step decay; and seven-day recency weights normalized to mean one.
  Its prior three-seed result was **0.604660 +/- 0.000309**. The DIMS runner
  reproduces it before changing a swept dimension.
- Validation runs use only `data/real_ws/train.npz` and `val.npz` (plus matching
  CSV columns solely for raw video ids and auxiliary outcomes absent from NPZ).
  Checkpoints are scored with `data/official/evaluate.py` every half epoch and
  selected on official PRIMARY. `metrics.json` records the full run history.
- Exploration uses seed 42. Any cell at least +0.002 over the fixed 0.6016 PRIMARY
  baseline is confirmed at seeds 42, 43, and 44 before being called a real win.
- The runner enforces a 330-second training alarm, leaving margin below the strict
  six-minute per-run cap. Every final CSV is independently rescored by the official
  evaluator before a sweep is closed.
- Requested outcomes `follow`, `comment`, and `forward` are not columns in either
  workspace NPZ or CSV. Their cells are retained below as explicit failures rather
  than being omitted or populated with invented targets.

## Sweep 1 — BPR pair sampling

The DIMS implementation of the strong control was first reproduced at all three
required seeds. Its PRIMARY was 0.605425 / 0.604281 / 0.604561, or **0.604756
+/- 0.000487** (GAUC 0.671830 +/- 0.000548; nDCG@5 0.537682 +/- 0.000476).
This agrees with the earlier `ablate_fields.py` confirmation, 0.604660 +/-
0.000309, while using the generalized pair sampler needed by this sweep.

| cell | negatives / positive | policy | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | runtime | result |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| S1-control | 1 | uniform | 42 | 0.672499 | 0.538352 | **0.605425** | +0.003825 | 30.8s | reproduced; confirmed above |
| S1-n3 | 3 | uniform | 42 | 0.672032 | 0.537512 | 0.604772 | +0.003172 | 79.8s | below control |
| S1-n5 | 5 | uniform | 42 | 0.672357 | 0.537404 | 0.604880 | +0.003280 | 127.4s | below control |
| S1-pop | 1 | popularity^0.75 | 42 | 0.640051 | 0.525853 | 0.582952 | -0.018648 | 64.7s | severe regression |
| S1-hard | 1 | top-half current-score hard | 42 | 0.667094 | 0.535035 | 0.601065 | -0.000535 | 82.8s | regression |
| S1-combo | 5 | top-half current-score hard | 42 | 0.669679 | 0.536659 | 0.603169 | +0.001569 | 198.1s | best altered axes combined; below control |

All runs completed under six minutes. The combo follows the requested staged
selection: five was the better altered negative count, and hard was the better
non-uniform policy. No altered cell beats the control, so no altered result is
claimed as a real gain and none requires three-seed confirmation. Increasing
uniform pairs gives a small loss despite much higher runtime; popularity
over-concentrates the within-user draws, while online hard mining destabilizes
early training and never recovers the control's ranking quality. **Retain one
uniform negative per positive.**

## Sweep 2 — auxiliary task set

All supported singles use weight 0.2. The requested `follow`, `comment`, and
`forward` outcomes are absent from both workspace NPZ and CSV exports; they are
recorded as failed/unavailable cells rather than silently skipped or fabricated.
The best-2 set selects the two strongest supported singles, click and like; the
best-3 set necessarily adds play-time fraction. Weight checks apply the same
weight to every task in the selected best-2 set.

| cell | auxiliary set | per-task weight | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | runtime | result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| S2-click | click BCE | 0.2 | 42 | 0.671627 | 0.537673 | **0.604650** | +0.003050 | 61.8s | best supported single |
| S2-like | like BCE | 0.2 | 42 | 0.671615 | 0.537495 | 0.604555 | +0.002955 | 64.0s | second supported single |
| S2-follow | follow BCE | 0.2 | - | - | - | - | - | - | unavailable: no exported target |
| S2-comment | comment BCE | 0.2 | - | - | - | - | - | - | unavailable: no exported target |
| S2-forward | forward BCE | 0.2 | - | - | - | - | - | - | unavailable: no exported target |
| S2-play | play-time fraction MSE | 0.2 | 42 | 0.669636 | 0.536892 | 0.603264 | +0.001664 | 66.1s | regression |
| S2-best2 | click + like | 0.2 | 42 | 0.671287 | 0.537704 | 0.604495 | +0.002895 | 54.7s | below either single |
| S2-best3 | click + like + play fraction | 0.2 | 42 | 0.669981 | 0.537241 | 0.603611 | +0.002011 | 77.7s | play target hurts |
| S2-w01 | click + like | 0.1 | 42 | 0.671878 | 0.538067 | **0.604973** | +0.003373 | 65.3s | best auxiliary cell |
| S2-w03 | click + like | 0.3 | 42 | 0.670016 | 0.537275 | 0.603645 | +0.002045 | 63.5s | excessive auxiliary pressure |

Every executable cell completed under six minutes. Although several seed-42
cells exceed the fixed 0.6016 baseline by 0.002, none beats the same-seed strong
control (0.605425), so they are exploration results rather than claimed real
gains and do not trigger redundant confirmation. The mild click+like setting at
0.1 is the best auxiliary variant but remains 0.000453 below control. Auxiliary
supervision is therefore unnecessary with the strong package; if it is desired
for representation sharing, keep click+like weak at 0.1 and avoid play-fraction
MSE and weights of 0.3.

## Sweep 3 — sparse-embedding optimizer

The strong architecture, regularization dropout, uniform BPR sampling, epoch
schedule, and no-aux objective are fixed. Unspecified learning rates retain the
control `1e-3`; the split optimizer uses Adagrad `0.05` for embeddings and Adam
`1e-3` for dense parameters. SparseAdam likewise applies only to sparse
embeddings, with Adam for dense parameters. Momentum is 0.9.

| cell | optimizer | learning rate(s) | seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | runtime | result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| S3-control | AdamW | 0.001 | 42 | 0.672499 | 0.538352 | **0.605425** | +0.003825 | 30.8s | sweep control |
| S3-ada01 | Adagrad | 0.01 | 42 | 0.672560 | 0.538145 | **0.605352** | +0.003752 | 49.3s | statistically tied at seed 42; -0.000073 |
| S3-ada05 | Adagrad | 0.05 | 42 | 0.664130 | 0.534290 | 0.599210 | -0.002390 | 31.0s | unstable/overfit |
| S3-sparse | SparseAdam embedding + Adam dense | 0.001 / 0.001 | 42 | 0.672270 | 0.538211 | 0.605241 | +0.003641 | 59.4s | close, but below control |
| S3-sgd | SGD + momentum 0.9 | 0.001 | 42 | 0.490456 | 0.460767 | 0.475612 | -0.125988 | 74.7s | does not converge at controlled LR |
| S3-split | Adagrad embedding + Adam dense | 0.05 / 0.001 | 42 | 0.669902 | 0.537012 | 0.603457 | +0.001857 | 38.3s | early peak then overfit |

All cells complete under six minutes. No candidate exceeds the same-seed
control, so no optimizer is claimed as a new gain and no candidate requires
three-seed confirmation. Adagrad 0.01 and sparse-Adam are competitive but add
runtime or implementation complexity without improving PRIMARY. Adagrad 0.05
and the split 0.05 embedding rate optimize training loss while validation
ranking deteriorates; controlled-LR SGD is unusably slow. **Retain AdamW.**

## Final summary and per-sweep conclusions

All three requested sweeps are complete on validation only, every executable
cell was selected by official PRIMARY, every predictions CSV was independently
rescored with the official evaluator, and every run finished below the
six-minute limit. The maximum was 198.1s for five online hard negatives.

| sweep | best requested altered cell (seed 42) | primary | control | conclusion |
|---|---|---:|---:|---|
| BPR sampling | 5 uniform negatives | 0.604880 | 0.605425 | one uniform negative is better and 4x faster |
| auxiliary tasks | click + like, weight 0.1 each | 0.604973 | 0.605425 | weak auxiliaries are tolerable; no auxiliary is best |
| optimizer | Adagrad 0.01 | 0.605352 | 0.605425 | essentially tied on one seed; AdamW remains simpler/faster |

The DIMS implementation confirms the requested strong L0 control at **0.604756
+/- 0.000487** across seeds 42/43/44, a mean gain of **+0.003156** over 0.6016.
No altered cell beats its same-seed 0.605425 control, so the verification rule
correctly produces no additional three-seed acceptance claims. The overall
recommendation is unchanged and now covers these previously under-swept axes:
five-field strong-regularized DCN-lite, one uniform within-user negative per
positive, no auxiliary loss, and AdamW at 1e-3 with the strong step schedule.

The unavailable follow/comment/forward targets are the only non-executable
cells, explicitly logged above. They require a new labeled export, not a model
or runner change. No test split was read or scored.

## Bonus — abandoned E6 Optuna cell

After closing all required DIMS work, the existing `zoo/final_optuna.py` E6
runner was executed for its intended 25 seeded TPE trials. `EXPERIMENTS_FINAL.md`
was not edited. The script's isolated PEP 723 environment omitted its Torch
dependency and failed before creating a study; the successful unchanged-runner
invocation overlaid Optuna on the working project environment:

```bash
uv run --with optuna python zoo/final_optuna.py --data-dir real \
  --out-dir /tmp/dims_bonus_e6 --seed 42 --trials 25
```

The best search trial scored 0.605278 at seed 42 with learning rate
0.000376406, weight decay 0.000071282, dropout 0.304314, embedding width 12,
BPR weight 0.513771, batch size 8192, and no recency weighting. The runner then
performed its full seed confirmation:

| seed | GAUC | nDCG@5 | primary | gate `(b,size,depth)` | runtime |
|---:|---:|---:|---:|---|---:|
| 42 | 0.672276 | 0.538295 | 0.605285 | (1,-1,1) | 86.1s |
| 43 | 0.671611 | 0.537529 | 0.604570 | (1,0,0) | 102.8s |
| 44 | 0.672527 | 0.538166 | 0.605347 | (1,0,0) | 101.6s |
| **mean +/- population std** | **0.672138 +/- 0.000386** | **0.537997 +/- 0.000335** | **0.605067 +/- 0.000352** | - | - |

The confirmed mean is **+0.003467 over 0.6016**, so E6 passes the specified
absolute-baseline acceptance rule. It is also 0.000312 above the DIMS strong
control mean, a small directional improvement rather than a decisive controlled
win. As in E4, the 27-point specialist gate is fit on validation and the Optuna
objective repeatedly selects on that same validation set, so 0.605067 is an
optimistically selected validation result and requires a fresh holdout before
deployment. All 25 search trials and three confirmations stayed under six
minutes; the longest combined specialist run was 102.8s.
