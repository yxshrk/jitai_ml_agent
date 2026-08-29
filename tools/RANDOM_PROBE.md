# Random-exposure transfer probe

## Verdict

The original five-seed ensemble `{46, 74, 93, 91, 60}` is the best available
ensemble on the test-window probe, with primary metric **0.381004285**. It was
the only ensemble evaluated because the optional coral late-selected seed set
was unavailable: `coral.local` could not be resolved, so
`~/techjam/ens_late.log` could not be retrieved. No planned local candidate was
skipped.

Seed 42, included as the requested single-model reference, is slightly higher
than the ensemble at **0.381385181** on the test-window probe. Thus the probe
supports the original set as the ensemble fallback, but does not show it beating
the single-model reference under random exposure.

## Results

`delta` is test-window primary minus val-window primary. All metrics use the
official evaluator's within-user GAUC and nDCG@5 conventions, with primary equal
to their mean.

| candidate | val-window primary | test-window primary | delta |
|---|---:|---:|---:|
| seed 46 | 0.369846506 | 0.380209655 | +0.010363149 |
| seed 74 | 0.369534939 | 0.380184540 | +0.010649601 |
| seed 93 | 0.369537588 | 0.380083653 | +0.010546065 |
| seed 91 | 0.368851004 | 0.379516053 | +0.010665049 |
| seed 60 | 0.369583224 | 0.379589237 | +0.010006013 |
| seed 42 (reference) | 0.370876037 | 0.381385181 | +0.010509144 |
| original ensemble `{46,74,93,91,60}` | 0.370191450 | 0.381004285 | +0.010812836 |

Absolute metric levels on this probe are **not comparable to the real hidden
test**, because the random exposure policy differs from the standard logging
policy. Only relative candidate ordering within this probe is meaningful. The
positive window deltas likewise describe these two random-log windows and should
not be interpreted as expected hidden-test improvement.

## Probe coverage

| window | inclusive dates | rows | unique users |
|---|---|---:|---:|
| val-window | 2022-04-22 through 2022-04-28 | 288,338 | 19,091 |
| test-window | 2022-04-29 through 2022-05-08 | 897,721 | 26,907 |

## Method and artifacts

This use is legal under the project's established split ruling:
`log_random_4_22_to_5_08_pure.csv` is not the hidden test set; the hidden test is
carved from `log_standard`. Random-log rows were never used for training or
checkpoint selection. They were used only for evaluation-only candidate
comparison. Every member was retrained fresh on the standard train split using
the exact `zoo/polish_stack.py` defaults, including standard-validation-only
checkpoint selection.

The probe rebuilds all five input vocabularies from `data/real_ws/train.csv` and
uses train-only duration deciles, matching `data/export_real_ws.py`. Unknown
probe values map to the train-derived unknown bucket. Ensemble scores are equal
per-user rank averages. The complete run took 415.7 seconds, below the 20-minute
budget.

Reproduce with:

```bash
uv run python tools/random_probe.py
```

The two NPZ files, six checkpoints, member scores, ensemble scores, and
`results.json` are written under `data/random_probe/`. That directory is
gitignored because the files are derived artifacts.
