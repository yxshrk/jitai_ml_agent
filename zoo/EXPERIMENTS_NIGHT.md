# Overnight breadth campaign

## Protocol

CPU-only, validation-only on `data/real_ws` (and `data/real_ws_1k` only in the
bonus section), with no LLM calls. Every reported metric is produced by the
official `data/official/evaluate.py` implementation. Exploration uses seed 42.
Any observed primary delta at least +0.001 over 0.6047 is confirmed at seeds
42/43/44; confirmation summaries use population standard deviation.

The micro-grid center is the supplied coral wide-search configuration: lr
0.00066, step-decay factor 0.73 every 1.5 epochs beginning at epoch 1.0,
dropout 0.32, and otherwise the `polish_stack.py` defaults. The frozen
comparison is the literal unmodified `polish_stack.py` default configuration.

## 1. Coral micro-grid

Eight corners vary lr, dropout, and decay factor independently to 80% or 120%
of the coral center. All other coral/frozen settings are held fixed.

| cell | lr | dropout | decay factor | seed | primary | delta vs 0.6047 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| lr80-do80-df80 | 0.000528 | 0.256 | 0.584 | 42 | 0.605056484 | +0.000356484 | below confirmation threshold |
| lr80-do80-df120 | 0.000528 | 0.256 | 0.876 | 42 | 0.604786651 | +0.000086651 | effectively flat |
| lr80-do120-df80 | 0.000528 | 0.384 | 0.584 | 42 | 0.604609626 | -0.000090374 | no improvement |
| lr80-do120-df120 | 0.000528 | 0.384 | 0.876 | 42 | 0.604701818 | +0.000001818 | flat |
| lr120-do80-df80 | 0.000792 | 0.256 | 0.584 | 42 | 0.604879377 | +0.000179377 | below confirmation threshold |
| lr120-do80-df120 | 0.000792 | 0.256 | 0.876 | 42 | 0.604735115 | +0.000035115 | effectively flat |
| lr120-do120-df80 | 0.000792 | 0.384 | 0.584 | 42 | 0.604942385 | +0.000242385 | below confirmation threshold |
| lr120-do120-df120 | 0.000792 | 0.384 | 0.876 | 42 | 0.604845706 | +0.000145706 | below confirmation threshold |

Best corner: `lr80-do80-df80` at seed 42. Per the cell specification it is
three-seed confirmed below against a fresh three-seed frozen-default comparison,
despite its seed-42 delta not independently triggering the global +0.001 rule.

| confirmed config | seed 42 | seed 43 | seed 44 | mean +- pop. std | delta vs 0.6047 | verdict |
|---|---:|---:|---:|---:|---:|---|
| best corner, lr80-do80-df80 | 0.605056484 | 0.604805697 | 0.604515085 | 0.604792422 +- 0.000221225 | +0.000092422 | no confirmed win |
| frozen `polish_stack.py` defaults | 0.604998355 | 0.604250357 | 0.604730000 | 0.604659570 +- 0.000309403 | -0.000040430 | frozen comparison |
| coral center (diagnostic) | 0.604913128 | 0.604695329 | 0.604756442 | 0.604788300 +- 0.000091725 | +0.000088300 | grid-center diagnostic, not frozen comparator |

The best corner's confirmed mean exceeds the literal frozen-default mean by
only 0.000132851. This is noise, not a real improvement. The coral-center runs
were initially used as the comparator; the wording was re-audited before the
1K section and the correct literal-default comparison was run and logged rather
than silently relabeling the earlier diagnostic.

## 2. Popularity-prior blend

Source model: frozen-default seed 42, primary 0.604998355. Popularity is computed
from training-window video IDs only. Exposure uses total impression count;
long-view uses positive-label count. Both it and the model score are tie-aware
ranked within user before blending.

| variant | weight | seed | primary | delta vs 0.6047 | delta vs source | verdict |
|---|---:|---:|---:|---:|---:|---|
| exposure count | 0.1 | 42 | 0.604820802 | +0.000120802 | -0.000177552 | flat/no win |
| exposure count | 0.2 | 42 | 0.602855549 | -0.001844451 | -0.002142806 | clear regression |
| long-view count | 0.1 | 42 | 0.604875979 | +0.000175979 | -0.000122375 | flat/no win |
| long-view count | 0.2 | 42 | 0.603750696 | -0.000949304 | -0.001247659 | regression |

Verdict: no popularity-prior blend improves on its frozen-stack source, and no
cell reaches the +0.001 confirmation trigger. Kill this branch.

Audit-only earlier runs from the coral-center seed-42 artifact, before the
frozen/default wording correction: exposure 0.1 = 0.604828704, exposure 0.2 =
0.602434709, long-view 0.1 = 0.604898911, long-view 0.2 = 0.603373483. They
lead to the same no-win verdict and are not the headline cell results.

## 3. Per-(user, tab) rank recombine

Source model: frozen-default seed 42, primary 0.604998355. Scores are first ranked
within each `(user, tab)`. Tabs are then ordered lexicographically by their
training-window long-view rate, retaining model order inside each tab.

| cell | seed | primary | delta vs 0.6047 | delta vs source | verdict |
|---|---:|---:|---:|---:|---:|---|
| per-(user, tab) rank recombine | 42 | 0.601384421 | -0.003315579 | -0.003613934 | clear regression; killed after one cell |

Audit-only earlier run from the coral-center seed-42 artifact: 0.601203284. It
also clearly regressed and is not the headline cell result.

## 1K bonus readiness

Frozen means the literal `polish_stack.py` defaults (five fields, recency
half-life 7). Operational max-runtime is raised to 500 seconds so the roughly
six-minute CPU jobs can complete; model/training hyperparameters are unchanged.

| cell | half-life | seed | primary | best epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---|
| frozen default | 7 | 42 | 0.613417255 | 1.0 | 249.5s | first readiness member |
| frozen default | 7 | 43 | 0.609026923 | 0.5 | 229.6s | readiness member; sizable seed spread |
| frozen default | 7 | 44 | 0.615573858 | 0.5 | 235.9s | readiness member |
| recency probe | 3 | 42 | 0.612045886 | 1.0 | 268.3s | -0.001371368 vs seed-42 half-life 7; hurts |
| recency probe | 14 | 42 | 0.612509017 | 1.0 | 266.7s | -0.000908238 vs seed-42 half-life 7; hurts |

Frozen-default 1K seeds 42/43/44: **0.612672679 +- 0.002724137**
(population standard deviation). The early 0.5--1.0 epoch checkpoints dominate;
later checkpoints regress sharply, so half-epoch official selection is crucial.

Readiness verdict: retain half-life 7 for 1K submission prep. Neither requested
one-seed alternative beats the aligned seed-42 default. Because the 1K export
has a different validation population, its absolute scores are not deltas
against the full-data 0.6047 threshold; the requested 1K default nevertheless
has its own three distinct seed readiness estimate above.

## Final verification

Every saved `scores.npy` from all headline runs and the explicitly retained
audit-only runs was reloaded and rescored directly with
`data.official.evaluate.evaluate`. All 31 artifact scores matched their recorded
`metrics.json` primary to less than 1e-12. The three confirmation groups were
also checked to contain the distinct ordered seeds 42, 43, and 44. Verification:
**PASS**.
