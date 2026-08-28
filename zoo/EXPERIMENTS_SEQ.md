# Sequence-model campaign

## Protocol and leakage guard

Validation only: train dates 2022-04-08..21 and validation dates
2022-04-22..28. All primary metrics are produced by
`data/official/evaluate.py`. Exploration uses seed 42. A seed-42 score at or
above 0.605700 triggers confirmation at seeds 42/43/44; a real win requires the
three-seed mean to be at least 0.605700 (+0.001 over 0.604700).

For each user, train rows are stable-sorted by `(date, hourmin, original row)`.
All targets at the same `(date, hourmin)` receive the history snapshot from
before that timestamp, then the whole tied group is appended. Consequently no
target sees its own, a same-timestamp, or a future outcome. Every validation row
receives the same immutable snapshot of that user's full train history, and
validation labels are never inputs. `tests/test_seq.py` and the full-cache
`assert_no_leakage` scan are run before every measured cell.

## Seed-42 exploration

Common recipe unless noted: batch 4096, 4 epochs, half-epoch official-metric
selection, 7-day recency weights, 0.5 BPR + 0.5 logloss, AdamW (lr 0.001,
weight decay 0.001), LR multiplied by 0.5 every half epoch, dropout 0.25, two
attention heads, and a 470-second process cap. Runtime includes cache loading,
training, checkpoint evaluation, and artifact writing.

| cell | encoder | outcome marks | history | blocks | k | seed | primary | runtime | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | SASRec | no | 50 | 1 | 32 | 42 | **0.581807875** | 260.6s | plain specified model; best epoch 4.0; leakage tests passed |
| B | SASRec | yes | 50 | 1 | 32 | 42 | **0.557692089** | 289.4s | past-outcome ablation; best epoch 4.0; substantially worse; leakage tests passed |
| C | SASRec | no | 20 | 1 | 32 | 42 | **0.580360171** | 111.3s | history-length ablation; best epoch 4.0; 50 beats 20; leakage tests passed |
| D1 | SASRec | no | 50 | 1 | 16 | 42 | **0.579563842** | 226.2s | capacity grid; best epoch 4.0; leakage tests passed |
| D2 | SASRec | no | 50 | 2 | 16 | 42 | **0.567453331** | 353.2s | capacity grid; best epoch 4.0; extra block hurts; leakage tests passed |
| D3 | SASRec | no | 50 | 2 | 32 | 42 | **0.566947572** | 453.6s | capacity grid; best epoch 4.0; under 8-minute cap; leakage tests passed |
| E | GRU | no | 50 | 1 | 32 | 42 | **0.581266900** | 257.5s | fallback; best epoch 4.0; behind one-block SASRec; leakage tests passed |

## Confirmation runs

No seed-42 configuration reached 0.605700, so the mandatory three-seed trigger
did not fire. There is no multi-seed claim and no confirmed win.

## Ensemble and segment results

The frozen member is the exact seed-42 reproduction at
`logs/run_rehearsal5/node_001/predictions.csv` (official primary 0.604998355).

| model | GAUC | nDCG@5 | primary | runtime | result |
|---|---:|---:|---:|---:|---|
| best SASRec A | 0.640387988 | 0.523227762 | **0.581807875** | 260.6s | best sequence cell |
| frozen stack seed 42 | 0.671858967 | 0.538137743 | **0.604998355** | pre-existing | reference |
| F: per-user rank average | 0.660249536 | 0.533239187 | **0.596744362** | 2.9s | worse than frozen; leakage tests passed |

History groups are split at the median train-history count among validation
users: LOW has at most 35 train impressions and HIGH has more than 35.

| model | LOW primary (11,366 users / 43,863 rows) | HIGH primary (11,011 users / 81,046 rows) | high-minus-low |
|---|---:|---:|---:|
| best SASRec A | **0.587832284** | **0.576190558** | -0.011641726 |
| frozen stack | **0.606722121** | **0.603218321** | -0.003503801 |
| rank ensemble F | **0.599995122** | **0.593563306** | -0.006431816 |

The frozen stack is indeed weaker on HIGH than LOW, but SASRec does not show the
expected specialist advantage: it trails frozen by 0.018889837 on LOW and by a
larger 0.027027763 on HIGH. The ensemble also loses more on HIGH.

### Exact ensemble recipe

First reproduce the selected sequence model, then rank-average it with the
frozen validation predictions:

```bash
uv run python zoo/seq_best.py --out-dir /tmp/seq-best --seed 42
uv run python zoo/seq_best.py --out-dir /tmp/seq-ensemble \
  --seq-scores /tmp/seq-best/scores.npy \
  --frozen-predictions logs/run_rehearsal5/node_001/predictions.csv
```

For each user and each member independently, stable ordinal ranks are scaled to
`[0,1]`; the two scaled ranks are averaged 50/50. No raw-score or global-rank
averaging is used. `metrics.json` includes official metrics and the history split.

## Implementation note

An initial history-20 attempt exposed NaN gradients in PyTorch's MPS fused
attention when both causal and redundant key-padding masks were supplied. It
was stopped and is not a measured cell. Since sequences are left-compacted and
right-padded, the causal mask alone prevents every selected valid state from
seeing padding. Removing the redundant mask fixed the issue; the repeated C
cell had finite losses throughout and passed all leakage tests.

## Final summary

Best sequence primary is **0.581807875** (one-block SASRec, k=32, history 50,
no outcome marks). The required diversity ensemble scores **0.596744362**, below
the frozen seed-42 stack at 0.604998355. No candidate qualified for multi-seed
confirmation, therefore **NO CONFIRMED WIN** was achieved. Longer history helps
slightly (50 beats 20), but the segment audit rejects the hoped-for HIGH-history
specialist effect: sequence underperformance is larger on HIGH-history users.
