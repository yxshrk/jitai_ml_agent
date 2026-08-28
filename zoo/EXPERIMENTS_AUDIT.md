# Validation-only fresh-eyes audit campaign

## Protocol and safety

- Inputs are only `data/real_ws/train.{npz,csv}` (through 2022-04-21),
  `data/real_ws/val.{npz,csv}` (through 2022-04-28), the permitted static user
  side table, and raw standard-log rows filtered to those same impressions.
  The loader rejects a validation date after 2022-04-28 and never names or opens
  a test-split artifact.
- Every score and half-epoch checkpoint uses `data/official/evaluate.py`.
  Each final `predictions.csv` was separately rescored with that evaluator and
  matched `metrics.json` exactly.
- Seed 42 explores. Any result at least +0.002 over the fixed 0.6016 baseline is
  confirmed at seeds 43 and 44. A conclusion distinguishes that fixed-baseline
  rule from the harder question of improvement over the ~0.604 control.
- Adam uses learning rate 1e-3 and batch size 8192. Each run is deterministic and
  remained below six minutes (maximum observed runtime 131.9 seconds).

## A0 — corrected DCN-lite control

Exactly the five offset-encoded NPZ fields, embedding dimension 16, one
DCNv2-style cross layer, MLP width 128 with dropout 0.1, and hybrid
`0.5 * within-user BPR + 0.5 * logloss`. Validation happens every half epoch;
the saved checkpoint maximizes official **primary**, not GAUC. `metrics.json`
contains the complete half-epoch `history`.

| seed | GAUC | nDCG@5 | primary | delta vs 0.6016 | selected half | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0.670715 | 0.537216 | **0.603965** | +0.002365 | 2.5 | reproduced ~0.604 |
| 43 | 0.670167 | 0.536952 | 0.603559 | +0.001959 | 2.0 | confirmation |
| 44 | 0.670114 | 0.537097 | 0.603605 | +0.002005 | 3.0 | confirmation |

Three-seed mean primary: **0.603710**. Seed 42 fell from 0.603965 at half 2.5
to 0.590790 at half 5.5, demonstrating why primary-based selection matters.

## A1 — user-uniform stochastic dnDCG@5 LambdaRank

For every optimization batch, eligible users are sampled uniformly. Each user
contributes a without-replacement group of 3–8 of their own training impressions.
Opposite-label pairs receive `|delta nDCG@5|` from swapping their current predicted
ranks. The weighted loss is normalized per user, averaged across users, and blended
with the unchanged control hybrid at the requested weights.

| lambda mix | seed | GAUC | nDCG@5 | primary | delta vs s42 control | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0.3 | 42 | 0.666161 | 0.534986 | 0.600574 | -0.003392 | no-win |
| 0.5 | 42 | 0.662007 | 0.533753 | 0.597880 | -0.006086 | no-win |

Both doses are below the fixed promotion bar, so confirmation is not run.

## A2 — duration-regime heads

The DCN/MLP representation is shared. Separate linear ranking heads are gated by
`duration_ms <= 18000`; the optional variant adds a learned output bias for each
`(tab, duration regime)`.

| variant | seed | GAUC | nDCG@5 | primary | delta vs s42 control | verdict |
|---|---:|---:|---:|---:|---:|---|
| two heads | 42 | 0.667442 | 0.535780 | 0.601611 | -0.002354 | no-win |
| two heads + tab bias | 42 | 0.667440 | 0.535807 | 0.601623 | -0.002342 | no-win |

Both are materially worse than control and below the promotion bar.

## A3 — coarse user-metadata crosses

Only `user_active_degree`, `follow_user_num_range`, `fans_user_num_range`,
`register_days_range`, and `is_video_author` are read from
`user_features_pure.csv`. Each is crossed separately with duration regime, tab,
and a coarse author bucket `floor(log2(train exposure count))` clipped at 15.
There is no raw high-cardinality user field or author identity cross.

| seed | GAUC | nDCG@5 | primary | delta vs same-seed control | verdict |
|---:|---:|---:|---:|---:|---|
| 42 | 0.670886 | 0.537006 | 0.603946 | -0.000019 | explore: tie; fixed-bar confirm |
| 43 | 0.668983 | 0.536256 | 0.602620 | -0.000940 | confirmation failed |
| 44 | 0.670126 | 0.536702 | 0.603414 | -0.000191 | confirmation failed |

Three-seed mean primary is 0.603327 (+0.001727 over 0.6016), so the apparent
seed-42 promotion does not confirm and the feature set is rejected.

## A4 — causal session features

Raw `time_ms` is aligned to exported rows by `(user_id, video_id, date, hourmin)`
in occurrence order. Because the export preserves raw source order, occurrence
order is the explicit fallback when duplicate user/video/time keys are ambiguous.
After alignment, train and validation impressions are stably sorted by
`(user_id, time_ms, source row)`. A row sees only its own timestamp and earlier
impressions for that user, including earlier validation impressions. A gap over
30 minutes starts a new session. The three added categorical fields are previous
exposure-gap bucket, within-session index clipped at 31, and session-start flag.

| seed | GAUC | nDCG@5 | primary | delta vs same-seed control | verdict |
|---:|---:|---:|---:|---:|---|
| 42 | 0.671029 | 0.537671 | **0.604350** | +0.000385 | fixed-bar promote |
| 43 | 0.670101 | 0.536902 | 0.603502 | -0.000057 | no control lift |
| 44 | 0.671014 | 0.537271 | 0.604142 | +0.000537 | tiny control lift |

Three-seed mean primary is **0.603998**, +0.002398 over 0.6016 and therefore a
confirmed fixed-baseline win. Its mean lift over the control is only +0.000288,
so it is not evidence of a material breakthrough over the DCN plateau.

## A5 — best combination

Session fields were the only positive seed-42 delta. Metadata crosses were the
only nearly neutral complement, so an additional diagnostic tried those two and
excluded the clearly harmful LambdaRank and duration-head branches. Metadata had
not confirmed, so this diagnostic cannot displace the strict winners-only
combination (session fields alone) unless it improves the score.

| config | seed | GAUC | nDCG@5 | primary | delta vs s42 control | verdict |
|---|---:|---:|---:|---:|---:|---|
| session + metadata | 42 | 0.670011 | 0.537110 | 0.603560 | -0.000405 | no-win |
| session only (`audit_best.py`) | 42 | 0.671029 | 0.537671 | **0.604350** | +0.000385 | final winner |

The combination is worse than session-only and just below the fixed promotion
bar, so no confirmation is run. The strict winning combination is therefore
session-only; its seed-43/44 confirmation numbers are the identical A4 runs and
are reused rather than recomputed.

## Summary and conclusions

The corrected control reproduces primary ~0.604 and exposes severe post-peak
overfit. LambdaRank and duration-specialized heads clearly hurt. Coarse metadata
crosses tie on seed 42 but fail confirmation. Causal session context is the only
positive seed-42 change and clears the fixed-baseline rule after confirmation,
but its +0.000288 three-seed lift over control is within noise. Combining it with
metadata is worse.

`zoo/audit_best.py` therefore freezes the numerically best configuration:
five-field DCN-lite plus the three causal session fields, one cross layer,
MLP128, and 0.5 BPR + 0.5 logloss, selected on primary every half epoch. Its
standalone seed-42 contract run scores GAUC 0.671029, nDCG@5 0.537671, primary
**0.604350**, and emits an 11-entry `history`. This beats 0.6041 on the exploration
seed, but the campaign found no confirmed material improvement over the control.
