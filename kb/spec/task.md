# Task specification (frozen)

Sources of truth, in priority order: `kuairand-starter-kit/evaluate.py` (scoring code), `baseline_scores.json`,
the starter-kit README, then `docs` §2.2–2.4. Where they disagree, `evaluate.py` wins — the README says so
explicitly. Known contradictions are listed in `corrections.md`.

## What is predicted
- Unit: one **impression** = one row of the standard log = one video shown to one user once.
- Label: `long_view` ∈ {0, 1}, a native column. Definition (kuairand.com):
  `long_view = 1  iff  play_time_ms >= min(duration_ms, 18000)` — verified on train + valid (97.8 % exact;
  the exceptions are rows with `duration_ms = 0`, always labelled 0, and a small completion tolerance —
  see `kb/data/facts.md` §3).
- Output: one real-valued `score` per row. Only the order **within each user** is used. NaN / Inf are rejected.

## Task form
**Within-user ranking over logged impressions.** For each user in the evaluation split, the model reorders the rows
that user actually has in that split. There is no retrieval and no candidate generation.

Consequence (a theorem, not a hypothesis): any score term that is constant across one user's rows cannot change
either metric. User-only features can act only through interactions with row-varying (item / context) features.

## Metrics — exactly as `evaluate.py` computes them
- **GAUC**: per-user AUC (Mann–Whitney U with average ranks for ties), computed only for users with
  `0 < #positives < #rows`, averaged with weight = #positives. If no user qualifies the value is 0.5.
- **nDCG@5**: per user, gain `2^rel − 1` (= rel for binary labels), discount `log2(position + 1)`, cutoff 5;
  users with zero positives score 0.0 and **are included**; unweighted mean over all users.
- **primary = (GAUC + nDCG@5) / 2** — the number that ranks submissions.
- Ties: AUC gives tied scores their average rank; nDCG sorts by `-score` with Python's stable sort, so tied rows
  keep file order. Do not emit ties.

## Data split — by date, fixed
| split | dates | rows | users |
|---|---|---|---|
| train | 20220408–20220421 (14 d) | 1,141,112 | — |
| valid | 20220422–20220428 (7 d) | 124,909 | — |
| test | 20220429–20220508 (10 d) | 170,588 | 23,875 |

Row order = read `log_standard_4_08_to_4_21_pure.csv`, then `log_standard_4_22_to_5_08_pure.csv`, filter by
date, keep file order. `row_id` is the 0-based index in that order. `(user_id, video_id)` is **not** unique
(3.06 % of test rows are repeated pairs, up to 12×).

## Columns
Log header (verified from the file):
`user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate,
long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab`

| known at show time → legal **features** of the row | outcomes → **never** features of the row being scored |
|---|---|
| `user_id, video_id, date, hourmin, time_ms, tab, duration_ms, is_rand` | `long_view, is_click, is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter` |
| side tables joined by id: `video_features_basic_pure.csv` (author_id, music_id, video_type, upload_type, upload date, tags …), `user_features_pure.csv` (user_active_degree, follow/fans/friend count ranges, register_days_range, anonymised one-hot attributes) | `video_features_statistic_pure.csv` — per-video counters averaged **over the whole month**, i.e. they include the test window → treat as leaky, do not use |

Outcome columns are legal as **training targets** (auxiliary tasks) and as **history features** built only from
rows strictly earlier in time than the row being scored.

## Submission
CSV `row_id,user_id,video_id,score`, one line per row of the split, `row_id` 0-based and contiguous.
Validate with `python3 submit.py --check --split test submission.csv` before anything is handed in.
