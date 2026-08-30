# Experiment script contract

Every experiment is ONE self-contained Python script. The harness runs it as

    python <script>.py --data-dir data --out-dir <out> --seed <int> [--score-extra <csv>]

with the current directory = this `workspace/` folder. Available: Python 3.9, numpy. (Other libraries only if the
harness config lists them.) `from evaluate import evaluate` gives you the official scorer.

## Inputs (read only from `--data-dir`)
| file | columns | notes |
|---|---|---|
| `train.csv` | `user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow, is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms, profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab` | 1,141,112 rows, dates 20220409–20220421, file order |
| `valid.csv` | `row_id, user_id, video_id, date, hourmin, time_ms, tab, duration_ms, is_rand, long_view` | 124,909 rows, dates 20220422–20220428; `row_id` 0..N-1 in file order |
| `video_features_basic.csv` | `video_id, author_id, video_type, upload_dt, upload_type, visible_status, video_duration, server_width, server_height, music_id, music_type, tag` | 7,583 videos |
| `user_features.csv` | `user_id, user_active_degree, is_lowactive_period, is_live_streamer, is_video_author, follow_user_num(_range), fans_user_num(_range), friend_user_num(_range), register_days(_range), onehot_feat0..17` | 27,285 users |

Rules baked into the data:
- **Features** of a row = columns known when the video was shown: `user_id, video_id, date, hourmin, time_ms, tab, duration_ms` and anything joined from the two side tables.
- **Outcome columns** (`long_view, is_click, is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter`) exist only in `train.csv`. They are legal as **training targets** and as **history features built from earlier rows** (strictly earlier `time_ms`). They are never features of the row being scored — `valid.csv` does not have them, and the hidden test file will not either.
- `duration_ms = 0` means unknown length; such rows are always `long_view = 0`.
- There is no test data here. Do not look for it.

## Outputs (write only to `--out-dir`)
- `predictions.csv` — header `row_id,user_id,video_id,score`; exactly one line per `valid.csv` row, in the same
  order, copying `row_id,user_id,video_id`; `score` any finite real number (only the within-user order matters).
- `metrics.json` — `{"gauc", "ndcg5", "primary", "best_epoch", "history": [{"epoch", "train_loss", "val_gauc",
  "val_ndcg5", "val_primary"}, ...], "seed", "duration_s"}`. `history` is the per-epoch learning curve — required,
  it is how overfitting vs underfitting is diagnosed. Compute the numbers with `evaluate(user_ids, labels, scores)`.
- If `--score-extra <csv>` is given (a file with `row_id,user_id,video_id,date,hourmin,time_ms,tab,duration_ms,is_rand`),
  also write `predictions_extra.csv` with the same header as `predictions.csv`, one line per row of that file, in order.
  The harness uses this once, at the end, on the hidden test features. The model must be the one selected on valid.

## Behaviour
- Honour the environment variable `SMOKE_EPOCHS` (an integer): when set, cap every training phase at that many
  epochs. The harness smoke-tests every script with `SMOKE_EPOCHS=1` and a 120 s timeout before the real run.
- Deterministic given `--seed`. Full run must finish within 30 minutes on CPU.
- The harness re-scores `predictions.csv` itself with the official `evaluate.py`; your `metrics.json` is diagnostic.
