# Zoo experiment log — 2026-08-28 sweep (REAL KuaiRand-Pure, VALIDATION ONLY)

Protocol: seed 42 for exploration; any config claiming acceptance (delta >= 0.002
over baseline valid primary **0.6016**, baseline seed std 0.0008) confirmed with
seeds 43,44 and reported as mean +- std over 3 seeds. All scores via the vendored
official evaluator (data/official/evaluate.py). Test split never evaluated.
Runner: `uv run python zoo/dcn_feats.py --data-dir real --out-dir <o> --seed <s> [flags]`.
Unless stated: k=16, 2 cross layers, hidden=64, bpr-weight 0.5, lr 1e-3, batch 8192,
early stop on valid GAUC (patience 3). Prior zoo results are in zoo/RESULTS.md.

## E1 — DCN-lite + fm_feats features merged (MENU #3+#4+#9)

| config | seed | gauc | ndcg5 | primary | delta | runtime | verdict |
|---|---|---|---|---|---|---|---|
| dcn_feats defaults | 42 | 0.6719 | 0.5372 | 0.6045 | +0.0029 | 22s | promising (>= eps, pending 3-seed) |

Beats both parents (dcn_lite 0.6041, fm_feats 0.6024) at seed 42. All models still
peak at epochs 1-2 and overfit fast.

## E2 — architecture sweep (seed 42, from E1)

| config | gauc | ndcg5 | primary | delta | runtime | verdict |
|---|---|---|---|---|---|---|
| cross=1 | 0.6716 | 0.5376 | 0.6046 | +0.0030 | 20s | tie (noise vs E1) |
| cross=3 | 0.6716 | 0.5371 | 0.6044 | +0.0028 | 28s | tie |
| k=32 | 0.6710 | 0.5367 | 0.6039 | +0.0023 | 35s | worse |
| hidden=128 | 0.6721 | 0.5373 | **0.6047** | +0.0031 | 27s | best of sweep |
| cross=1, hidden=128 | 0.6714 | 0.5373 | 0.6044 | +0.0028 | 23s | tie |
| hidden=256 | 0.6713 | 0.5370 | 0.6042 | +0.0026 | 39s | worse |
| cross=1, hidden=256 | 0.6714 | 0.5374 | 0.6044 | +0.0028 | 39s | tie |

Spread across the whole sweep is 0.0008 — arch choice barely matters. Kept
cross=2, hidden=128, k=16.

## E3 — MTL aux heads (click + effective-view proxy) on E2 best (seed 42)

| aux weight | gauc | ndcg5 | primary | delta | runtime | verdict |
|---|---|---|---|---|---|---|
| 0.1 | 0.6720 | 0.5376 | **0.6048** | +0.0032 | 36s | best overall seed-42 run |
| 0.2 | 0.6718 | 0.5375 | 0.6046 | +0.0030 | 40s | tie |
| 0.3 | 0.6717 | 0.5375 | 0.6046 | +0.0030 | 41s | tie |

## E4 — Tier-3 item-side aggregates (train-window Bayesian video/author long_view
rates, prior strength 20, 20 quantile buckets) on hidden=128 (seed 42)

| config | gauc | ndcg5 | primary | delta | runtime | verdict |
|---|---|---|---|---|---|---|
| --item-agg | 0.6709 | 0.5367 | 0.6038 | +0.0022 | 47s | **worse** than same config without (0.6047) — branch dead |

Likely the id embeddings already carry this signal; the bucketed rate adds noise.

## E5 — BPR/logloss mix on hidden=128 + aux 0.1 (seed 42)

| bpr weight | gauc | ndcg5 | primary | delta | verdict |
|---|---|---|---|---|---|
| 0.0 (pure logloss) | 0.6705 | 0.5372 | 0.6038 | +0.0022 | worse |
| 0.3 | 0.6713 | 0.5376 | 0.6045 | +0.0029 | slightly worse |
| **0.5** | 0.6720 | 0.5376 | **0.6048** | +0.0032 | best (= E3 row) |
| 0.7 | 0.6705 | 0.5367 | 0.6036 | +0.0020 | worse |
| 1.0 (pure BPR) | 0.6706 | 0.5367 | 0.6036 | +0.0020 | worse |

0.5/0.5 hybrid confirmed optimal. Listwise per-user softmax loss: not attempted
(time spent on E6/E8 instead) — noted as unfinished.

## Confirmation — 3-seed runs of the two finalists

best = dcn_feats --hidden 128 --aux-weight 0.1 (cross=2, k=16, bpr 0.5):

| config | s42 | s43 | s44 | mean +- std | delta | verdict |
|---|---|---|---|---|---|---|
| hidden=128, aux 0.1 | 0.6048 | 0.6028 | 0.6042 | **0.6039 +- 0.0010** | **+0.0023** | **ACCEPTED** (>= eps 0.002) |
| hidden=128, no aux | 0.6047 | 0.6025 | 0.6041 | 0.6038 +- 0.0011 | +0.0022 | accepted, but aux 0.1 edges it |

Seed 43 is a consistently weak seed for both (~0.6027). Honest note: the two
finalists are statistically indistinguishable; aux 0.1 kept as the frozen best
(zoo/best.py). 3-seed GAUC for best: 0.6720/0.6697/0.6708; nDCG@5 0.5376/0.5360/0.5376.

## E6 — video content features (video_type, upload_type, music_id top-200, first tag)

Raw video_features_basic_pure.csv read directly inside the zoo script (data/
untouched — no loader edit was needed).

| config | seed | gauc | primary | delta | verdict |
|---|---|---|---|---|---|
| best + --content | 42 | 0.6708 | 0.6039 | +0.0023 | **worse** than best without (0.6048) — dead |

## E7 — seed ensemble of the best config (seeds 42-46, extra runs s45 0.6036, s46 0.6040)

| ensemble | gauc | ndcg5 | primary | verdict |
|---|---|---|---|---|
| mean of 5 seeds' scores | 0.6715 | 0.5377 | 0.6046 | stabilizer |
| rank-average of 5 seeds | 0.6716 | 0.5378 | **0.6047** | best ensemble |
| mean of 3 seeds (42-44) | 0.6713 | 0.5374 | 0.6043 | — |

Ensembling lifts the *expected* score from the seed mean 0.6039 to 0.6047 (about
the best single seed) — a variance reducer, not a level gain. Deterministic
single-artifact zoo/best.py stays the deliverable; ensemble is a harness-level
option for the final submission.

## E8 — LightGBM lambdarank + rank blend (via `uv run --with lightgbm`, pyproject untouched)

Features: encoded video/author/tab/dur-bucket ids (categorical), duration, <=18s
flag, hour, smoothed video/author train-window rates. Grouped by user, lambdarank,
400 trees, lr 0.05, 63 leaves, seed 42.

| model | gauc | ndcg5 | primary | verdict |
|---|---|---|---|---|
| LightGBM alone | 0.6621 | 0.5328 | 0.5974 | below baseline |
| rank blend 0.8 NN-ens + 0.2 LGBM | 0.6699 | 0.5368 | 0.6034 | worse than NN alone |
| blend 0.7/0.3 | 0.6695 | 0.5362 | 0.6029 | worse |
| blend 0.5/0.5 | 0.6677 | 0.5355 | 0.6016 | worse |

Branch dead: the tree model is too weak to add diversity value.

## Final summary

**Winner (zoo/best.py): DCN-lite + fm_feats features, k=16, 2 cross layers,
hidden=128, aux heads (click + effective-view) at 0.1, 0.5 BPR + 0.5 logloss,
GAUC early stopping. Valid primary 0.6039 +- 0.0010 (seeds 42/43/44),
delta +0.0023 vs 0.6016 baseline — ACCEPTED.** 5-seed rank ensemble: 0.6047.

Target 0.610 not reached: every capacity/feature/loss branch beyond the winner
plateaued at 0.604-0.605 seed-42 / ~0.604 seed-mean; models overfit by epoch 2-3.
Promising but unfinished: listwise per-user softmax loss (E5, skipped), stronger
regularization / lr decay to survive past epoch 2, CWM-style censored watch-time
loss (MENU #13), and user-history sequence features.
