---
id: model-lightgbm-lambdarank
family: model
target_component: model
source: Ke et al. 2017 "LightGBM" (NeurIPS); Burges 2010 "From RankNet to LambdaRank to LambdaMART" (MSR-TR-2010-82) — the lambdarank objective; standard winning recipe of tabular ranking competitions (KDD Cup, RecSys Challenge); allowed by rules.md (any open-source library), ADR-0014
applies_when:
  - the useful signal is heterogeneous, engineered features (target statistics, duration, session position, exposure counts) rather than id embeddings — facts §1, §3, §4, §10: tab, duration and session context dominate; user × author taste is unobservable for 96.6 % of valid rows
  - the metric is within-user ranking, so a per-query (per-user) listwise objective fits directly
  - the FM family has plateaued (three runs at 0.6044 with BPR + seed averaging) — capacity on the five ids is not the bottleneck
expected_delta: [0.001, 0.006]
expected_delta_basis: trees over engineered features usually beat factorization models on logs like this once context features exist, but raw high-cardinality ids (user_id 26 K values) are weak as tree splits, so the gain depends on the feature block (features-exposure-session, target statistics); a blend with the BPR ensemble is the safer target
cost: ~120 lines (feature builder + lightgbm.train with a callback that scores valid every 25 rounds); runtime 1–3 min at 4 threads (100 trees on 200 K × 20 measured at 0.5 s); library: lightgbm 4.6 (installed; linked to torch's libomp, see README setup)
composes_with: [features-exposure-session, ensembling-seed-average, ensembling-heterogeneous-rank-average, loss-bpr-pairwise-within-user, history-user-aggregates]
conflicts_with: [model-din-history-attention]
status: untried
evidence: []
---
## Claim
A gradient-boosted tree ranker (LightGBM, lambdarank objective, query = user) over engineered row features can
represent the non-linear tab × duration × session-context structure that a second-order FM cannot, and it is the
standard strong model on impression logs of this shape.

## Mechanism (why it moves within-user ranking)
lambdarank weights each mis-ordered pair by the nDCG it would recover, so the gradient targets the top of each
user's list — the nDCG half the FM never moved. Trees split on raw continuous features (duration_ms, position,
gap, smoothed rates), so no bucketing is needed and interactions of any order come for free. Personalisation enters
through per-user target statistics and, in the blend, the BPR FM's rank.

## How to implement on node_000
1. Keep node_000's loading; build a feature matrix per row: tab, duration_ms, log duration, dur = 0 flag, hour;
   smoothed long-view rates and counts for video, author, video × tab, author × tab from STRICTLY EARLIER train rows
   (Beta(2, 4) prior; for valid/extra rows use full-train totals); user's train count and rate; the session/exposure
   features of features-exposure-session as raw integers; optionally user_id and video_id as `categorical_feature`.
2. Sort train by user; `group` = rows per user; label = long_view; drop users with a single class (no pairs).
3. `lightgbm.train(params, Dataset(X, y, group=g), num_boost_round=min(600, SMOKE cap), callbacks=[cb])` with
   params: objective lambdarank, metric none, learning_rate 0.05, num_leaves 63, min_data_in_leaf 100,
   feature_fraction 0.8, bagging_fraction 0.8, bagging_freq 1, lambdarank_truncation_level 10, num_threads =
   int(os.environ.get('OMP_NUM_THREADS', 1)), seed = seed, deterministic = True, force_row_wise = True, verbose −1.
4. `cb` every 25 rounds: predict valid with `num_iteration = iteration + 1`, `evaluate()` → append to `history`
   (one entry per 25 rounds), keep the best iteration; predict the final valid and extra files at the best iteration.
5. `SMOKE_EPOCHS` caps rounds (`SMOKE_EPOCHS * 25`). Write predictions.csv / metrics.json per contract.
6. Second node (compose): rank-average the LightGBM score with the BPR seed-ensemble within user (0.5 / 0.5 first).

## Risks / failure modes
- Target statistics computed with the row's own label leak (a validation score far above 0.61 means leakage —
  scoring.md); use strictly earlier rows for train, full train for valid, never valid labels.
- user_id as a categorical feature overfits sparse users; prefer the per-user statistics, test the id both ways.
- LightGBM ignores rows of single-class users under lambdarank; ~7 % of train users — fine, but log it.
- Determinism: bagging + multithreading is reproducible only with deterministic = True and fixed num_threads.

## Measured
(none yet)
