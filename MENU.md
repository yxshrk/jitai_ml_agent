# Improvement menu — the agent's ranked search space

Ground truth from the starter kit: baseline FM (k=16, logloss, Adam) over exactly 5
fields: user_id, video_id, author_id, tab, dur_bucket (10 train-quantile duration
buckets). Beating primary 0.5946 (test) / 0.6016 (valid) is the goal. Published
KuaiRand-Pure reference: CWM (KDD'24) reaches GAUC ~0.713-0.715 → target band 0.70+.
Research basis: ../research/*.md. Expected gains are estimates, not promises.

## BINDING CONSTRAINT (measured, run_real_01 + manual campaign E1-E8)
Every architecture/feature variant plateaus at valid primary ~0.604-0.605, because
ALL models overfit by epoch 2-3 (val GAUC peaks then falls). Single-lever changes
from that plateau land within noise and get rejected. Priorities that follow:
A. **Fight the overfit FIRST**: dropout on embeddings/MLP, weight decay / per-row
   embedding L2, lr decay schedules, label smoothing, smaller lr + more epochs —
   anything that lets training survive past epoch 3. UNEXPLORED and the most
   plausible route past 0.605.
B. **Unexplored objectives**: per-user listwise softmax loss; ordinal watch-ratio
   auxiliary (play_time/duration buckets, TPM-lite); CWM-style censored watch-time
   loss (one-sided regression on play_time truncated at duration).
C. **Compound hypotheses are allowed and encouraged** when single levers plateau:
   one coherent THEME per iteration (e.g. "regularization package: dropout 0.1 +
   weight decay 1e-5 + lr decay"), not one micro-knob.
D. Do NOT re-test dead branches: item-side aggregates, video content features,
   k=32, LightGBM blends, pure BPR or pure logloss (all measured worse, see below).

Ranked by expected gain per unit implementation risk:

## Tier 1 — do first
1. **Within-user pairwise loss (BPR)** on the same FM features. Build (pos, neg) pairs
   inside each user's training impressions; optimize sigmoid(s_pos - s_neg). Directly
   optimizes what GAUC measures. Hybrid 0.5*BPR + 0.5*logloss is the safe variant.
   Expect +0.005-0.015 GAUC. ~60 lines.
   MEASURED (valid): mix sweep {0,0.3,0.5,0.7,1.0} at seed 42 — 0.5/0.5 hybrid best (0.6048); pure BPR or pure logloss lose ~0.001 (EXPERIMENTS.md E5).

2. **Early stopping + model selection on validation GAUC** (baseline stops on epochs/
   logloss). Free correctness fix. Expect +0.002-0.005.
3. **Finer duration handling**: the label IS duration-defined (long_view = watched
   >= min(duration, 18s)). Add: 50 buckets instead of 10, plus a direct
   duration<=18s indicator field, plus dur_bucket x tab cross. Expect +0.003-0.01.

   MEASURED (valid, 3-seed): in the winning stack (with #4/#9): 0.6039 +- 0.0010; features are part of zoo/best.py.

## Tier 2 — model capacity
4. **DCNv2-lite / FinalMLP head** on the same embeddings (1-2 cross layers, small MLP).
   Architectures beyond that (xDeepFM/AutoInt) overfit at 1.4M rows - skip.
   Expect +0.003-0.01 over tuned FM.
   MEASURED (valid, 3-seed): DCN-lite + #3/#9 features + aux 0.1 = 0.6039 +- 0.0010, delta +0.0023 ACCEPTED — current best (zoo/best.py). hidden 128 > 64 ~ 256; cross layers 1-3 all within noise.

5. **Multi-task shared-bottom**: auxiliary heads for click, like, effective_view at
   loss weight 0.1-0.3 (labels from the log's other signal columns - as TARGETS only,
   never inputs). Expect +0.003-0.008. PLE/MMoE only if this works.
   MEASURED (valid, 3-seed): aux 0.1 on the DCN stack = 0.6039 +- 0.0010 vs 0.6038 +- 0.0011 without — tiny/tied but kept; aux 0.2/0.3 no better (seed 42).

6. **Embedding dim sweep 16->32 (+ per-row L2)**. Cheap. Expect +0.00-0.005.

   MEASURED (valid, seed 42): k=32 = 0.6039 vs k=16 = 0.6047 — 32 is worse; keep 16.

## Tier 3 — features
7. **Train-window item/author aggregate rates**: video's and author's long_view rate
   computed ONLY over train dates, smoothed (Bayesian prior), as a scalar feature.
   Item-side only - user-side rates cannot move a within-user metric. Leakage rule:
   aggregates must use train window only. Expect +0.003-0.01.
   MEASURED (valid, seed 42): bucketed smoothed rates (prior 20) HURT: 0.6038 vs 0.6047 without — dead (E4).

8. **Video-side content features** from video_features_basic (video_type, upload_type,
   music_id, tags first-tag). Expect +0.002-0.008.
   MEASURED (valid, seed 42): video_type/upload_type/music_id-top200/first-tag HURT: 0.6039 vs 0.6048 — dead (E6).

9. **Temporal context**: hour-of-day bucket from hourmin, day-of-week from date.
   Expect +0.001-0.005.

   MEASURED (valid, 3-seed): included in the winning stack (with #3); not ablated separately.

## Tier 4 — advanced / stretch
10. **Ordinal watch-ratio auxiliary (TPM-lite)**: predict play_time/duration ordinal
    buckets as an auxiliary task. Expect +0.002-0.008.
11. **LightGBM lambdarank** on target-encoded aggregates over all 12 signals
    (train-window only) as a parallel model; rank-average blend with the NN.
    Expect +0.002-0.006 from the blend.
    MEASURED (valid, seed 42): LGBM alone 0.5974 (below baseline); every rank blend with the NN hurts — dead (E8).

12. **Seed ensemble**: average predictions over 3-5 seeds of the best config.
    Free +0.002-0.005, do at the very end.
    MEASURED (valid): 5-seed rank-average of best.py = 0.6047 vs seed-mean 0.6039 — variance reducer, ~best-single-seed level (E7).

13. **CWM-style censored watch-time loss** (KDD'24) - the published SOTA idea.
    High risk/reward; only if iterations remain.

## Known traps (the agent must respect)
- video_features_statistic counters are FULL-PERIOD aggregates -> temporal leakage.
  Do not use as-is; only train-window recomputations are legal.
- Other feedback signals (click, like, play_time...) are OUTCOMES of the impression:
  usable as auxiliary TARGETS, never as input features.
- GAUC is within-user: user-constant features cannot help GAUC (they can still help
  nDCG ordering? No - also within-user). Spend features on item-side variation.
- Improvements < 0.002 on validation are within noise (baseline seed std 0.0008;
  official epsilon = 0.002). Acceptance rule: keep a change only if val primary
  improves >= 0.002, else revert.
