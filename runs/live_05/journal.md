# Run journal — live_05

## Summary
```json
{
 "run_id": "live_05",
 "stop_reason": "converged: 3 generations without a >= 0.001 cumulative rise of the champion fresh-seed mean (ADR-0012)",
 "generations": 4,
 "nodes": 16,
 "champion": 2,
 "champion_metrics": {
  "gauc": 0.6703073475190711,
  "ndcg5": 0.5369853914191499,
  "primary": 0.6036463694691105,
  "ndcg5_disc": 0.7234219277427735,
  "by_group": {
   "dur18-60s": {
    "rows": 33235,
    "gauc": 0.6445,
    "ndcg5": 0.4601,
    "primary": 0.5523
   },
   "dur60-180s": {
    "rows": 46565,
    "gauc": 0.6481,
    "ndcg5": 0.4874,
    "primary": 0.5677
   },
   "dur<18s": {
    "rows": 20101,
    "gauc": 0.7021,
    "ndcg5": 0.4054,
    "primary": 0.5538
   },
   "dur=0": {
    "rows": 2107,
    "gauc": 0.6512,
    "ndcg5": 0.0867,
    "primary": 0.3689
   },
   "dur>180s": {
    "rows": 22901,
    "gauc": 0.6321,
    "ndcg5": 0.3784,
    "primary": 0.5052
   },
   "tab=0": {
    "rows": 13726,
    "gauc": 0.5501,
    "ndcg5": 0.0625,
    "primary": 0.3063
   },
   "tab=1": {
    "rows": 92672,
    "gauc": 0.6202,
    "ndcg5": 0.5407,
    "primary": 0.5805
   },
   "tab=2": {
    "rows": 3834,
    "gauc": 0.6414,
    "ndcg5": 0.516,
    "primary": 0.5787
   },
   "tab=4": {
    "rows": 7877,
    "gauc": 0.5831,
    "ndcg5": 0.5477,
    "primary": 0.5654
   },
   "tab=6": {
    "rows": 5170,
    "gauc": 0.6416,
    "ndcg5": 0.1059,
    "primary": 0.3738
   }
  }
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00218,
 "top3_valid": [
  {
   "n": 15,
   "primary": 0.6041063837624581
  },
  {
   "n": 9,
   "primary": 0.6040371716770998
  },
  {
   "n": 12,
   "primary": 0.6039854360682793
  }
 ],
 "designated": 12,
 "final_ranking": [
  {
   "n": 12,
   "valid_primary": 0.6039854360682793,
   "fresh_seeds": [
    0.6035553784688616,
    0.6038930832904412,
    0.6035371886526912
   ],
   "accepted": false,
   "mean": 0.6036618834706647,
   "std": 0.00020043137192465713
  },
  {
   "n": 15,
   "valid_primary": 0.6041063837624581,
   "fresh_seeds": [
    0.6036108420918753,
    0.6033562758628401,
    0.6037692631098412
   ],
   "accepted": false,
   "mean": 0.6035787936881856,
   "std": 0.0002083505259909047
  },
  {
   "n": 9,
   "valid_primary": 0.6040371716770998,
   "fresh_seeds": [
    0.6035478025962604,
    0.6030966080689154,
    0.6037692631098412
   ],
   "accepted": false,
   "mean": 0.6034712245916724,
   "std": 0.0003428036524174605
  }
 ],
 "usage": {
  "calls": 54,
  "tokens_in": 1799295,
  "tokens_out": 165173,
  "cache_read": 946243,
  "cache_write": 0,
  "cost_usd": 9.693571500000003
 },
 "wall_clock_s": 2072.2,
 "champion_seed_mean": 0.60313,
 "best_single_seed": 0.6041063837624581,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": {
  "best_single_seed": 0.6036463694691105,
  "streak": 3,
  "converged_at_generation": 4,
  "champion_at_stop": 2
 },
 "official_rule_submission": {
  "node": 2,
  "generation": 4,
  "valid_primary": 0.6036463694691105,
  "fresh_seed_mean": 0.60313,
  "fresh_seeds": 3
 },
 "convergence_switch": "confirmed",
 "tokens": {
  "in_uncached": 1799295,
  "in_cached": 946243,
  "out": 165173
 },
 "interventions": 0,
 "k": 5,
 "k_later": 3,
 "eps": 0.002,
 "n_converge": 3,
 "iteration_unit": "node",
 "iterations_used": 16
}
```

## Iterations

### n=0 — node_000 (reproduce_baseline, parent None)
**Hypothesis:** Reproduce the official FM baseline under the harness contract.
**Method:** official FM · target `None` · expected Δ 0.0 (published valid primary 0.6016)
**Result:** GAUC 0.6671 · nDCG@5 0.5358 · primary 0.6015
**Diff:** `None` (None changed lines) · duration 14s · tokens in/out 0/0 · intervention: False

### n=1 — node_001 (explore, parent 0)
**Hypothesis:** Train one FM to answer absolute watch-threshold queries using auxiliary labels derived from play_time_ms, then score each impression at its required threshold min(duration_ms, 18000), providing denser supervision for the duration-dependent long-view mechanism.
**Method:** query-conditioned discrete watch-survival FM · target `aux-targets` · expected Δ 0.0012 (Data Fact #3 shows that long_view is determined by reaching a row-specific threshold and that completed plays create censoring, so discrete survival queries reuse each observed watch time without relying on leaky inference features, though the existing duration field limits the expected gain.)
**Result:** GAUC 0.6678 · nDCG@5 0.5352 · primary 0.6015 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6015048914055572, 'node_seeds': [0.600929068155176, 0.6005636033781725, 0.6008654327730905], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': -0.00067, 'se': 0.000235, 'z': -2.83, 'sigma_pooled': 0.000288, 'sigma_df': 4, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/001.patch` (30 changed lines) · duration 63s · tokens in/out 56140/9722 · intervention: False

### n=2 — node_002 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with same-user BPR pairs so training directly optimizes mixed-user ordering, with particular potential to improve weak top ordering for short and >180-second videos and GAUC in tab 4.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0015 (The card was accepted four times over the official FM with fresh-seed gains of +0.0011 to +0.0017, and Diagnosis identifies BPR as the next probe.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6036 · realized Δ +0.0022 · ACCEPTED · seed confirmation {'node_seed0': 0.6036463694691105, 'node_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00168, 'se': 0.000258, 'z': 6.49, 'sigma_pooled': 0.000316, 'sigma_df': 6, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/002.patch` (34 changed lines) · duration 18s · tokens in/out 57096/4607 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Raise embedding L2 modestly so rare-ID memorization is reduced near the sharp validation peak without removing the FM's personalization signal.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0009 (The champion falls from 0.6015 at epoch 7 to 0.5990 at epoch 11, and the card previously gained +0.0009 fresh-seed mean on the official FM with only a two-line change.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6025029206269998, 'node_seeds': [0.6028729444838677, 0.6017090117891591, 0.6020694898356386, 0.6027857969882158, 0.6026234157272281], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00096, 'se': 0.000277, 'z': 3.47, 'sigma_pooled': 0.000379, 'sigma_df': 10, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/003.patch` (2 changed lines) · duration 30s · tokens in/out 55520/2976 · intervention: False

### n=4 — node_004 (improve, parent 0)
**Hypothesis:** Add leakage-safe historical user rates by author, tab, and duration bucket so sparse users can rank row-varying contexts using preferences not captured robustly by ID embeddings alone.
**Method:** history-user-aggregates · target `history` · expected Δ 0.001 (The card achieved a confirmed +0.0010 fresh-seed gain on the official FM, while Data Fact #2 reports a median of 35 training interactions per user.)
**Result:** GAUC 0.6677 · nDCG@5 0.5361 · primary 0.6019 · realized Δ +0.0004 · rejected · seed confirmation {'node_seed0': 0.601864542119245, 'node_seeds': [0.602126753907549, 0.6016283378604528, 0.6029421188953756, 0.6014368724382404, 0.6021026589976413], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.0006, 'se': 0.000316, 'z': 1.89, 'sigma_pooled': 0.000433, 'sigma_df': 14, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/004.patch` (87 changed lines) · duration 32s · tokens in/out 58415/8374 · intervention: False

### n=5 — node_005 (retest, parent 0)
**Hypothesis:** Average normalized within-user ranks from five independently early-stopped pointwise FMs to cancel initialization noise and stabilize ordering around the narrow epoch-7 optimum.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0009 (The card measured +0.0009 on a BPR stack and lists a 0.001–0.003 family range; the current pointwise champion has a sharply peaked learning curve that makes independent checkpoint averaging relevant.)
**Result:** GAUC 0.6687 · nDCG@5 0.5364 · primary 0.6025 · realized Δ +0.0011 · rejected · seed confirmation {'node_seed0': 0.6025438020993364, 'node_seeds': [0.6025774129235256, 0.6020164043692549, 0.6024061905255802, 0.6022087742581226, 0.6022268377504006], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00084, 'se': 0.000293, 'z': 2.85, 'sigma_pooled': 0.000402, 'sigma_df': 18, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/005.patch` (65 changed lines) · duration 71s · tokens in/out 114767/17545 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_002; best 0.6031; tokens 447372/51848; 813s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 7 (0.6015), then falls steadily to 0.5990 by epoch 11; epoch 5 was already 0.6010.
Champion balance: GAUC 0.6671, nDCG@5 0.5358, and mixed-user nDCG@5 0.7214; top-list quality remains the weaker half.
Last generation contains no candidate nodes, so there are no GAUC/nDCG movements or by-group deltas to attribute.
Next probe: within-user BPR pairwise loss, targeting the loss component; Data Fact #7 shows ranking signal is concentrated in mixed-label users.
Validation-overfitting risk is currently low but nonzero: streak 0, with 0 accepted sub-0.002 wins; repeated selection around the sharp epoch-7 peak will increase risk.
Weakest actionable duration cohort is >180s (primary 0.4982, GAUC 0.6196, nDCG 0.3768); short videos have low nDCG (0.4048) despite strong GAUC (0.6946), indicating top-order weakness.
Among substantial tabs, tab=4 has weakest GAUC (0.5636); tab=0 and tab=6 have near-zero nDCG largely from all-negative users, so those values should not drive selection.
No node moved any group this generation; the deepen slot should test whether BPR improves top ordering for short and >180s videos, especially within tab=4.
_Plan for next generation:_ `{"note": "BPR delivered the only large confirmed gain, while stronger L2 independently added a confirmed +0.0010 through a different component; merge them before adding complexity.\nThe champion regressed on tab 6 and remains weak there, making the untried label-free same-author run feature a targeted exploration; leave the other three slots to the Selector.", "plan": [{"type": "merge", "merge_parents": [2, 3], "hypothesis": "Combine same-user BPR with embedding L2=1e-5 so metric-aligned pair training retains BPR's ordering gain while shrinking noisy rare-ID embeddings and reducing the sharp post-epoch-8 decline."}, {"type": "explore", "parent": "champion", "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and prev-same-author\u00d7tab categorical fields to recover within-tab exposure-sequence effects, especially the champion's tab-6 regression, without using any evaluation-window outcomes."}]}`

### n=6 — node_006 (explore, parent 2)
**Hypothesis:** Augment ordinary same-user BPR with same-user, same-tab positive-negative pairs so the model cannot satisfy most updates through the dominant tab prior and must learn finer item ordering within each tab.
**Method:** context-matched two-stream BPR · target `loss` · expected Δ 0.0007 (Data Fact #4 shows tab has an extreme 0.04–0.49 positive-rate spread, while the champion remains weak on tab 4 GAUC, making context-matched pairs a cheap way to redirect BPR from easy cross-tab comparisons toward unresolved within-tab ordering.)
**Result:** GAUC 0.6699 · nDCG@5 0.5368 · primary 0.6033 · realized Δ -0.0003 · rejected
**Diff:** `diffs/006.patch` (16 changed lines) · duration 16s · tokens in/out 57327/5228 · intervention: False

### n=7 — node_007 (merge, parent 2, merge of [2, 3])
**Hypothesis:** Combine node_002's same-user BPR with node_003's L2=1e-5 so rare-ID shrinkage preserves BPR's ordering gain while reducing its sharp post-epoch-8 overfit.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0006 (Node_003 confirmed a +0.00096 fresh-seed gain, and the diagnosis shows the BPR champion falling from 0.6036 at epoch 8 to 0.6006 at epoch 12, although prior BPR-stack evidence warrants a conservative estimate.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6037 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6036798758461768, 'node_seeds': [0.6034092499923741, 0.6034185207299905, 0.6035365750210787], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00033, 'se': 0.000314, 'z': 1.04, 'sigma_pooled': 0.000385, 'sigma_df': 20, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/007.patch` (2 changed lines) · duration 16s · tokens in/out 58780/2918 · intervention: False

### n=8 — node_008 (deepen, parent 2)
**Hypothesis:** Preserve the champion's ordering within known- and unknown-duration rows but force every duration_ms=0 row below known-duration rows of the same user because unknown-duration impressions are deterministically negative.
**Method:** duration-zero-lexicographic-demotion · target `features` · expected Δ 0.0006 (Data Fact #3 says duration_ms=0 rows are always negative, while the champion's 2,107-row duration-zero cohort has nDCG@5 only 0.0867 and the weaker learned categorical variant previously moved a BPR stack by +0.0002.)
**Result:** GAUC 0.6697 · nDCG@5 0.5367 · primary 0.6032 · realized Δ -0.0004 · rejected
**Diff:** `diffs/008.patch` (20 changed lines) · duration 16s · tokens in/out 55988/5032 · intervention: False

### n=9 — node_009 (deepen, parent 2)
**Hypothesis:** Halve the BPR model's learning rate after epoch 5 so it can refine the epoch-6 plateau without overshooting into the sharp decline observed after epoch 8.
**Method:** training-schedule-lr-decay-early-stop · target `training-schedule` · expected Δ 0.0005 (The schedule card spans 0.000 to 0.003, and the champion rises only from 0.6033 at epoch 5 to 0.6036 at epoch 8 before losing 0.0030 by epoch 12, supporting a lower-third estimate on this previously untested standard-FM BPR schedule.)
**Result:** GAUC 0.6708 · nDCG@5 0.5373 · primary 0.6040 · realized Δ +0.0004 · rejected · seed confirmation {'node_seed0': 0.6040371716770998, 'node_seeds': [0.6035478025962604, 0.6030966080689154, 0.6037692631098412], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00034, 'se': 0.000312, 'z': 1.1, 'sigma_pooled': 0.000382, 'sigma_df': 22, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/009.patch` (2 changed lines) · duration 17s · tokens in/out 55723/3364 · intervention: False

- _event_ (generation 2): librarian (web search) added cards: none

#### generation 2 closed — no improvement; streak 1; champion node_002; best 0.6031; tokens 459483/35450; 371s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 8 (0.6036) after 0.6034 at epoch 6, then falls to 0.6006 by epoch 12.
Watch-survival: GAUC +0.0007 but nDCG@5 −0.0006 (mixed-user nDCG −0.0010); no top-list gain and seed mean was negative.
BPR: GAUC +0.0032 and nDCG@5 +0.0012 (mixed-user nDCG +0.0020); most of its gain is pair ordering rather than top-five ordering.
L2: GAUC +0.0014, nDCG +0.0007; history aggregates: +0.0006/+0.0002, effectively dead here; seed ensemble: +0.0015/+0.0007 but unconfirmed.
Most informative next probe: merge BPR with L2=1e-5, targeting regularization of metric-aligned training; Data Fact 2’s sparse median user history makes rare-ID shrinkage plausible.
Validation-overfitting risk is moderate despite streak 0: two accepted wins have fresh-seed gains below 0.002, and repeated early-stop selection occurs around a sharp peak.
Weakest actionable duration cohorts are >180s (primary 0.5052) and <18s top-ordering (nDCG 0.4054); BPR improved them +0.0070/+0.0041, while L2 improved >180s +0.0061.
Tab 6 is weakest (0.3738, though near-zero nDCG reflects many all-negative users) and BPR hurt it −0.0043; L2/history improved it +0.0043/+0.0039, so deepen with same-author run features supported by Data Fact 10.2.
_Plan for next generation:_ `{"note": "Generation 2 produced no accepted improvement: L2 and delayed LR decay were positive but only +0.0003 fresh-seed mean, while context-matched BPR and forced duration-zero demotion regressed.\nDo not repeat the failed merge or schedule; use one exploration slot on an untried, label-free history signal targeted at the champion\u2019s weak tab-6 cohort and leave three slots to the Selector.", "plan": [{"type": "explore", "parent": "champion", "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and prev-same-author\u00d7tab categorical fields to the BPR champion, using only strictly earlier exposure features, to capture the strong within-tab negative effect of consecutive same-author impressions and target the weak tab-6 ordering without relying on unavailable evaluation-window outcomes."}]}`

### n=10 — node_010 (explore, parent 2)
**Hypothesis:** Learn a separate scalar for each FM field-pair interaction so the BPR ranker can independently control user-video, user-tab, tab-duration, and other relation strengths without the parameter growth of a field-aware FM.
**Method:** field-weighted FM relation scalars · target `model` · expected Δ 0.0006 (Pan et al.'s Field-weighted Factorization Machine directly addresses heterogeneous field relations, while Data Facts 3 and 4 show unusually strong duration and tab structure, but the tiny measured field-aware architecture effect warrants a conservative estimate.)
**Result:** GAUC 0.6698 · nDCG@5 0.5371 · primary 0.6035 · realized Δ -0.0002 · rejected
**Diff:** `diffs/010.patch` (23 changed lines) · duration 24s · tokens in/out 56258/4747 · intervention: False

### n=11 — node_011 (deepen, parent 2)
**Hypothesis:** Add a 25% auxiliary stream of ordinary same-user BPR pairs whose positive row is either under 18 seconds or over 180 seconds, concentrating updates on the two weak duration cohorts without the harmful same-tab constraint.
**Method:** duration-cohort-weighted-bpr · target `loss` · expected Δ 0.0006 (Node_002's BPR improved the >180-second and short-video cohorts by about +0.0070 and +0.0041, while they remain the champion's weakest actionable duration groups; this incremental weighting is calibrated below the BPR card's full-family range.)
**Result:** GAUC 0.6677 · nDCG@5 0.5356 · primary 0.6017 · realized Δ -0.0020 · rejected
**Diff:** `diffs/011.patch` (16 changed lines) · duration 13s · tokens in/out 58035/4672 · intervention: False

### n=12 — node_012 (deepen, parent 2)
**Hypothesis:** Blend multiseed within-user ranks from the BPR champion and the independently accepted pointwise L2 model, weighting BPR 75%, to reduce seed variance while recovering complementary regularized pointwise ordering.
**Method:** ensembling-multiseed-heterogeneous-rank-blend · target `ensembling` · expected Δ 0.0008 (The full heterogeneous multiseed card measured +0.00166, node_003 independently confirmed +0.00096 over the baseline, and node_005's pointwise seed ensemble produced a +0.00084 fresh-seed mean, supporting a conservative sub-card estimate for this changed objective-diverse stack.)
**Result:** GAUC 0.6708 · nDCG@5 0.5372 · primary 0.6040 · realized Δ +0.0003 · rejected · seed confirmation {'node_seed0': 0.6039854360682793, 'node_seeds': [0.6035553784688616, 0.6038930832904412, 0.6035371886526912], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00053, 'se': 0.000304, 'z': 1.76, 'sigma_pooled': 0.000372, 'sigma_df': 24, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/012.patch` (100 changed lines) · duration 65s · tokens in/out 60847/9782 · intervention: False

- _event_ (generation 3): librarian (web search) added cards: ['regularization-adversarial-personalized-ranking', 'model-first-order-exposure-transition-fm']

#### generation 3 closed — no improvement; streak 2; champion node_002; best 0.6031; tokens 391293/34461; 481s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 8 (0.6036), then falls to 0.6006 by epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6703/0.5370/0.7234.
Node 006 two-stream BPR hurt both halves: GAUC −0.0004, nDCG@5 −0.0003, mixed-user nDCG about −0.0004; context matching did not improve top ordering.
Node 007 stronger L2 was flat: GAUC ~0.0000, nDCG@5 ~0.0000, mixed-user nDCG about +0.0001; fresh-seed gain was only +0.0003.
Node 008 duration-zero demotion hurt GAUC −0.0006 and nDCG@5 −0.0004; within-duration group deltas were exactly zero, so this deterministic feature is dead here.
Node 009 delayed LR decay improved GAUC +0.0004 and nDCG@5 +0.0003 (mixed-user nDCG about +0.0006), but fresh-seed gain was only +0.0003.
Next probe: history-same-author-run-features, targeting exposure-history features; Data Fact 10.2 shows a strong label-free within-tab negative effect from consecutive same-author impressions.
Validation-overfitting risk is moderate: streak 1/3, with two accepted wins below +0.002 fresh-seed gain and repeated checkpoint selection around a sharp peak.
Weakness is dur>180s (primary 0.5052), short-video top ordering (nDCG 0.4054), and tab-4 GAUC (0.5831); node 009 helped tab 6 +0.0026/short +0.0006 but hurt tab 4 −0.0034, while node 006 severely hurt tab 4 −0.0089 and tab 6 −0.0202, so deepen on same-author runs within those tabs.
_Plan for next generation:_ `{"note": "Generation 3 produced no accepted improvement; the heterogeneous ensemble was the only positive candidate but gained just +0.00053 fresh-seed mean at z=1.76 and sharply harmed short/unknown-duration cohorts.\nWith the non-improving streak at 2, reserve one exploration for the still-untried label-free sequence signal directly supported by within-tab measurements, leaving the other two slots to the Selector.", "plan": [{"type": "explore", "parent": "champion", "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and prev-same-author\u00d7tab categorical fields to the BPR champion, computed from strictly earlier exposure features with equal-time rows committed together, to capture the measured negative effect of consecutive same-author impressions and target weak tab-6 ordering without outcome leakage."}]}`

### n=13 — node_013 (explore, parent 2)
**Hypothesis:** Add a TimeSVD++-style continuous date head with global, tab-specific, and video-specific slopes to the BPR score so the model can extrapolate changing item and context preferences across the validation week instead of treating every date identically.
**Method:** timesvd-continuous-item-drift-head · target `model` · expected Δ 0.0007 (Koren's TimeSVD++ models temporal preference drift, and Data Fact #5 shows a tenfold volume collapse plus material positive-rate drift while the official FM loses 0.007 primary from valid to test.)
**Result:** GAUC 0.6704 · nDCG@5 0.5371 · primary 0.6038 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6037779242176148, 'node_seeds': [0.6028103545299907, 0.602589355331101, 0.6039633793033266], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': -1e-05, 'se': 0.000332, 'z': -0.02, 'sigma_pooled': 0.000407, 'sigma_df': 26, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/013.patch` (61 changed lines) · duration 11s · tokens in/out 62162/6781 · intervention: False

### n=14 — node_014 (deepen, parent 2)
**Hypothesis:** Expand node_012's objective-diverse ensemble from three to five independently stopped BPR members and three to five pointwise-L2 members, preserving the 75/25 rank blend so additional averaging can stabilize its positive broad-ranking gain.
**Method:** ensembling-multiseed-heterogeneous-rank-blend · target `ensembling` · expected Δ 0.0007 (Node_012 achieved a +0.00053 fresh-seed mean at z=1.76, while the card's proven five-member-per-branch recipe gained +0.00166 on another heterogeneous stack, supporting a conservative variance-reduction gain.)
**Result:** ERROR at stage `implement`: no runnable script produced — recovery: None
**Diff:** `None` (None changed lines) · duration 0s · tokens in/out 198052/19682 · intervention: False

### n=15 — node_015 (deepen, parent 2)
**Hypothesis:** Apply the successful epoch-6 learning-rate halving and halve it again before epoch 9 so BPR can refine the epoch-9 plateau without entering the subsequent overfit decline.
**Method:** training-schedule-lr-decay-early-stop · target `training-schedule` · expected Δ 0.0005 (Node_009 improved both metric halves and gained +0.00034 fresh-seed mean; its primary peaked at epochs 9–10 before declining, giving a specific location for a second conservative decay.)
**Result:** GAUC 0.6709 · nDCG@5 0.5373 · primary 0.6041 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.6041063837624581, 'node_seeds': [0.6036108420918753, 0.6033562758628401, 0.6037692631098412], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00045, 'se': 0.000324, 'z': 1.39, 'sigma_pooled': 0.000397, 'sigma_df': 28, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/015.patch` (2 changed lines) · duration 14s · tokens in/out 58769/3355 · intervention: False

#### generation 4 closed — no improvement; streak 3; champion node_002; best 0.6031; tokens 501147/43414; 392s
_Diagnosis:_ Dynamics: clear overfit—champion peaks at epoch 8 (0.6036), then declines steadily to 0.6006 by epoch 12.
Node 010 moved GAUC −0.0005 but nDCG@5 +0.0001 (mixed-user nDCG +0.0003): relation scalars did not improve overall ordering.
Node 011 hurt both halves—GAUC −0.0026 and nDCG@5 −0.0014 (mixed-user nDCG −0.0023); duration-weighted BPR is dead here.
Node 012 mainly improved GAUC +0.0005, with nDCG@5 only +0.0002 (mixed-user +0.0003); the ensemble reduced broad ranking error but barely improved top-five order.
Most informative next probe: history-same-author-run-features, targeting exposure history; Data Fact 10.2 shows a strong label-free negative effect that survives conditioning on tab.
Validation-overfitting risk is high: streak is 2/3, checkpoint selection repeatedly targets a sharp peak, and both accepted wins had fresh-seed gains below 0.002.
Weakness remains dur>180s (primary 0.5052), short-video top ordering (nDCG 0.4054), and tab-4 GAUC (0.5831); tab-6’s low nDCG is partly cohort composition.
Node 010 helped short videos +0.0010 but hurt tabs 4/6; node 011 hurt dur>180s −0.0033 and tab 4 −0.0059 despite tab 6 +0.0070; node 012 helped tab 6 +0.0052/tab 4 +0.0008 but hurt short videos −0.0030, so deepen on same-author runs within tabs 4/6.
_Plan for next generation:_ `{"note": "Generation 4 produced no accepted gain: date drift was seed-flat, while two-stage LR decay improved only +0.00045 fresh-seed mean and harmed several weak cohorts; no merge qualifies.\nThe run has converged at streak 3, but if another generation is permitted, recover the failed ensemble implementation and test the still-untried, directly measured same-author sequence signal.", "plan": [{"type": "retest", "parent": "champion", "card": "ensembling-multiseed-heterogeneous-rank-blend", "hypothesis": "Successfully implement node_014's five-BPR plus five-pointwise-L2 ensemble with the existing 75/25 tie-free within-user rank blend, independently early-stopping every member and applying the identical blend to score-extra rows.", "reason": "Node_014 failed before producing a runnable script, so it supplied no evidence; its three-plus-three predecessor had a positive +0.00053 fresh-seed mean on this changed objective-diverse stack, making implementation recovery more justified than another fusion variant."}, {"type": "explore", "parent": "champion", "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and prev-same-author-by-tab categorical fields to the BPR champion, committing equal-time rows together, to capture the measured within-tab fatigue from consecutive same-author exposures and target weak tab-6 ordering without using outcomes."}]}`
