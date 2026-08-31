# Run journal — live_09

## Summary
```json
{
 "run_id": "live_09",
 "stop_reason": "converged: 3 generations without a >= 0.001 cumulative rise of the champion fresh-seed mean (ADR-0012)",
 "generations": 7,
 "nodes": 25,
 "champion": 14,
 "champion_metrics": {
  "gauc": 0.6710025969968142,
  "ndcg5": 0.5376505409925081,
  "primary": 0.6043265689946611,
  "ndcg5_disc": 0.7245731422220862,
  "by_group": {
   "dur18-60s": {
    "rows": 33235,
    "gauc": 0.6461,
    "ndcg5": 0.4602,
    "primary": 0.5531
   },
   "dur60-180s": {
    "rows": 46565,
    "gauc": 0.6496,
    "ndcg5": 0.4878,
    "primary": 0.5687
   },
   "dur<18s": {
    "rows": 20101,
    "gauc": 0.7012,
    "ndcg5": 0.4053,
    "primary": 0.5532
   },
   "dur=0": {
    "rows": 2107,
    "gauc": 0.6153,
    "ndcg5": 0.0865,
    "primary": 0.3509
   },
   "dur>180s": {
    "rows": 22901,
    "gauc": 0.6279,
    "ndcg5": 0.378,
    "primary": 0.503
   },
   "tab=0": {
    "rows": 13726,
    "gauc": 0.5778,
    "ndcg5": 0.0635,
    "primary": 0.3207
   },
   "tab=1": {
    "rows": 92672,
    "gauc": 0.6211,
    "ndcg5": 0.5415,
    "primary": 0.5813
   },
   "tab=2": {
    "rows": 3834,
    "gauc": 0.6428,
    "ndcg5": 0.5158,
    "primary": 0.5793
   },
   "tab=4": {
    "rows": 7877,
    "gauc": 0.5781,
    "ndcg5": 0.5474,
    "primary": 0.5628
   },
   "tab=6": {
    "rows": 5170,
    "gauc": 0.6361,
    "ndcg5": 0.1059,
    "primary": 0.371
   }
  },
  "by_pair": {
   "same_tab": {
    "share": 0.735,
    "err": 0.381,
    "contrib": 0.2796
   },
   "diff_tab": {
    "share": 0.265,
    "err": 0.186,
    "contrib": 0.0494
   },
   "tab1_x_tab1": {
    "share": 0.686,
    "err": 0.38,
    "contrib": 0.2603
   },
   "same_date": {
    "share": 0.236,
    "err": 0.335,
    "contrib": 0.0793
   },
   "diff_date": {
    "share": 0.764,
    "err": 0.327,
    "contrib": 0.2497
   },
   "gap>1d": {
    "share": 0.618,
    "err": 0.326,
    "contrib": 0.2017
   },
   "gap<10min": {
    "share": 0.02,
    "err": 0.34,
    "contrib": 0.0068
   },
   "pos_shorter": {
    "share": 0.513,
    "err": 0.298,
    "contrib": 0.1529
   },
   "pos_longer": {
    "share": 0.485,
    "err": 0.363,
    "contrib": 0.1759
   },
   "total_err": 0.329
  }
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00286,
 "top3_valid": [
  {
   "n": 16,
   "primary": 0.6046305106501695
  },
  {
   "n": 17,
   "primary": 0.6046125157576204
  },
  {
   "n": 18,
   "primary": 0.604511966159397
  }
 ],
 "designated": 14,
 "final_ranking": [
  {
   "n": 14,
   "valid_primary": 0.6043265689946611,
   "fresh_seeds": [
    0.6046725943624534,
    0.6043597781166279,
    0.604702099707588
   ],
   "accepted": true,
   "mean": 0.6045781573955564,
   "std": 0.00018969652987572085,
   "n_seeds": 3
  },
  {
   "n": 1,
   "valid_primary": 0.6036463694691105,
   "fresh_seeds": [
    0.6031184279276939,
    0.6027235946155616,
    0.6035425471235136
   ],
   "accepted": true,
   "mean": 0.603128189888923,
   "std": 0.00040956351703431667,
   "n_seeds": 3
  },
  {
   "n": 3,
   "valid_primary": 0.6021140948752941,
   "fresh_seeds": [
    0.6024596194372596,
    0.6023584723292854,
    0.6025516371872324
   ],
   "accepted": true,
   "mean": 0.6024565763179258,
   "std": 9.661837826147237e-05,
   "n_seeds": 3
  },
  {
   "n": 2,
   "valid_primary": 0.6025029206269998,
   "fresh_seeds": [
    0.6028729444838677,
    0.6017090117891591,
    0.6020694898356386,
    0.6027857969882158,
    0.6026234157272281
   ],
   "accepted": true,
   "mean": 0.6024121317648219,
   "std": 0.0005020946538723934,
   "n_seeds": 5
  },
  {
   "n": 17,
   "valid_primary": 0.6046125157576204,
   "fresh_seeds": [
    0.604866487530594,
    0.6044991650533298,
    0.6047135229100419
   ],
   "accepted": false,
   "mean": 0.6046930584979886,
   "std": 0.00018451434821281203,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  },
  {
   "n": 16,
   "valid_primary": 0.6046305106501695,
   "fresh_seeds": [
    0.6047709032427752,
    0.6043679844221574,
    0.6043433445432675
   ],
   "accepted": false,
   "mean": 0.6044940774027334,
   "std": 0.00024005455670337398,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  },
  {
   "n": 18,
   "valid_primary": 0.604511966159397,
   "fresh_seeds": [
    0.6043267291880454,
    0.6037250220169611,
    0.6042217611481353
   ],
   "accepted": false,
   "mean": 0.6040911707843806,
   "std": 0.0003214082485916658,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  }
 ],
 "usage": {
  "calls": 92,
  "tokens_in": 4018861,
  "tokens_out": 393023,
  "cache_read": 2716484,
  "cache_write": 0,
  "cost_usd": 19.660816999999998
 },
 "wall_clock_s": 6977.4,
 "champion_seed_mean": 0.60458,
 "best_single_seed": 0.6046305106501695,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": {
  "best_single_seed": 0.6036463694691105,
  "streak": 6,
  "converged_at_generation": 4,
  "champion_at_stop": 14
 },
 "official_rule_submission": {
  "node": 14,
  "generation": 4,
  "valid_primary": 0.6043265689946611,
  "fresh_seed_mean": 0.60458,
  "fresh_seeds": 3
 },
 "convergence_switch": "confirmed",
 "campaigns": true,
 "families": {
  "aux-targets": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "data-weighting": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "encoding": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "ensembling": {
   "status": "closed",
   "generations": [
    4,
    5,
    6
   ],
   "nodes": [
    14,
    16,
    18,
    20,
    21
   ],
   "best_gain": 0.00145,
   "flat_streak": 2,
   "evidence": "closed at generation 6: 2 campaign generations without an accepted node (nodes [14, 16, 18, 20, 21], best gain 0.00145)"
  },
  "features": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "history": {
   "status": "closed",
   "generations": [
    2,
    3
   ],
   "nodes": [
    7,
    8,
    11,
    12
   ],
   "best_gain": 7e-05,
   "flat_streak": 2,
   "evidence": "closed at generation 3: 2 campaign generations without an accepted node (nodes [7, 8, 11, 12], best gain 7e-05)"
  },
  "ranking-loss": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "watch-time": {
   "status": "exhausted",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": "nothing left to measure on this stack"
  },
  "model": {
   "status": "open",
   "generations": [
    7
   ],
   "nodes": [
    23,
    24
   ],
   "best_gain": 0.0001,
   "flat_streak": 1,
   "evidence": ""
  },
  "sequential-model": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "regularization": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "training-schedule": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  }
 },
 "designation_rule": "strict",
 "designation_events": [
  "designation (strict): node_017 leads on fresh-seed mean (0.60469) but was not accepted; excluded \u2014 accepted lineage only"
 ],
 "best_unaccepted": {
  "n": 17,
  "mean": 0.60469,
  "valid_primary": 0.6046125157576204,
  "n_seeds": 3
 },
 "tokens": {
  "in_total": 4018861,
  "in_cached": 2716484,
  "in_uncached": 1302377,
  "out": 393023
 },
 "interventions": 0,
 "k": 5,
 "k_later": 3,
 "eps": 0.002,
 "n_converge": 3,
 "iteration_unit": "node",
 "iterations_used": 25
}
```

## Iterations

### n=0 — node_000 (reproduce_baseline, parent None)
**Hypothesis:** Reproduce the official FM baseline under the harness contract.
**Method:** official FM · target `None` · expected Δ 0.0 (published valid primary 0.6016)
**Result:** GAUC 0.6671 · nDCG@5 0.5358 · primary 0.6015
**Diff:** `None` (None changed lines) · duration 15s · tokens in/out 0/0 · intervention: False

- _screen_ (generation 1) DROPPED: `user-lifecycle-cohort-interactions` — best_gain +0.0000 (lifecycle_cohort); stack +0.0000; lifecycle_cohort: varies 0.0, GAUC 0.5, additive +0.0000

- _screen_ (generation 1) kept: `history-user-aggregates` — best_gain +0.0008 (stack); stack +0.0008; author_history_rate: varies 0.134, GAUC 0.5018, additive +0.0000; author_history_count: varies 0.136, GAUC 0.5044, additive -0.0004; tab_history_rate: varies 0.352, GAUC 0.554, additive -0.0000; tab_history_count: varies 0.387, GAUC 0.5303, additive +0.0001; duration_history_rate: varies 0.708, GAUC 0.5089, additive -0.0003; duration_history_count: varies 0.716, GAUC 0.5108, additive -0.0005

### n=1 — node_001 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with same-user BPR so training directly optimizes pair ordering, especially the tab1×tab1 pairs that constitute 69% of residual GAUC mass.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0015 (The card has repeatedly confirmed +0.0010 to +0.0022 on the official-FM stack, and the diagnosis identifies BPR as the most informative ranking-aligned probe.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6036 · realized Δ +0.0022 · ACCEPTED · seed confirmation {'node_seed0': 0.6036463694691105, 'node_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00168, 'se': 0.000278, 'z': 6.04, 'sigma_pooled': 0.00034, 'sigma_df': 4, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/001.patch` (38 changed lines) · duration 19s · tokens in/out 71454/5127 · intervention: False

### n=2 — node_002 (improve, parent 0)
**Hypothesis:** Raise embedding L2 to 1e-5 so sparse ID embeddings retain useful ordering signal without producing the sharp post-epoch-7 validation decline.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0009 (The champion falls from 0.6015 at epoch 7 to 0.5990 at epoch 11, while this two-line regularization change previously confirmed gains of +0.0009 to +0.0010 on official FM.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6025029206269998, 'node_seeds': [0.6028729444838677, 0.6017090117891591, 0.6020694898356386, 0.6027857969882158, 0.6026234157272281], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00096, 'se': 0.000293, 'z': 3.28, 'sigma_pooled': 0.000401, 'sigma_df': 8, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/002.patch` (2 changed lines) · duration 30s · tokens in/out 70004/3583 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Average predictions from five independently initialized and early-stopped FM members to cancel seed-specific within-user ordering errors.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0013 (Five-seed averaging repeatedly gained about +0.0013 on official FM, within the card's confirmed upper bound of +0.0016.)
**Result:** GAUC 0.6682 · nDCG@5 0.5360 · primary 0.6021 · realized Δ +0.0006 · ACCEPTED · seed confirmation {'node_seed0': 0.6021140948752941, 'node_seeds': [0.6024596194372596, 0.6023584723292854, 0.6025516371872324], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00101, 'se': 0.000305, 'z': 3.3, 'sigma_pooled': 0.000373, 'sigma_df': 10, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/003.patch` (49 changed lines) · duration 66s · tokens in/out 70782/6610 · intervention: False

### n=4 — node_004 (improve, parent 0)
**Hypothesis:** Add leakage-safe user-by-author, user-by-tab, and user-by-duration historical-rate fields so candidates can be ordered using preferences learned from strictly earlier impressions.
**Method:** history-user-aggregates · target `history` · expected Δ 0.0003 (The card recorded a confirmed +0.0010 on official FM, but the current ceiling analysis bounds most train-history effects tightly, so this first run is conservatively placed in the lower third.)
**Result:** GAUC 0.6681 · nDCG@5 0.5364 · primary 0.6022 · realized Δ +0.0008 · rejected · seed confirmation {'node_seed0': 0.6022312148412909, 'node_seeds': [0.6020333666607715, 0.601522801864333, 0.6020294777516492], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00041, 'se': 0.000298, 'z': 1.38, 'sigma_pooled': 0.000364, 'sigma_df': 12, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/004.patch` (108 changed lines) · duration 26s · tokens in/out 73967/15409 · intervention: False

### n=5 — node_005 (explore, parent 0)
**Hypothesis:** Use row-conditioned attention over the FM's ten field-pair interactions so the model can vary interaction importance across videos, tabs, and duration contexts instead of summing every pair equally.
**Method:** model-attentional-factorization-machine · target `model` · expected Δ 0.0001 (The untried card's +0.0010 analogy ceiling is discounted to +0.0003 by calibration and then placed in its lower third because prior interaction-capacity additions were mostly flat.)
**Result:** GAUC 0.6666 · nDCG@5 0.5351 · primary 0.6009 · realized Δ -0.0006 · rejected
**Diff:** `diffs/005.patch` (65 changed lines) · duration 46s · tokens in/out 71248/8162 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_001; best 0.6031; tokens 555946/54517; 587s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 7 (0.6015), then falls steadily to 0.5990 by epoch 11; GAUC/nDCG@5 are 0.6671/0.5358, with mixed-user nDCG@5 0.7214.
Last generation had no candidate nodes, so no GAUC-versus-nDCG movement can be attributed.
Most informative next probe: within-user BPR, targeting the loss; it directly aligns with GAUC and has repeatedly confirmed gains, while fact §11 shows 69% of residual pair mass is tab1×tab1.
Validation-selection risk is currently low: streak 0 and no sub-0.002 wins accepted in this run, though future small wins require fresh-seed confirmation.
Group weakness is concentrated in dur>180s (primary 0.4982, GAUC 0.6196) and same-tab/tab1 ordering; low nDCG for tab=0, tab=6, and dur=0 mostly reflects all-negative composition, not actionable ranking weakness.
The sharper error map is tab1×tab1 pairs (69% mass, 0.383 misordered), especially broad temporal gaps; no generation node moved this mass, and gap<10min features would touch only 2%.
_Plan for next generation:_ `{"note": "BPR is the new champion with a confirmed +0.0017 mean gain, while stronger embedding L2 independently improved the same pointwise parent by +0.0010 and targets a different component.\nMerge only these qualifying branches; leave the other four slots to the Selector because the generation improved and neither rejected history aggregates nor AFM warrants a same-context retest.", "plan": [{"type": "merge", "merge_parents": [1, 2], "hypothesis": "Apply the node_002 embedding L2=1e-5 regularization to the node_001 same-user BPR champion; BPR's validation curve peaks sharply at epoch 8 and then declines, so stronger shrinkage may preserve its pairwise ordering gains while reducing sparse-ID overfitting."}]}`

- _screen_ (generation 2) DROPPED: `history-click-conversion-profile` — best_gain +0.0002 (user_tab_click_rate); stack -0.0003; user_tab_click_rate: varies 0.39, GAUC 0.5589, additive +0.0002; ut_impressions: varies 0.387, GAUC 0.5303, additive -0.0002; user_duration_click_to_long_rate: varies 0.714, GAUC 0.5076, additive -0.0004; ud_clicks: varies 0.703, GAUC 0.507, additive -0.0009

- _screen_ (generation 2) DROPPED: `history-same-author-run-features` — best_gain -0.0001 (stack); stack -0.0001; same_author_run_so_far_cap5: varies 0.03, GAUC 0.5003, additive -0.0005

### n=6 — node_006 (merge, parent 1, merge of [1, 2])
**Hypothesis:** Apply node_002's L2=1e-5 setting to the node_001 BPR champion so sparse-ID shrinkage reduces its sharp post-epoch-8 overfitting without changing the ranking-aligned objective.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0003 (Node_002 independently gained +0.0010 fresh-seed mean and the champion falls from 0.6036 at epoch 8 to 0.6006 at epoch 12, although prior L2 tests on BPR topped out near +0.0003.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6037 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6036798758461768, 'node_seeds': [0.6034092499923741, 0.6034185207299905, 0.6035365750210787], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00033, 'se': 0.000281, 'z': 1.16, 'sigma_pooled': 0.000344, 'sigma_df': 14, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/006.patch` (2 changed lines) · duration 13s · tokens in/out 73318/3149 · intervention: False

### n=7 — node_007 (deepen, parent 4)
**Hypothesis:** Train node_004's aggregate-history FM with a mostly-BPR hybrid objective so pairwise optimization supplies the champion's ordering gain while a small pointwise term preserves the history signal that helped diff-tab pairs.
**Method:** history-user-aggregates · target `loss` · expected Δ 0.0003 (Node_004 improved diff-tab errors by 0.004 and had seed-0 gain +0.0008, while node_001's BPR gained +0.0017 fresh-seed mean; the estimate is discounted because earlier pure-BPR aggregate compositions were flat.)
**Result:** GAUC 0.6689 · nDCG@5 0.5365 · primary 0.6027 · realized Δ -0.0010 · rejected
**Diff:** `diffs/007.patch` (43 changed lines) · duration 17s · tokens in/out 77378/6557 · intervention: False

### n=8 — node_008 (deepen, parent 1)
**Hypothesis:** Blend the BPR champion with a separately trained pointwise recurrence FM at low rank weight so the pointwise-confirmed latest-positive attribute signal remains complementary instead of being absorbed by BPR embeddings.
**Method:** history-last-positive-attribute-recurrence · target `ensembling` · expected Δ 0.0002 (The recurrence card confirmed +0.0009 on pointwise FM but was flat when inserted into a BPR seed ensemble, while Fact §11 caps observable train-history taste near +0.0002; a separate low-weight branch tests complementarity conservatively.)
**Result:** GAUC 0.6701 · nDCG@5 0.5370 · primary 0.6036 · realized Δ -0.0001 · rejected
**Diff:** `diffs/008.patch` (144 changed lines) · duration 35s · tokens in/out 153312/34344 · intervention: False

#### generation 2 closed — no improvement; streak 1; champion node_001; best 0.6031; tokens 553751/65626; 1291s
_Diagnosis:_ Dynamics: overfit—champion peaks at epoch 8 (0.6036), then falls steadily to 0.6006 by epoch 12.
BPR moved both halves versus node_000: GAUC +0.0032, nDCG@5 +0.0012, mixed-user nDCG +0.0020; stronger pair-ordering gain than top-list gain.
L2 moved GAUC +0.0014 and nDCG +0.0007 (mixed +0.0012); seed averaging +0.0011/+0.0002 (mixed +0.0004), mostly GAUC variance reduction.
History aggregates moved GAUC +0.0010 and nDCG +0.0006 (mixed +0.0009) but failed confirmation; AFM moved neither, falling −0.0005/−0.0007 (mixed −0.0012).
Most informative next probe: merge L2 into BPR, targeting regularization of sparse ID embeddings; Fact §2 shows median user history is only 35 rows, and BPR itself sharply overfits after epoch 8.
Validation-overfitting risk is moderate: streak 0, but three accepted gains have fresh-seed means below 0.002, so selection is already operating near the noise-sensitive regime.
Weakness remains same-tab/tab1×tab1 ordering (69% pair mass, 0.380 misordered), especially broad gaps; BPR genuinely moved it −0.003 and gap>1d −0.003, while L2 moved −0.001/−0.002 and other nodes were flat or harmful. BPR also improved dur>180s +0.007 and tabs 2/4 strongly; low tab0/dur0 nDCG is mostly composition, not the deepen target.
History campaign: aggregates helped diff-tab/diff-date pairs (−0.004/−0.002) and tab4 (+0.0083) but worsened core tab1×tab1 (+0.001); with other history mechanisms bounded or dead, it should not remain a priority.
_Plan for next generation:_ `{"note": "Generation 2 was flat: stronger L2 and both history compositions failed, so there is no qualifying merge or deepen.\nUse two changed-stack retests with prior confirmation on BPR, leaving the remaining history-campaign slots to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "ensembling-seed-average", "hypothesis": "Average five independently initialized, independently early-stopped BPR-FM members to reduce seed-specific within-user ordering errors while preserving the champion's ranking-aligned objective.", "reason": "The run tested seed averaging only on the pointwise parent, whereas the champion stack is now BPR; this exact BPR composition previously produced confirmed gains up to +0.0016."}, {"type": "retest", "parent": "champion", "card": "features-exposure-session", "hypothesis": "Add leakage-safe session-position, recent-density, and previous-gap buckets to the BPR champion so attention-budget state can reorder impressions within users, especially same-day and short-gap pairs.", "reason": "The stack has changed from the pointwise FM where session features failed to the BPR stack where this card previously confirmed +0.0009; the champion currently has no exposure-sequence inputs."}]}`

- _screen_ (generation 3) DROPPED: `history-bayesian-user-music-affinity` — best_gain -0.0000 (user_music_support); stack -0.0002; user_music_rate: varies 0.139, GAUC 0.5025, additive -0.0000; user_music_affinity: varies 0.139, GAUC 0.5025, additive -0.0008; user_music_support: varies 0.139, GAUC 0.5058, additive -0.0000; user_music_pos: varies 0.054, GAUC 0.5024, additive -0.0001; user_music_neg: varies 0.123, GAUC 0.5052, additive -0.0001

- _screen_ (generation 3) DROPPED: `history-last-positive-attribute-recurrence` — best_gain +0.0000 (latest_positive_music_match); stack -0.0004; has_latest_positive: varies 0.0, GAUC 0.5, additive +0.0000; latest_positive_gap_seconds: varies 0.783, GAUC 0.5158, additive -0.0000; latest_positive_tag_match: varies 0.345, GAUC 0.511, additive -0.0016; latest_positive_tag_known_pair: varies 0.038, GAUC 0.5, additive -0.0003; latest_positive_music_match: varies 0.005, GAUC 0.5004, additive +0.0000; latest_positive_music_known_pair: varies 0.0, GAUC 0.5, additive +0.0000; latest_positive_type_match: varies 0.044, GAUC 0.5015, additive -0.0000; latest_positive_type_known_pair: varies 0.0, GAUC 0.5, additive +0.0000

- _screen_ (generation 3) kept: `history-user-aggregates` — best_gain +0.0004 (stack); stack +0.0004; author_rate: varies 0.136, GAUC 0.5018, additive -0.0001; author_support: varies 0.136, GAUC 0.5044, additive -0.0006; tab_rate: varies 0.39, GAUC 0.559, additive +0.0003; tab_support: varies 0.387, GAUC 0.5303, additive -0.0002; duration_rate: varies 0.732, GAUC 0.5099, additive +0.0001; duration_support: varies 0.722, GAUC 0.5064, additive +0.0003

### n=9 — node_009 (retest, parent 1)
**Hypothesis:** Average five independently initialized and independently early-stopped BPR-FM members to cancel seed-specific ordering errors without weakening the champion's pairwise objective.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0013 (This exact BPR composition previously confirmed gains of +0.0013 to +0.0016, and Fact §11 places 20-seed and multi-lineage ensembles near 0.6044–0.6047.)
**Result:** GAUC 0.6709 · nDCG@5 0.5373 · primary 0.6041 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.6041048866291208, 'node_seeds': [0.6036982285733025, 0.603722396181196, 0.6035349494588524], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00052, 'se': 0.000268, 'z': 1.95, 'sigma_pooled': 0.000328, 'sigma_df': 16, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/009.patch` (54 changed lines) · duration 44s · tokens in/out 75079/5290 · intervention: False

### n=10 — node_010 (retest, parent 1)
**Hypothesis:** Add strictly causal session-position, recent-density, and previous-gap buckets to BPR so attention-budget state can reorder impressions within users.
**Method:** features-exposure-session · target `features` · expected Δ 0.0009 (The card previously confirmed a +0.0009 fresh-seed gain on the FM+BPR stack, and Fact §10.5 reports large within-tab attention decay even though short-gap pairs are only 2% of GAUC mass.)
**Result:** GAUC 0.6701 · nDCG@5 0.5376 · primary 0.6038 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.6038316485661749, 'node_seeds': [0.6040520186884266, 0.6032083584140264, 0.6042151796177052, 0.6044974652306726, 0.6029803245069025], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00066, 'se': 0.000294, 'z': 2.25, 'sigma_pooled': 0.000403, 'sigma_df': 20, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/010.patch` (72 changed lines) · duration 22s · tokens in/out 72916/10343 · intervention: False

### n=11 — node_011 (deepen, parent 4)
**Hypothesis:** Cross node_004's historical-rate bins with coarse support-count bins so the FM can distinguish reliable preferences from noisy low-count rates while retaining the tab=4 improvement.
**Method:** history-user-aggregates · target `encoding` · expected Δ 0.0002 (Node_004 had a +0.0004 fresh-seed mean and improved tab=4 by about +0.0083, but Fact §11 caps legal train-history additions near +0.0002.)
**Result:** GAUC 0.6677 · nDCG@5 0.5358 · primary 0.6018 · realized Δ -0.0019 · rejected
**Diff:** `diffs/011.patch` (7 changed lines) · duration 20s · tokens in/out 77100/5692 · intervention: False

### n=12 — node_012 (deepen, parent 6)
**Hypothesis:** Replace node_006's uniform stronger L2 with user-history-support-adaptive embedding shrinkage so sparse users are regularized without suppressing well-supported personalized interactions.
**Method:** history-user-aggregates · target `regularization` · expected Δ 0.0001 (Node_006's uniform L2 gained only +0.0003 fresh-seed mean, while Fact §2's median of 35 interactions and the champion's post-epoch-8 decline support a narrowly targeted rather than globally stronger penalty.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6037 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6036580632977433, 'node_seeds': [0.6031459529027117, 0.6028849129383169, 0.6035779345038896], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 7e-05, 'se': 0.000326, 'z': 0.23, 'sigma_pooled': 0.000399, 'sigma_df': 22, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/012.patch` (9 changed lines) · duration 15s · tokens in/out 72731/5323 · intervention: False

#### generation 3 closed — no improvement; streak 2; champion node_001; best 0.6031; tokens 596851/52172; 521s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 8 at 0.6036, then falls to 0.6006 by epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6703/0.5370/0.7234.
Node 006 was flat on both halves (GAUC +0.0000, nDCG +0.0000; disc +0.0001), so stronger L2 adds no useful BPR ordering.
Node 007 regressed both GAUC −0.0014 and nDCG −0.0006 (disc −0.0010); node 008 was effectively flat but slightly negative (GAUC −0.0002, nDCG −0.0000, disc −0.0001).
Next probe: five-member independently stopped BPR seed averaging, targeting ensembling/variance; Fact §11 reports 20-seed and multi-lineage ensembles near 0.6044–0.6047.
Validation-overfitting risk is moderate-high: streak 1, and all three accepted wins have fresh-seed gains below 0.002, so further marginal selection is noise-sensitive.
The actionable weakness is same-tab tab1×tab1 ordering, 69% of pair mass at 0.380 error, mostly across days; node 006 left it unchanged, while nodes 007 and 008 each worsened it by +0.001.
Group gains were narrow: node 006 helped tab2 +0.0013, node 007 dur>180s +0.0015, and node 008 tab6 +0.0018/tab0 +0.0009; none offsets broad regressions, while low tab0/dur0 nDCG is composition-driven and no group is HARD.
History campaign should close: the aggregate hybrid moved neither core ordering nor top-list quality, and recurrence only nudged small tab groups while worsening same-tab/tab1 and gap<10min pairs.
_Plan for next generation:_ `{"note": "No generation-3 node was accepted or improved the champion by >= 0.001, so no merge qualifies; aggregate-history mechanisms are now closed.\nThe session near-miss has a clear cohort-specific refinement, while APR is the strongest untried regularizer for the champion\u2019s sharp BPR overfitting; leave three slots to the Selector.", "plan": [{"type": "deepen", "parent": 10, "card": "features-exposure-session", "mechanism": "tab1-gated-session-attention-state", "target_group": "tab=1 / same_date", "hypothesis": "Gate the session-position, recent-density, and previous-gap fields to tab=1, mapping other tabs to a neutral level, so the variant retains node_010's gains on same-date (+0.002), gap<10min (+0.013), and tab1\u00d7tab1 pairs while avoiding its large regressions on tab=6 and smaller losses on tabs 0, 2, and 4."}, {"type": "explore", "parent": 1, "card": "regularization-adversarial-personalized-ranking", "hypothesis": "Add a conservative adversarial BPR term with epsilon=0.05 and weight 0.5 to stabilize row-varying embedding interactions against worst-case perturbations; this directly targets the champion's decline from 0.6036 at epoch 8 to 0.6006 at epoch 12 after ordinary stronger L2 proved flat on BPR."}]}`

### n=13 — node_013 (explore, parent 1)
**Hypothesis:** Blend the champion with a BPR expert that infers whether the immediately previous impression was long-viewed from its inter-impression gap and duration threshold, allowing label-conditioned same-author continuation to influence same-window ranking.
**Method:** ensembling-pseudo-dwell-transition-expert · target `ensembling` · expected Δ 0.0004 (Fact §3 makes long_view depend on min(duration_ms, 18000), while Fact §10.1 reports same-author continuation rates of 0.78 after a positive versus 0.077 after a negative; timestamp-derived pseudo-feedback may expose part of this otherwise unavailable signal.)
**Result:** GAUC 0.6702 · nDCG@5 0.5370 · primary 0.6036 · realized Δ -0.0001 · rejected
**Diff:** `diffs/013.patch` (124 changed lines) · duration 28s · tokens in/out 155482/31991 · intervention: False

### n=14 — node_014 (improve, parent 1)
**Hypothesis:** Blend five independently stopped standard-FM BPR members with five field-aware-FM BPR members using tie-free within-user ranks so architecture and seed diversity cancel complementary ordering errors.
**Method:** ensembling-multiseed-heterogeneous-rank-blend · target `ensembling` · expected Δ 0.0005 (The card's best confirmed gain is +0.0017, but its prior FM+BPR result was +0.0005 and Fact §11 reports two lineages at 0.6047 versus roughly 0.6044 for seed averaging alone.)
**Result:** GAUC 0.6710 · nDCG@5 0.5377 · primary 0.6043 · realized Δ +0.0007 · ACCEPTED · seed confirmation {'node_seed0': 0.6043265689946611, 'node_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'champion_seeds': [0.6031184279276939, 0.6027235946155616, 0.6035425471235136], 'delta_mean': 0.00145, 'se': 0.000317, 'z': 4.58, 'sigma_pooled': 0.000388, 'sigma_df': 24, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/014.patch` (109 changed lines) · duration 121s · tokens in/out 77162/11713 · intervention: False

#### generation 4 closed — improved; streak 0; champion node_014; best 0.6046; tokens 509766/59165; 916s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 8 (0.6036), then falls to 0.6006 by epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6703/0.5370/0.7234.
Node 009 moved both halves slightly: GAUC +0.0005, nDCG@5 +0.0004, disc +0.0006; node 010 traded GAUC −0.0003 for nDCG@5 +0.0006, disc +0.0011, indicating top-list-only benefit.
Node 011 regressed both halves (GAUC −0.0026, nDCG@5 −0.0012, disc −0.0021); node 012 was flat on GAUC, nDCG@5, and disc, so adaptive shrinkage is dead here.
Most informative next probe: heterogeneous multiseed rank blending, targeting ensembling; Fact §11 reports two lineages reaching 0.6047 versus roughly 0.6044 for seed averaging alone.
Validation-overfitting risk is high: streak 2 of 3, with all three accepted wins having fresh-seed gains below 0.002.
The sharp error remains same-tab/tab1×tab1 across days (69% pair mass, 0.380 error): node 009 left it flat, node 010 worsened it +0.001 while worsening gap<10min +0.013, node 011 worsened it +0.004, and node 012 was flat.
Group gains were narrow: node 009 helped tab0/tab2 and dur>180s but hurt tab6/tab4; node 010 helped 18–60s and >180s but badly hurt tab6; low tab0/dur0 nDCG is composition-driven, and no group is HARD.
Ensembling should stay open for one heterogeneous-lineage probe: homogeneous seed averaging modestly improved diff-tab pairs (−0.001) but failed confirmation and did not touch the dominant tab1×tab1 error.
_Plan for next generation:_ `{"note": "No merge qualifies: node_014 was accepted but its seed-0 improvement was only +0.0007, while node_013 was flat and targeted the same ensembling component.\nRetest only the session signal because it had a +0.0007 fresh-seed near-miss on single-model BPR and the champion has changed to a heterogeneous field-aware/standard ensemble; leave two slots to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "features-exposure-session", "hypothesis": "Add strictly causal session-position, recent-density, and previous-gap fields only to the five field-aware BPR members, retaining the five standard members as an unmodified anchor before the existing 0.6/0.4 within-user rank blend.", "reason": "The earlier BPR session node improved nDCG@5 and achieved fresh-seed mean +0.0007 but failed significance; the new heterogeneous ensemble changes the stack and currently regresses gap<10min and same-date pair ordering, giving the session-aware field-aware branch a specific complementary role."}]}`

### n=15 — node_015 (explore, parent 14)
**Hypothesis:** Diversify only the field-aware ensemble branch with a pooled creator-role × candidate-format × tab field so sparse or unseen users can inherit cohort-level format preferences while the unchanged standard branch anchors memorized ID rankings.
**Method:** ensembling-creator-role-format-branch · target `ensembling` · expected Δ 0.0002 (Fact §1 reports 3.6% unseen-user test rows, while the within-user invariance theorem says user metadata can help only through interactions with row-varying context; the closed catalogue and prior flat content ablations justify a lower-third ensembling estimate.)
**Result:** GAUC 0.6711 · nDCG@5 0.5375 · primary 0.6043 · realized Δ -0.0000 · rejected
**Diff:** `diffs/015.patch` (51 changed lines) · duration 211s · tokens in/out 77547/7987 · intervention: False

### n=16 — node_016 (improve, parent 14)
**Hypothesis:** Increase each architecture branch from five to ten independently seeded and early-stopped members before the existing 0.6/0.4 rank blend to reduce residual member-level ordering variance.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0002 (The card has confirmed gains up to +0.0016, but node_014 already averages five members per lineage and Fact §11 places 20-seed ensembles near the current ceiling, so only a small incremental gain is expected.)
**Result:** GAUC 0.6718 · nDCG@5 0.5375 · primary 0.6046 · realized Δ +0.0003 · rejected · seed confirmation {'node_seed0': 0.6046305106501695, 'node_seeds': [0.6047709032427752, 0.6043679844221574, 0.6043433445432675], 'champion_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'delta_mean': -8e-05, 'se': 0.00031, 'z': -0.27, 'sigma_pooled': 0.00038, 'sigma_df': 26, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/016.patch` (6 changed lines) · duration 316s · tokens in/out 76482/4404 · intervention: False

### n=17 — node_017 (retest, parent 14)
**Hypothesis:** Add strictly causal session-position, recent-density, and previous-gap fields only to the five field-aware BPR members while retaining the standard members as an unmodified anchor in the existing blend.
**Method:** features-exposure-session · target `features` · expected Δ 0.0005 (Node_010 achieved a +0.0007 fresh-seed mean near-miss and improved top-list quality, while the changed heterogeneous stack gives the session-aware branch a complementary role supported by Fact §10.5.)
**Result:** GAUC 0.6712 · nDCG@5 0.5380 · primary 0.6046 · realized Δ +0.0003 · rejected · seed confirmation {'node_seed0': 0.6046125157576204, 'node_seeds': [0.604866487530594, 0.6044991650533298, 0.6047135229100419], 'champion_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'delta_mean': 0.00011, 'se': 0.000303, 'z': 0.38, 'sigma_pooled': 0.000371, 'sigma_df': 28, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/017.patch` (102 changed lines) · duration 296s · tokens in/out 78464/7604 · intervention: False

### n=18 — node_018 (deepen, parent 14)
**Hypothesis:** Use a mixture of k=8, k=16, and k=32 members inside each existing architecture branch so equally competitive capacities contribute less-correlated within-user ordering errors.
**Method:** ensembling-multiseed-heterogeneous-rank-blend — latent-dimension-diverse-members · target `model` · expected Δ 0.0002 (Node_014 gained +0.00145 broadly from architecture diversity, but the organizer's k=8/16/32 ablation was flat, limiting this variant to a small decorrelation benefit in the lower third of the card's range.)
**Result:** GAUC 0.6713 · nDCG@5 0.5377 · primary 0.6045 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.604511966159397, 'node_seeds': [0.6043267291880454, 0.6037250220169611, 0.6042217611481353], 'champion_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'delta_mean': -0.00049, 'se': 0.000301, 'z': -1.62, 'sigma_pooled': 0.000368, 'sigma_df': 30, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/018.patch` (4 changed lines) · duration 194s · tokens in/out 72723/4896 · intervention: False

#### generation 5 closed — no improvement; streak 1; champion node_014; best 0.6046; tokens 599762/51242; 1546s
_Diagnosis:_ Dynamics: mild overfit—primary peaks at epoch 6 (0.6045) then falls to 0.6043 at epoch 7; champion GAUC/nDCG@5/ndcg5_disc = 0.6710/0.5377/0.7246.
Node 013 moved neither half: GAUC −0.0001 and nDCG@5 ≈0.0000 versus its parent, so pseudo-dwell transition ensembling is dead here.
Node 014 improved both halves: GAUC +0.0007 and nDCG@5 +0.0006, with mixed-user nDCG +0.0012; heterogeneous multiseed blending adds broad rather than purely top-list value.
Next probe: add causal session-position/density/gap fields only to the field-aware branch, targeting feature-conditioned ensembling; Fact §10.5 shows strong attention decay, though Fact §11 caps short-gap pairs at 2% of mass.
Validation-selection risk remains high despite streak 0: all four accepted wins have fresh-seed mean gains below 0.002.
Weakness is dur>180s and tabs 4/6 beyond composition-driven tab0/dur0 nDCG; node 014 hurt them −0.0022/−0.0026/−0.0028 while helping 18–180s and tab0, whereas node 013 moved no useful group; no groups are HARD.
The sharper error remains tab1×tab1 across days (69% pair mass, 0.380 error): neither node changed it; node 014 only improved same-date, gap>1d, gap<10min, and pos-longer pairs, while node 013 worsened short-gap pairs.
Ensembling should stay open for the single changed-stack session-aware branch test: pseudo-dwell failed, but heterogeneous rank blending produced the accepted +0.00145 seed-mean gain.
_Plan for next generation:_ `{"note": "No merge qualifies: every generation-5 node improved the champion by less than 0.001, and additional seeds, session features, and latent-dimension diversity were flat or negative after confirmation.\nUse one untried regularization probe outside the now-flat ensembling campaign; leave the other three slots to the Selector.", "plan": [{"type": "explore", "parent": 14, "card": "regularization-adversarial-personalized-ranking", "hypothesis": "Apply conservative APR regularization (epsilon=0.05, adversarial weight=0.5) to the five field-aware BPR members while retaining the five standard members as an unchanged anchor, testing whether robust pairwise embeddings reduce post-peak ordering instability without sacrificing the champion's complementary architecture blend."}]}`

- _event_ (generation 6): free-slot rule violated twice by the Selector; proceeded

### n=19 — node_019 (explore, parent 14)
**Hypothesis:** Blend the champion with a causal tab-conditioned daily-attention expert that lowers scores as a user accumulates impressions across all sessions that day, capturing cross-session fatigue missed by the 30-minute-reset session features.
**Method:** ensembling-cross-session-daily-budget-prior · target `ensembling` · expected Δ 0.0003 (Fact §10.5 shows steep attention decay with exposure position, while 24% of champion GAUC pair mass is same-date but only 2% is within ten minutes, supporting a modest gain from cumulative day position beyond previously tested session-local signals.)
**Result:** GAUC 0.6711 · nDCG@5 0.5375 · primary 0.6043 · realized Δ -0.0000 · rejected
**Diff:** `diffs/019.patch` (84 changed lines) · duration 162s · tokens in/out 77168/12029 · intervention: False

### n=20 — node_020 (deepen, parent 17)
**Hypothesis:** Turn node_017's session-aware field branch into a dur>180s specialist by activating its causal session fields only for long-duration rows, retaining the unchanged standard branch to avoid the session variant's regressions elsewhere.
**Method:** ensembling-long-duration-slot-specialists — causal-session long-duration branch · target `ensembling` · expected Δ 0.0002 (Node_017 improved dur>180s primary by about +0.0029 despite only +0.0001 fresh-seed mean overall; this changed stack contains a causal session branch absent from the earlier long-duration-specialist test, while the card's measured record caps expectation at +0.0003.)
**Result:** GAUC 0.6699 · nDCG@5 0.5368 · primary 0.6033 · realized Δ -0.0010 · rejected
**Diff:** `diffs/020.patch` (15 changed lines) · duration 270s · tokens in/out 76465/7271 · intervention: False

### n=21 — node_021 (deepen, parent 14)
**Hypothesis:** Replace one of the five field-aware members with a BPR-trained attentional FM so row-conditioned interaction weighting contributes a small third error pattern while the other nine members anchor the ensemble.
**Method:** ensembling-multiseed-heterogeneous-rank-blend — row-conditioned-attention member diversity · target `model` · expected Δ 0.0001 (The AFM card's analogy promise calibrates to at most +0.0003 and its pointwise node_005 regressed, so only a lower-third diversity gain is credible when one member is introduced under the accepted heterogeneous BPR blend.)
**Result:** GAUC 0.6705 · nDCG@5 0.5374 · primary 0.6039 · realized Δ -0.0004 · rejected
**Diff:** `diffs/021.patch` (85 changed lines) · duration 170s · tokens in/out 76143/8427 · intervention: False

#### generation 6 closed — no improvement; streak 2; champion node_014; best 0.6046; tokens 625228/56834; 876s
_Diagnosis:_ Dynamics: mild overfit—primary peaks at epoch 6 (0.6045) then slips to 0.6043 at epoch 7; champion GAUC/nDCG@5/ndcg5_disc = 0.6710/0.5377/0.7246.
Node 015 was dead overall: GAUC +0.0001, nDCG@5 −0.0002, mixed-user nDCG −0.0002.
Node 016 moved GAUC +0.0008 but nDCG@5 −0.0002 (mixed −0.0003); extra averaging improved pair ordering only on seed 0 and reversed across fresh seeds.
Node 017 moved GAUC +0.0002 and nDCG@5 +0.0003 (mixed +0.0006); node 018 moved GAUC +0.0003 but essentially no nDCG@5 (+0.0000, mixed +0.0001); neither confirmed.
Next probe: conservative APR on field-aware members, targeting BPR regularization; Fact §2’s median 35-row user history supports robustifying sparse-ID embeddings while retaining the standard branch as anchor.
Validation-selection risk is high: streak 1/3 and all 4 accepted wins have fresh-seed gains below 0.002.
Weakness remains dur>180s and tabs 4/6 beyond composition-driven tab0/dur0; node 015 helped dur>180s +0.0022/tab4 +0.0020, node 017 +0.0029/+0.0097 on dur>180s/tab6, but both regressed other groups; no groups are HARD.
The core error is different-day tab1×tab1 mass (~69%): nodes 016–018 each showed only −0.001 pair error and failed confirmation, while node 017 worsened gap<10min by +0.011; ensembling variants touched narrow mechanisms only, so close that campaign.
_Plan for next generation:_ `{"note": "No merge qualifies: all generation-6 candidates regressed or were flat, and the ensembling campaign is now closed after two flat generations.\nUse one untried regularization probe on the champion; leave two slots to the Selector because session specialization and attention-member diversity are closed.", "plan": [{"type": "explore", "parent": 14, "card": "regularization-adversarial-personalized-ranking", "hypothesis": "Apply conservative APR regularization (epsilon=0.05, adversarial weight=0.5) only to the five field-aware BPR members while retaining the five standard members as an unchanged anchor, testing whether robust pairwise embeddings reduce the champion's mild post-epoch-6 ordering decline without sacrificing architecture diversity."}]}`

### n=22 — node_022 (explore, parent 14)
**Hypothesis:** Add a TimeSVD++-style tab-conditioned temporal bias for each candidate's age since upload to the five field-aware BPR members, allowing freshness effects to change across impression dates while the unchanged standard branch anchors ID-based rankings.
**Method:** model-timesvd-upload-age-bias · target `model` · expected Δ 0.0001 (Koren's TimeSVD++ (KDD 2009) and the champion's 62% pair mass at gaps over one day motivate candidate-specific temporal dynamics, but Fact §11 caps video-side information near +0.0003, so the estimate uses its lower third.)
**Result:** GAUC 0.6712 · nDCG@5 0.5377 · primary 0.6044 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6044367404567378, 'node_seeds': [0.6043937077209951, 0.6043594919879425, 0.6045320965975863], 'champion_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'delta_mean': -0.00015, 'se': 0.000293, 'z': -0.51, 'sigma_pooled': 0.000358, 'sigma_df': 32, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/022.patch` (106 changed lines) · duration 165s · tokens in/out 77207/11417 · intervention: False

### n=23 — node_023 (improve, parent 14)
**Hypothesis:** Apply a zero-initialized AFM residual to every standard-FM BPR member while retaining the field-aware branch as an unchanged anchor, testing whether branch-wide row-conditioned interaction weighting is more stable than the rejected one-slot attention replacement.
**Method:** model-attentional-factorization-machine · target `model` · expected Δ 0.0001 (The card's unmeasured upper bound of +0.0010 calibrates to at most +0.0003 under ADR-0018, and the pointwise AFM and one-member attention failures justify a lower-third estimate of +0.0001.)
**Result:** GAUC 0.6710 · nDCG@5 0.5378 · primary 0.6044 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6043980536117246, 'node_seeds': [0.6048775273129443, 0.6043476760262603, 0.6047972085599997], 'champion_seeds': [0.6046725943624534, 0.6043597781166279, 0.604702099707588], 'delta_mean': 0.0001, 'se': 0.00029, 'z': 0.33, 'sigma_pooled': 0.000355, 'sigma_df': 34, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/023.patch` (87 changed lines) · duration 282s · tokens in/out 77227/10063 · intervention: False

### n=24 — node_024 (deepen, parent 14)
**Hypothesis:** Add one independently stopped DCN-BPR lineage at low fixed rank weight to the existing standard/field-aware blend, seeking higher-order model diversity without perturbing either accepted branch.
**Method:** model-dcn-cross-head · target `ensembling` · expected Δ 0.0002 (The DCN card's measured record is +0.0005 and its prior heterogeneous-stack attempt reached +0.0004 at z=1.47, supporting a lower-third +0.0002 estimate for an independently fused lineage.)
**Result:** GAUC 0.6710 · nDCG@5 0.5377 · primary 0.6043 · realized Δ -0.0000 · rejected
**Diff:** `diffs/024.patch` (82 changed lines) · duration 174s · tokens in/out 79237/9218 · intervention: False

#### generation 7 closed — no improvement; streak 3; champion node_014; best 0.6046; tokens 577557/53467; 1225s
_Diagnosis:_ Dynamics: mild overfit; primary peaks at epoch 6 (0.6045) then slips to 0.6043 at epoch 7; champion GAUC/nDCG@5/ndcg5_disc = 0.6710/0.5377/0.7246.
Node 019 was a trade-off: GAUC +0.0001, nDCG@5 −0.0002 (disc −0.0003), hence no net movement.
Node 020 regressed both halves: GAUC −0.0011 and nDCG@5 −0.0009 (disc −0.0016); node 021 likewise fell GAUC −0.0005 and nDCG@5 −0.0003 (disc −0.0005).
Most informative next probe: conservative APR on field-aware members, targeting regularization; Fact §2’s median 35-row user history and the post-epoch-6 decline motivate robustifying sparse-ID BPR embeddings while retaining the standard branch as anchor.
Validation-overfitting risk is high: streak 2/3, and all four accepted wins had fresh-seed gains below 0.002.
Actionable weakness is dur>180s (primary 0.5030) and tabs 4/6 beyond composition effects: node 020 gained +0.0008/+0.0004/+0.0011 there, and node 021 +0.0013/−0.0005/+0.0022, but both regressed broadly; no groups are HARD.
The dominant error remains different-day tab1×tab1 pairs (~69% mass): node 019 improved them only ≈0.001, while nodes 020–021 were flat there and mainly touched gap<10min pairs (2% mass), so the core error was effectively unmoved.
Model campaign: attention-member diversity produced narrow dur>180s/tab6 gains but reduced both metrics overall; with earlier AFM failure, this mechanism should close rather than stay open.
_Plan for next generation:_ `{"note": "No generation-7 node was accepted or improved the champion by >= 0.001; upload-age, AFM residual, and DCN lineage changes were flat across seeds.\nThe non-improving streak has reached 3, triggering ADR-0012 convergence, so the run stops and no next-generation slots should be scheduled.", "plan": []}`

- _event_ (generation 7): designation (strict): node_017 leads on fresh-seed mean (0.60469) but was not accepted; excluded — accepted lineage only
