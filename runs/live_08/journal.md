# Run journal — live_08

## Summary
```json
{
 "run_id": "live_08",
 "stop_reason": "wall-clock 21600s",
 "generations": 4,
 "nodes": 16,
 "champion": 6,
 "champion_metrics": {
  "gauc": 0.6708002088480625,
  "ndcg5": 0.5374674075413834,
  "primary": 0.604133808194723,
  "ndcg5_disc": 0.7242561821141262,
  "by_group": {
   "dur18-60s": {
    "rows": 33235,
    "gauc": 0.6434,
    "ndcg5": 0.46,
    "primary": 0.5517
   },
   "dur60-180s": {
    "rows": 46565,
    "gauc": 0.6509,
    "ndcg5": 0.488,
    "primary": 0.5695
   },
   "dur<18s": {
    "rows": 20101,
    "gauc": 0.7033,
    "ndcg5": 0.4055,
    "primary": 0.5544
   },
   "dur=0": {
    "rows": 2107,
    "gauc": 0.5921,
    "ndcg5": 0.0863,
    "primary": 0.3392
   },
   "dur>180s": {
    "rows": 22901,
    "gauc": 0.6293,
    "ndcg5": 0.3779,
    "primary": 0.5036
   },
   "tab=0": {
    "rows": 13726,
    "gauc": 0.5676,
    "ndcg5": 0.0633,
    "primary": 0.3154
   },
   "tab=1": {
    "rows": 92672,
    "gauc": 0.6211,
    "ndcg5": 0.5414,
    "primary": 0.5813
   },
   "tab=2": {
    "rows": 3834,
    "gauc": 0.6435,
    "ndcg5": 0.5148,
    "primary": 0.5791
   },
   "tab=4": {
    "rows": 7877,
    "gauc": 0.5795,
    "ndcg5": 0.5479,
    "primary": 0.5637
   },
   "tab=6": {
    "rows": 5170,
    "gauc": 0.6459,
    "ndcg5": 0.106,
    "primary": 0.376
   }
  },
  "by_pair": {
   "same_tab": {
    "share": 0.735,
    "err": 0.38,
    "contrib": 0.2793
   },
   "diff_tab": {
    "share": 0.265,
    "err": 0.188,
    "contrib": 0.0499
   },
   "tab1_x_tab1": {
    "share": 0.686,
    "err": 0.379,
    "contrib": 0.2601
   },
   "same_date": {
    "share": 0.236,
    "err": 0.335,
    "contrib": 0.0792
   },
   "diff_date": {
    "share": 0.764,
    "err": 0.327,
    "contrib": 0.25
   },
   "gap>1d": {
    "share": 0.618,
    "err": 0.326,
    "contrib": 0.2018
   },
   "gap<10min": {
    "share": 0.02,
    "err": 0.34,
    "contrib": 0.0068
   },
   "pos_shorter": {
    "share": 0.513,
    "err": 0.299,
    "contrib": 0.1532
   },
   "pos_longer": {
    "share": 0.485,
    "err": 0.362,
    "contrib": 0.1758
   },
   "total_err": 0.3292
  }
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00267,
 "top3_valid": [
  {
   "n": 12,
   "primary": 0.6043100060222562
  },
  {
   "n": 11,
   "primary": 0.6042946856591713
  },
  {
   "n": 10,
   "primary": 0.6042785207152845
  }
 ],
 "designated": 6,
 "final_ranking": [
  {
   "n": 6,
   "valid_primary": 0.604133808194723,
   "fresh_seeds": [
    0.6043982388227487,
    0.6041081160515795,
    0.6043710216628997
   ],
   "accepted": true,
   "mean": 0.6042924588457427,
   "std": 0.00016022450738714482,
   "n_seeds": 3
  },
  {
   "n": 7,
   "valid_primary": 0.6037623529690725,
   "fresh_seeds": [
    0.6035022487083593,
    0.6039991791668944,
    0.6041407323251042
   ],
   "accepted": true,
   "mean": 0.603880720066786,
   "std": 0.0003353203707080456,
   "n_seeds": 3
  },
  {
   "n": 5,
   "valid_primary": 0.6036025872235276,
   "fresh_seeds": [
    0.6037850265789089,
    0.6032429221606551,
    0.6036001948959435,
    0.6037473923506763,
    0.6035326876994711
   ],
   "accepted": true,
   "mean": 0.6035816447371309,
   "std": 0.00021581840092065174,
   "n_seeds": 5
  },
  {
   "n": 8,
   "valid_primary": 0.6039525877444525,
   "fresh_seeds": [
    0.6036597883114984,
    0.6033329940719132,
    0.6035348619903409,
    0.6036895414211436,
    0.6034873877878435
   ],
   "accepted": true,
   "mean": 0.6035409147165479,
   "std": 0.0001434865768563354,
   "n_seeds": 5
  },
  {
   "n": 3,
   "valid_primary": 0.6029172490595951,
   "fresh_seeds": [
    0.6029694019136509,
    0.6024001158605996,
    0.602831949124412
   ],
   "accepted": true,
   "mean": 0.6027338222995542,
   "std": 0.00029705775173724107,
   "n_seeds": 3
  },
  {
   "n": 1,
   "valid_primary": 0.6030457178505284,
   "fresh_seeds": [
    0.6028506298315512,
    0.602047857756379,
    0.6025559918563459
   ],
   "accepted": true,
   "mean": 0.6024848264814254,
   "std": 0.0004060900566496989,
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
   "n": 12,
   "valid_primary": 0.6043100060222562,
   "fresh_seeds": [
    0.6046157958166243,
    0.6044273740133262,
    0.6048999492748063
   ],
   "accepted": false,
   "mean": 0.6046477063682523,
   "std": 0.00023789820705426255,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  },
  {
   "n": 10,
   "valid_primary": 0.6042785207152845,
   "fresh_seeds": [
    0.6045178124394952,
    0.6042453883304504,
    0.6045052844844434
   ],
   "accepted": false,
   "mean": 0.6044228284181297,
   "std": 0.0001537952403397722,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  },
  {
   "n": 11,
   "valid_primary": 0.6042946856591713,
   "fresh_seeds": [
    0.6044102022008228,
    0.6044464948235186,
    0.6043860673978856
   ],
   "accepted": false,
   "mean": 0.604414254807409,
   "std": 3.0416872904143558e-05,
   "n_seeds": 3,
   "excluded": "not accepted (strict designation: the run never submits a node it rejected)"
  }
 ],
 "usage": {
  "calls": 54,
  "tokens_in": 2172098,
  "tokens_out": 202532,
  "cache_read": 1272707,
  "cache_write": 0,
  "cost_usd": 11.209268500000002
 },
 "wall_clock_s": 32213.1,
 "champion_seed_mean": 0.60429,
 "best_single_seed": 0.6043100060222562,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": {
  "best_single_seed": 0.604133808194723,
  "streak": 2,
  "converged_at_generation": null,
  "champion_at_stop": null
 },
 "official_rule_submission": {
  "note": "the literal single-seed rule had not converged when the run ended",
  "node": null
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
    2,
    3,
    4
   ],
   "nodes": [
    6,
    9,
    13,
    15
   ],
   "best_gain": 0.00156,
   "flat_streak": 2,
   "evidence": "closed at generation 4: 2 campaign generations without an accepted node (nodes [6, 9, 13, 15], best gain 0.00156)"
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
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
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
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
   "evidence": ""
  },
  "model": {
   "status": "open",
   "generations": [],
   "nodes": [],
   "best_gain": null,
   "flat_streak": 0,
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
  "designation (strict): node_012 leads on fresh-seed mean (0.60465) but was not accepted; excluded \u2014 accepted lineage only"
 ],
 "best_unaccepted": {
  "n": 12,
  "mean": 0.60465,
  "valid_primary": 0.6043100060222562,
  "n_seeds": 3
 },
 "tokens": {
  "in_total": 2172098,
  "in_cached": 1272707,
  "in_uncached": 899391,
  "out": 202532
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
**Diff:** `None` (None changed lines) · duration 15s · tokens in/out 0/0 · intervention: False

- _screen_ (generation 1) DROPPED: `history-gap-inferred-previous-completion` — best_gain -0.0001 (inferred_previous_completion); stack -0.0010; previous_gap_seconds: varies 0.818, GAUC 0.508, additive -0.0004; previous_threshold_seconds: varies 0.476, GAUC 0.5009, additive -0.0011; gap_minus_threshold_seconds: varies 0.818, GAUC 0.508, additive -0.0004; previous_same_author: varies 0.03, GAUC 0.5003, additive -0.0008; inferred_previous_completion: varies 0.365, GAUC 0.5072, additive -0.0001; inferred_previous_state: varies 0.371, GAUC 0.5071, additive -0.0003; state_same_author_cross: varies 0.377, GAUC 0.5074, additive -0.0004

### n=1 — node_001 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with same-user BPR so training directly optimizes the positive-negative ordering measured by GAUC.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0016 (Foundations §3 and the diagnosis identify BPR as the most informative probe, with a previously confirmed +0.0016 gain on the official FM stack.)
**Result:** GAUC 0.6694 · nDCG@5 0.5367 · primary 0.6030 · realized Δ +0.0016 · ACCEPTED · seed confirmation {'node_seed0': 0.6030457178505284, 'node_seeds': [0.6028506298315512, 0.602047857756379, 0.6025559918563459], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00103, 'se': 0.000277, 'z': 3.73, 'sigma_pooled': 0.000339, 'sigma_df': 4, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/001.patch` (33 changed lines) · duration 14s · tokens in/out 68232/5321 · intervention: False

### n=2 — node_002 (improve, parent 0)
**Hypothesis:** Raise embedding L2 from 1e-6 to 1e-5 to reduce the sparse-ID overfitting visible after epoch 7 without altering the FM representation.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0009 (The champion declines from 0.6015 at epoch 7 to 0.5990 at epoch 11, and this card has repeatedly produced +0.0009 to +0.0010 on the official FM stack.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6025029206269998, 'node_seeds': [0.6028729444838677, 0.6017090117891591, 0.6020694898356386, 0.6027857969882158, 0.6026234157272281], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00096, 'se': 0.000293, 'z': 3.28, 'sigma_pooled': 0.000401, 'sigma_df': 8, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/002.patch` (2 changed lines) · duration 26s · tokens in/out 66879/3384 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Average within-user normalized ranks from five independently seeded copies of the exact champion to cancel seed-specific ordering errors.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0013 (Five-seed averaging was accepted twice on the official pointwise FM with fresh-seed mean gains of +0.0013.)
**Result:** GAUC 0.6692 · nDCG@5 0.5367 · primary 0.6029 · realized Δ +0.0014 · ACCEPTED · seed confirmation {'node_seed0': 0.6029172490595951, 'node_seeds': [0.6029694019136509, 0.6024001158605996, 0.602831949124412], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00128, 'se': 0.000317, 'z': 4.05, 'sigma_pooled': 0.000388, 'sigma_df': 10, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/003.patch` (71 changed lines) · duration 61s · tokens in/out 68498/8323 · intervention: False

### n=4 — node_004 (explore, parent 0)
**Hypothesis:** Add a small NFM residual over the FM bi-interaction vector so latent interaction dimensions can combine nonlinearly while training starts close to the baseline.
**Method:** model-neural-factorization-machine · target `model` · expected Δ 0.0002 (The untried card's +0.0008 upper prior discounts to at most +0.00024 under ADR-0018, and +0.0002 lies in the lower third while respecting the observed small architecture gains.)
**Result:** GAUC 0.6666 · nDCG@5 0.5356 · primary 0.6011 · realized Δ -0.0004 · rejected
**Diff:** `diffs/004.patch` (46 changed lines) · duration 17s · tokens in/out 67820/7415 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_003; best 0.6027; tokens 429590/37784; 504s
_Diagnosis:_ Dynamics: overfitting—primary peaks at epoch 7 (0.6015), then declines steadily to 0.5990 by epoch 11; best epoch is not the final epoch.
Champion metrics: GAUC 0.6671, nDCG@5 0.5358, mixed-user nDCG@5 0.7214.
No generation-1 child nodes were run, so there are no GAUC/nDCG half-movements or by-group/by-pair deltas to attribute.
Most informative next probe: within-user BPR loss, targeting the loss component; Fact 3 shows it aligns directly with GAUC and previously delivered a seed-confirmed +0.0016.
Validation-overfitting risk is currently low but nonzero: streak 0 and zero sub-0.002 wins accepted in this run, though selecting among many small probes will increase winner’s-curse risk.
Weak group maps include tab=0 and dur=0, but their near-zero nDCG largely reflects all-negative users; the sharper error concentration is tab1×tab1 (69% pair mass, 0.383 misordered), especially across dates/gap>1d, and this generation did not move it.
No groups are HARD and no campaign family is named.
_Plan for next generation:_ `{"note": "Three distinct components were accepted; the highest-value consolidation is to carry the proven loss and regularization edits onto the five-seed champion.\nBPR and stronger L2 should be tested separately on the ensemble because prior evidence suggests their gains may not compose; remaining slots are left to the Selector.", "plan": [{"type": "merge", "merge_parents": [1, 3], "hypothesis": "Train each of the champion's five independently seeded FM members with same-user BPR, then average tie-free within-user normalized ranks; BPR improves systematic pair ordering while seed averaging removes initialization and sampling variance."}, {"type": "merge", "merge_parents": [2, 3], "hypothesis": "Apply embedding L2=1e-5 independently to every member of the five-seed rank ensemble; shrinkage should reduce sparse-ID overfitting within each member while rank averaging preserves complementary ordering errors."}]}`

### n=5 — node_005 (explore, parent 3)
**Hypothesis:** Blend five-seed pointwise-FM and BPR-FM rank ensembles with a candidate-tab-specific gate derived from the user's prior positive and negative support, favoring BPR where pairwise evidence is reliable and pointwise training where it is sparse or single-class.
**Method:** history-confidence-gated-pointwise-bpr-ensemble · target `ensembling` · expected Δ 0.0004 (Fact §2 reports only 35 median training rows per user, while the journal shows BPR improves GAUC much more than nDCG and cannot train on single-class histories, making a support-gated mixture a plausible lower-third ensembling gain.)
**Result:** GAUC 0.6704 · nDCG@5 0.5368 · primary 0.6036 · realized Δ +0.0007 · ACCEPTED · seed confirmation {'node_seed0': 0.6036025872235276, 'node_seeds': [0.6037850265789089, 0.6032429221606551, 0.6036001948959435, 0.6037473923506763, 0.6035326876994711], 'champion_seeds': [0.6029694019136509, 0.6024001158605996, 0.602831949124412], 'delta_mean': 0.00085, 'se': 0.00026, 'z': 3.25, 'sigma_pooled': 0.000357, 'sigma_df': 14, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/005.patch` (85 changed lines) · duration 138s · tokens in/out 71864/9596 · intervention: False

### n=6 — node_006 (improve, parent 3)
**Hypothesis:** Replace the homogeneous ensemble with the proven ten-model 0.6 field-aware-FM-BPR and 0.4 standard-FM-BPR within-user rank blend so architecture and sampling errors cancel.
**Method:** ensembling-multiseed-heterogeneous-rank-blend · target `ensembling` · expected Δ 0.001 (The card recorded a +0.0017 fresh-seed gain and an absolute primary of 0.6045, but node_003 already captures homogeneous seed averaging, so the estimate discounts overlap.)
**Result:** GAUC 0.6708 · nDCG@5 0.5375 · primary 0.6041 · realized Δ +0.0012 · ACCEPTED · seed confirmation {'node_seed0': 0.604133808194723, 'node_seeds': [0.6043982388227487, 0.6041081160515795, 0.6043710216628997], 'champion_seeds': [0.6029694019136509, 0.6024001158605996, 0.602831949124412], 'delta_mean': 0.00156, 'se': 0.000279, 'z': 5.58, 'sigma_pooled': 0.000342, 'sigma_df': 16, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/006.patch` (118 changed lines) · duration 179s · tokens in/out 72808/11317 · intervention: False

### n=7 — node_007 (merge, parent 1, merge of [1, 3])
**Hypothesis:** Train every member of the five-seed champion with same-user BPR so systematic pair ordering improves while rank averaging removes pair-sampling and initialization variance.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.001 (Node_001 gained +0.00103 on fresh seeds, and the diagnosis reports that BPR reduced errors on tab1×tab1 and gap>1d pairs by 0.002 and 0.003 respectively.)
**Result:** GAUC 0.6701 · nDCG@5 0.5374 · primary 0.6038 · realized Δ +0.0008 · ACCEPTED · seed confirmation {'node_seed0': 0.6037623529690725, 'node_seeds': [0.6035022487083593, 0.6039991791668944, 0.6041407323251042], 'champion_seeds': [0.6029694019136509, 0.6024001158605996, 0.602831949124412], 'delta_mean': 0.00115, 'se': 0.000279, 'z': 4.11, 'sigma_pooled': 0.000342, 'sigma_df': 18, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/007.patch` (73 changed lines) · duration 59s · tokens in/out 148040/8937 · intervention: False

### n=8 — node_008 (merge, parent 2, merge of [2, 3])
**Hypothesis:** Use L2=1e-5 in every member of the five-seed ensemble so sparse-ID shrinkage reduces each model's overfitting while rank averaging preserves complementary errors.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0005 (Node_002 gained +0.00096 on fresh seeds and the card's official-FM record is +0.0010, discounted because both shrinkage and averaging mainly improved broad GAUC ordering.)
**Result:** GAUC 0.6704 · nDCG@5 0.5375 · primary 0.6040 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6039525877444525, 'node_seeds': [0.6036597883114984, 0.6033329940719132, 0.6035348619903409, 0.6036895414211436, 0.6034873877878435], 'champion_seeds': [0.6029694019136509, 0.6024001158605996, 0.602831949124412], 'delta_mean': 0.00081, 'se': 0.000233, 'z': 3.46, 'sigma_pooled': 0.000319, 'sigma_df': 22, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/008.patch` (71 changed lines) · duration 118s · tokens in/out 72142/3894 · intervention: False

### n=9 — node_009 (deepen, parent 3)
**Hypothesis:** Average each seed's best checkpoint with its immediately preceding checkpoint before the cross-seed rank average to smooth epoch-level ordering noise without training more models.
**Method:** ensembling-seed-average · target `training-schedule` · expected Δ 0.0001 (Checkpoint weight averaging has measured at most +0.0001, and the champion's primary is flat from epochs 7 through 11, so only a small variance-reduction effect is credible.)
**Result:** GAUC 0.6683 · nDCG@5 0.5364 · primary 0.6023 · realized Δ -0.0006 · rejected
**Diff:** `diffs/009.patch` (25 changed lines) · duration 99s · tokens in/out 70438/7246 · intervention: False

#### generation 2 closed — improved; streak 0; champion node_006; best 0.6043; tokens 602802/61337; 1402s
_Diagnosis:_ Dynamics: flat after convergence—primary rises from 0.5883 at epoch 1 to 0.6029 at epoch 7, then stays 0.6029 through epoch 11; champion GAUC/nDCG@5/ndcg5_disc = 0.6692/0.5367/0.7228.
BPR moved mainly GAUC: +0.0023 versus node_000, with nDCG@5 +0.0009 and mixed-user nDCG +0.0015; it improves broad pair ordering more than the top five.
Stronger L2 moved GAUC +0.0014 and nDCG@5 +0.0007 (mixed +0.0012), again primarily the GAUC half.
Seed averaging moved GAUC +0.0021 and nDCG@5 +0.0009 (mixed +0.0014), a broad variance-reduction gain.
NFM moved neither half: GAUC −0.0005, nDCG@5 −0.0002, mixed −0.0005; this model head is dead on the pointwise stack.
Most informative next probe: merge BPR into the five-seed ensemble, targeting loss; Fact 3 establishes same-user BPR as directly aligned with GAUC while averaging can remove its sampling variance.
Weakness is dominant tab1×tab1 ordering (69% pair mass, 0.382 misordered), especially gap>1d; BPR improved these by −0.002/−0.003, while NFM worsened them +0.002/+0.001. Tab 4 has weak GAUC, and BPR moved it +0.0073; long-duration ordering remains weak and BPR worsened pos-longer pairs.
Validation-overfitting risk is rising but currently moderate: streak 0, yet three sub-0.002 wins were accepted. The ensembling campaign should stay open for the BPR merge; seed averaging moved both halves and the dominant pair type, while no other ensemble mechanism was tested.
_Plan for next generation:_ `{"note": "Nodes 006 and 008 improved the prior champion by at least 0.001 in distinct components, so stronger L2 should be merged into every branch of the heterogeneous multiseed champion.\nDCN previously worked on field-aware FM; the new five-seed field-aware BPR branch provides a materially changed, variance-reduced context for one targeted retest.", "plan": [{"type": "merge", "merge_parents": [6, 8], "hypothesis": "Apply L2=1e-5 to every standard and field-aware FM-BPR member of the champion's ten-model 0.6/0.4 rank blend, combining sparse-ID shrinkage with architecture and seed variance reduction."}, {"type": "retest", "parent": "champion", "card": "model-dcn-cross-head", "hypothesis": "Add the shallow DCN cross residual only to the champion's five field-aware FM-BPR members while retaining the standard branch and 0.6/0.4 rank fusion, testing whether explicit higher-order interactions contribute complementary ordering errors.", "reason": "DCN was accepted on a field-aware stack but its prior heterogeneous test used a single-node rank average; the current independently stopped five-seed field-aware branch can stabilize its small architecture gain and is a changed stack under ADR-0004."}]}`

- _screen_ (generation 3) DROPPED: `ensembling-heterogeneous-rank-average — support-gated pointwise third branch` — best_gain -0.0000 (user_tab_positive_support); stack -0.0008; user_tab_positive_support: varies 0.371, GAUC 0.5565, additive -0.0000; user_tab_negative_support: varies 0.382, GAUC 0.5154, additive -0.0001

### n=10 — node_010 (explore, parent 6)
**Hypothesis:** Fuse the champion with a small empirical rank expert based on the current video's long-view threshold and the cumulative threshold demand of strictly earlier impressions in the session, capturing duration-weighted attention depletion that static duration and tab IDs miss.
**Method:** attention-budget-threshold-expert-fusion · target `ensembling` · expected Δ 0.0004 (Facts §3 and §10.5 show that long_view requires up to 18 seconds of attention while recent impression density changes P(long_view) from 0.413 to 0.120; the estimate is conservative because ordinary session features added only +0.0002 on a seed blend.)
**Result:** GAUC 0.6709 · nDCG@5 0.5376 · primary 0.6043 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6042785207152845, 'node_seeds': [0.6045178124394952, 0.6042453883304504, 0.6045052844844434], 'champion_seeds': [0.6043982388227487, 0.6041081160515795, 0.6043710216628997], 'delta_mean': 0.00013, 'se': 0.000253, 'z': 0.51, 'sigma_pooled': 0.00031, 'sigma_df': 24, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/010.patch` (71 changed lines) · duration 215s · tokens in/out 150602/22240 · intervention: False

### n=11 — node_011 (merge, parent 6, merge of [6, 8])
**Hypothesis:** Apply L2=1e-5 to every standard and field-aware BPR member so sparse-ID shrinkage repairs long-positive and multi-day ordering errors while heterogeneous rank averaging preserves complementary errors.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0005 (Node_008 gained +0.00081 on fresh seeds and reduced pos-longer and >1-day pair errors, but overlap with the champion's variance reduction warrants a threshold-level +0.0005 estimate.)
**Result:** GAUC 0.6710 · nDCG@5 0.5376 · primary 0.6043 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.6042946856591713, 'node_seeds': [0.6044102022008228, 0.6044464948235186, 0.6043860673978856], 'champion_seeds': [0.6043982388227487, 0.6041081160515795, 0.6043710216628997], 'delta_mean': 0.00012, 'se': 0.000245, 'z': 0.5, 'sigma_pooled': 0.0003, 'sigma_df': 26, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/011.patch` (2 changed lines) · duration 223s · tokens in/out 74329/4202 · intervention: False

### n=12 — node_012 (retest, parent 6)
**Hypothesis:** Add the shallow DCN residual only to the five field-aware BPR members while retaining the standard branch, allowing stabilized higher-order interactions to contribute complementary ranks without perturbing the whole ensemble.
**Method:** model-dcn-cross-head · target `model` · expected Δ 0.0003 (DCN recorded a +0.0005 seed-mean gain on a field-aware stack, but its earlier heterogeneous follow-up was negative and Fact §11 bounds architecture gains near acceptance scale.)
**Result:** GAUC 0.6711 · nDCG@5 0.5375 · primary 0.6043 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.6043100060222562, 'node_seeds': [0.6046157958166243, 0.6044273740133262, 0.6048999492748063], 'champion_seeds': [0.6043982388227487, 0.6041081160515795, 0.6043710216628997], 'delta_mean': 0.00036, 'se': 0.000242, 'z': 1.47, 'sigma_pooled': 0.000296, 'sigma_df': 28, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/012.patch` (57 changed lines) · duration 231s · tokens in/out 72460/9031 · intervention: False

### n=13 — node_013 (deepen, parent 6)
**Hypothesis:** Make one field-aware member a same-tab BPR specialist while leaving the other nine members uniformly sampled, turning the dominant same-tab error into ensemble diversity rather than imposing context matching on every model.
**Method:** ensembling-multiseed-heterogeneous-rank-blend — same-tab sampler-diverse member · target `loss` · expected Δ 0.0002 (Same-tab pairs carry 73% of GAUC mass and tab1×tab1 alone carries 69%, but earlier globally context-matched BPR was weak, so a single-member specialization merits only a conservative +0.0002 expectation.)
**Result:** GAUC 0.6707 · nDCG@5 0.5375 · primary 0.6041 · realized Δ -0.0000 · rejected
**Diff:** `diffs/013.patch` (21 changed lines) · duration 210s · tokens in/out 72108/5803 · intervention: False

#### generation 3 closed — no improvement; streak 1; champion node_006; best 0.6043; tokens 675429/71701; 1516s
_Diagnosis:_ Dynamics are flat: primary rose to 0.6041 at epoch 8 and remained 0.6041 through epoch 13, with no late decline.
Champion metrics are GAUC 0.6708, nDCG@5 0.5375, and mixed-user nDCG@5 0.7243.
Versus node_003, node_005 moved almost only GAUC (+0.0013; nDCG +0.0001, mixed +0.0002); node_006 moved both GAUC +0.0016 and nDCG +0.0008 (mixed +0.0015).
Node_007 moved GAUC +0.0009/nDCG +0.0007; node_008 +0.0012/+0.0008; snapshot smoothing node_009 hurt both, −0.0009/−0.0003, and is dead.
Next probe: merge L2=1e-5 into all champion branches, targeting regularization; Fact §11 shows capacity adds little, while node_008 reduced long-positive and >1-day pair errors.
Validation-overfitting risk is elevated despite streak 0: seven sub-0.002 wins have already been accepted, increasing selection exposure.
Meaningful weaknesses are dur>180s and tab=4 GAUC, not near-zero tab=0/dur=0 nDCG; node_006 improved them +0.0035/+0.0036, while node_008 repaired pos-longer pairs (−0.002). No HARD groups.
Most error remains tab1×tab1 and >1-day pairs; nodes 005–008 reduced these by 0.002–0.003, while node_009 worsened tab1×tab1 by 0.001. Ensembling should stay open because heterogeneous blending moved the dominant mass; snapshot smoothing did not.
_Plan for next generation:_ `{"note": "No generation-3 node was accepted or improved the champion by >= 0.001, so no merge qualifies; DCN and the attention-budget expert nevertheless expose actionable transient and directional gains.\nDeepen those two near-misses, and use one exploration slot for WARP because the generation was non-improving and top-five movement remains weaker than GAUC movement.", "plan": [{"type": "deepen", "parent": 12, "card": "model-dcn-cross-head", "mechanism": "synchronized-ensemble-epoch-selection", "target_group": "all", "hypothesis": "Checkpoint the complete ten-member fused ensemble at the epoch maximizing ensemble validation primary, rather than restoring each DCN and standard member by its individual optimum; node_012 transiently reached 0.6046 at epochs 6\u20137 but emitted 0.6043, indicating that independently selected checkpoints destroy complementary ordering."}, {"type": "deepen", "parent": 10, "card": "attention-budget-threshold-expert-fusion", "mechanism": "short-threshold-directional-gating", "target_group": "pos_shorter", "hypothesis": "Apply the attention-budget expert only to candidates whose long-view threshold is below their user's scored-set median, with zero expert weight otherwise; node_010 reduced pos_shorter misordering by 0.001 but increased pos_longer misordering by 0.001, so directional gating should retain the useful half without the symmetric reversal."}, {"type": "explore", "parent": 6, "card": "loss-warp-within-user-rank-weighting", "hypothesis": "Replace uniform BPR with WARP in one standard-FM ensemble member only, using repeated same-user violation search and harmonic estimated-rank weighting to add top-of-list error diversity while preserving the other nine proven BPR members."}]}`

### n=14 — node_014 (explore, parent 6)
**Hypothesis:** Replace the fixed 0.6/0.4 field-aware/standard rank blend with a candidate-specific gate that favors field-aware FM when the user-video or user-author relation has training support and favors standard FM for novel or sparsely supported relations, matching each architecture to where its parameter sharing is reliable.
**Method:** relation-support-adaptive-architecture-fusion · target `ensembling` · expected Δ 0.0003 (Facts §1–2 show a closed catalogue but only 35 median rows per user, so candidate-specific relation support varies sharply, while the journal's heterogeneous blend gained +0.0016 seed-mean and leaves room for a conservative lower-third improvement from adapting its fixed architecture weight.)
**Result:** GAUC 0.6709 · nDCG@5 0.5376 · primary 0.6042 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6042155584886066, 'node_seeds': [0.6044234674148132, 0.6040886464687556, 0.604322000818103], 'champion_seeds': [0.6043982388227487, 0.6041081160515795, 0.6043710216628997], 'delta_mean': -1e-05, 'se': 0.000237, 'z': -0.06, 'sigma_pooled': 0.000291, 'sigma_df': 30, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/014.patch` (33 changed lines) · duration 149s · tokens in/out 73920/7557 · intervention: False

### n=15 — node_015 (deepen, parent 10)
**Hypothesis:** Apply node_010's attention-budget expert only to candidates with below-user-median long-view thresholds, retaining its improvement when the positive is shorter while avoiding the equal harm when the positive is longer.
**Method:** ensembling-cohort-gated-session-rank-fusion — short-threshold directional gating · target `ensembling` · expected Δ 0.0003 (The generation-3 diagnosis reports that node_010 reduced pos_shorter misordering by 0.001 but increased pos_longer misordering by 0.001, and the measured card record caps the expected gain at +0.0005.)
**Result:** GAUC 0.6708 · nDCG@5 0.5375 · primary 0.6041 · realized Δ -0.0000 · rejected
**Diff:** `diffs/015.patch` (32 changed lines) · duration 152s · tokens in/out 75361/8132 · intervention: False

#### generation 4 closed — no improvement; streak 2; champion node_006; best 0.6043; tokens 464277/31710; 28776s
_Diagnosis:_ Dynamics are flat: primary reaches 0.6041 at epoch 8 and remains 0.6041 through epoch 13; champion GAUC/nDCG@5/mixed-user nDCG are 0.6708/0.5375/0.7243.
Node 010 moved both halves only marginally: GAUC +0.0001, nDCG +0.0001, mixed nDCG +0.0002.
Node 011 was mostly GAUC: +0.0002 GAUC, +0.0001 nDCG, +0.0001 mixed nDCG; node 012 was GAUC-only at +0.0003, with nDCG and mixed nDCG flat.
Node 013 moved neither half: GAUC −0.0001 and nDCG/mixed nDCG effectively flat, so the same-tab specialist is dead here.
Most informative next probe is synchronized ensemble-epoch checkpointing for node 012’s model/training selection: it transiently reached 0.6046 at epoch 7 but emitted 0.6043; Fact §11 says architecture headroom is tiny, making preservation of complementary ranks more credible than added capacity.
Validation-overfitting risk is elevated: streak 1, with seven accepted wins below 0.002 already exposing selection to winner’s curse.
Meaningful weak areas are dur>180s and tab=4 GAUC; node 011 improved dur>180s +0.0020, while no node repaired tab=4. The dominant error remains tab1×tab1 and gap>1d, and no node moved either; node 013 only improved gap<10min by 0.002, just 2% of pair mass.
The ensembling campaign produced only node 010’s pos-shorter improvement (−0.001) offset by pos-longer harm (+0.001), while same-tab specialization was flat; keep it open only for targeted directional or synchronized fusion.
_Plan for next generation:_ `{"note": "No generation-4 node improved the champion, and neither qualifies for a cross-component merge; relation-support gating was seed-flat and short-threshold gating is now closed.\nUse one exploration slot on the still-untried ranking-loss family, then leave the other two slots to the Selector because the ensembling campaign is closed.", "plan": [{"type": "explore", "parent": 6, "card": "loss-warp-within-user-rank-weighting", "hypothesis": "Replace uniform BPR with WARP violation search and harmonic estimated-rank weighting in one standard-FM ensemble member only, preserving the other nine proven members while adding top-of-list ordering diversity to address nDCG@5 lag without risking the full champion."}]}`

- _event_ (generation 4): designation (strict): node_012 leads on fresh-seed mean (0.60465) but was not accepted; excluded — accepted lineage only
