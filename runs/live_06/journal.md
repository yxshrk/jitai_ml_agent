# Run journal — live_06

## Summary
```json
{
 "run_id": "live_06",
 "stop_reason": "converged: 3 generations without a >= 0.001 cumulative rise of the champion fresh-seed mean (ADR-0012)",
 "generations": 5,
 "nodes": 21,
 "champion": 7,
 "champion_metrics": {
  "gauc": 0.6706470560076542,
  "ndcg5": 0.5372670890995723,
  "primary": 0.6039570725536132,
  "ndcg5_disc": 0.723909478906422,
  "by_group": {
   "dur18-60s": {
    "rows": 33235,
    "gauc": 0.646,
    "ndcg5": 0.4605,
    "primary": 0.5532
   },
   "dur60-180s": {
    "rows": 46565,
    "gauc": 0.6502,
    "ndcg5": 0.4878,
    "primary": 0.569
   },
   "dur<18s": {
    "rows": 20101,
    "gauc": 0.703,
    "ndcg5": 0.4056,
    "primary": 0.5543
   },
   "dur=0": {
    "rows": 2107,
    "gauc": 0.5572,
    "ndcg5": 0.0859,
    "primary": 0.3215
   },
   "dur>180s": {
    "rows": 22901,
    "gauc": 0.628,
    "ndcg5": 0.3778,
    "primary": 0.5029
   },
   "tab=0": {
    "rows": 13726,
    "gauc": 0.5497,
    "ndcg5": 0.0628,
    "primary": 0.3062
   },
   "tab=1": {
    "rows": 92672,
    "gauc": 0.6208,
    "ndcg5": 0.5412,
    "primary": 0.581
   },
   "tab=2": {
    "rows": 3834,
    "gauc": 0.644,
    "ndcg5": 0.5166,
    "primary": 0.5803
   },
   "tab=4": {
    "rows": 7877,
    "gauc": 0.5817,
    "ndcg5": 0.5475,
    "primary": 0.5646
   },
   "tab=6": {
    "rows": 5170,
    "gauc": 0.6212,
    "ndcg5": 0.1055,
    "primary": 0.3633
   }
  }
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00249,
 "top3_valid": [
  {
   "n": 10,
   "primary": 0.6043737306391745
  },
  {
   "n": 17,
   "primary": 0.6042310624770972
  },
  {
   "n": 19,
   "primary": 0.6041401551174728
  }
 ],
 "designated": 7,
 "final_ranking": [
  {
   "n": 7,
   "valid_primary": 0.6039570725536132,
   "fresh_seeds": [
    0.6041188687289722,
    0.6042451985221791,
    0.6047650493502633
   ],
   "accepted": true,
   "mean": 0.6043763722004716,
   "std": 0.00034247956032530295,
   "tie_break": "within one SE (0.00029) of the best mean; accepted lineage preferred"
  },
  {
   "n": 10,
   "valid_primary": 0.6043737306391745,
   "fresh_seeds": [
    0.6044934573613927,
    0.604434679131614,
    0.6050167548948051
   ],
   "accepted": false,
   "mean": 0.6046482971292706,
   "std": 0.0003204443224820737
  },
  {
   "n": 17,
   "valid_primary": 0.6042310624770972,
   "fresh_seeds": [
    0.6044728917816915,
    0.6044414664461526,
    0.6046650262058673
   ],
   "accepted": false,
   "mean": 0.6045264614779038,
   "std": 0.00012102489745968404
  },
  {
   "n": 19,
   "valid_primary": 0.6041401551174728,
   "fresh_seeds": [
    0.6041495240890606,
    0.6045283268079515,
    0.6043954976258115
   ],
   "accepted": false,
   "mean": 0.6043577828409412,
   "std": 0.00019219697892563706
  }
 ],
 "usage": {
  "calls": 64,
  "tokens_in": 2408307,
  "tokens_out": 190414,
  "cache_read": 1558696,
  "cache_write": 0,
  "cost_usd": 10.739823000000001
 },
 "wall_clock_s": 2276.4,
 "champion_seed_mean": 0.60438,
 "best_single_seed": 0.6043737306391745,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": {
  "best_single_seed": 0.6039570725536132,
  "streak": 3,
  "converged_at_generation": 5,
  "champion_at_stop": 7
 },
 "official_rule_submission": {
  "node": 7,
  "generation": 5,
  "valid_primary": 0.6039570725536132,
  "fresh_seed_mean": 0.60438,
  "fresh_seeds": 3
 },
 "convergence_switch": "confirmed",
 "tokens": {
  "in_total": 2408307,
  "in_cached": 1558696,
  "in_uncached": 849611,
  "out": 190414
 },
 "interventions": 0,
 "k": 5,
 "k_later": 3,
 "eps": 0.002,
 "n_converge": 3,
 "iteration_unit": "node",
 "iterations_used": 21
}
```

## Iterations

### n=0 — node_000 (reproduce_baseline, parent None)
**Hypothesis:** Reproduce the official FM baseline under the harness contract.
**Method:** official FM · target `None` · expected Δ 0.0 (published valid primary 0.6016)
**Result:** GAUC 0.6671 · nDCG@5 0.5358 · primary 0.6015
**Diff:** `None` (None changed lines) · duration 15s · tokens in/out 0/0 · intervention: False

### n=1 — node_001 (explore, parent 0)
**Hypothesis:** Weight each training day to contribute equal total gradient mass, preventing the high-volume early logging regime from overwhelming patterns representative of validation and test.
**Method:** inverse-day-volume weighting · target `data-weighting` · expected Δ 0.0015 (Data Fact §5 shows roughly 230–280K rows on early days versus about 20K later, with 04-10/11 alone contributing 44% of train, so inverse logging-intensity weighting addresses covariate imbalance rather than merely favoring recent rows.)
**Result:** GAUC 0.6638 · nDCG@5 0.5339 · primary 0.5989 · realized Δ -0.0026 · rejected
**Diff:** `diffs/001.patch` (20 changed lines) · duration 23s · tokens in/out 64713/4191 · intervention: False

### n=2 — node_002 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with same-user positive-negative BPR so training directly optimizes the within-user ordering measured by GAUC.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0016 (The card was accepted five times on official FM, with confirmed fresh-seed gains of +0.0011 to +0.0017, and Data Fact §7 says most training users are discriminative.)
**Result:** GAUC 0.6692 · nDCG@5 0.5365 · primary 0.6028 · realized Δ +0.0014 · ACCEPTED · seed confirmation {'node_seed0': 0.6028286789333857, 'node_seeds': [0.6029525159912483, 0.601968666398895, 0.6033155626423989], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00129, 'se': 0.000361, 'z': 3.59, 'sigma_pooled': 0.000442, 'sigma_df': 4, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/002.patch` (36 changed lines) · duration 14s · tokens in/out 66174/7337 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Raise sparse-embedding L2 from the baseline value to 1e-5 so the model retains its epoch-7 ranking peak instead of fitting rare-ID noise.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.001 (The champion falls from 0.6015 at epoch 7 to 0.5990 at epoch 11, while this one-parameter change previously produced confirmed gains of +0.0009 and +0.0010 on official FM.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · rejected · seed confirmation {'node_seed0': 0.6025029206269998, 'node_seeds': [0.6028729444838677, 0.6017090117891591, 0.6020694898356386], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00077, 'se': 0.000389, 'z': 1.97, 'sigma_pooled': 0.000477, 'sigma_df': 6, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/003.patch` (2 changed lines) · duration 28s · tokens in/out 64599/3427 · intervention: False

### n=4 — node_004 (improve, parent 0)
**Hypothesis:** Add leakage-safe bucketized user preferences for the current author, tab, and duration bucket so sparse users can rank row-varying contexts using their earlier behavior.
**Method:** history-user-aggregates · target `history` · expected Δ 0.0008 (The method previously gained +0.0010 fresh-seed mean on official FM and another exact-stack probe moved +0.0006; Data Fact §2 reports a median of 35 training rows per user.)
**Result:** GAUC 0.6677 · nDCG@5 0.5361 · primary 0.6019 · realized Δ +0.0004 · rejected · seed confirmation {'node_seed0': 0.6018722623859434, 'node_seeds': [0.6021197760408004, 0.601621487783518, 0.6029344438819488], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00077, 'se': 0.000418, 'z': 1.85, 'sigma_pooled': 0.000512, 'sigma_df': 8, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/004.patch` (75 changed lines) · duration 32s · tokens in/out 67080/8841 · intervention: False

### n=5 — node_005 (retest, parent 0)
**Hypothesis:** Average normalized within-user ranks from five independently initialized FM models to suppress seed-specific ordering errors without mixing incomparable logit scales.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0008 (The prior official-FM seed ensemble was a positive but unconfirmed +0.0008 near-miss, and normalized ranks are justified by the metrics' within-user monotone invariance.)
**Result:** GAUC 0.6692 · nDCG@5 0.5367 · primary 0.6029 · realized Δ +0.0015 · ACCEPTED · seed confirmation {'node_seed0': 0.6029498705162017, 'node_seeds': [0.6029798477500652, 0.6024102758902958, 0.6028004012335035], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00128, 'se': 0.000398, 'z': 3.22, 'sigma_pooled': 0.000487, 'sigma_df': 10, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/005.patch` (68 changed lines) · duration 64s · tokens in/out 66126/8239 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_002; best 0.6028; tokens 451601/41959; 461s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 7 (0.6015), then falls steadily to 0.5990 by epoch 11; epoch 5 already reached 0.6010.
Champion metrics: GAUC 0.6671, nDCG@5 0.5358, mixed-user nDCG@5 0.7214; the raw nDCG is depressed by all-negative cohorts.
Last generation had no candidate nodes, so there are no GAUC-versus-nDCG deltas to attribute.
Most informative next probe: within-user BPR loss, targeting the loss component; Data Fact §7 says most training users are discriminative, and same-user pairs align directly with GAUC.
Validation-overfitting risk is currently low from search exposure: streak 0 and zero sub-0.002 wins accepted in this run, though the sharply peaked curve requires fresh-seed confirmation.
Group weakness: ignoring structurally near-zero tab=0/dur=0 nDCG, dur>180s is weakest actionable duration cohort (0.4982 primary; GAUC 0.6196, nDCG 0.3768).
Among substantial tabs, tab=1 has weak GAUC (0.6173) despite nDCG 0.5401; no node moved any group yet, so the deepen slot should target pairwise ordering, especially dur>180s/tab=1.
_Plan for next generation:_ `{"note": "BPR and five-seed rank averaging were the only accepted changes, target different components, and each delivered about +0.0013 fresh-seed mean.\nTheir combination is the highest-confidence next step; rejected L2/history branches lack sufficient evidence for another forced merge, so remaining slots should stay with the Selector.", "plan": [{"type": "merge", "merge_parents": [2, 5], "hypothesis": "Train five independently early-stopped same-user BPR FM models and average their tie-free normalized within-user ranks, combining BPR's metric-aligned ordering gain with seed averaging's variance reduction."}]}`

### n=6 — node_006 (explore, parent 2)
**Hypothesis:** Add a factorized third-order user×tab×duration interaction to the BPR FM so it can represent personal duration preferences that change by tab rather than only pairwise effects.
**Method:** targeted third-order user-tab-duration HOFM · target `model` · expected Δ 0.0008 (Higher-Order Factorization Machines (Blondel et al., 2016) model interactions absent from an FM, while Data Facts §3–4 show strong duration and tab structure and the champion remains weak on dur>180s.)
**Result:** GAUC 0.6692 · nDCG@5 0.5366 · primary 0.6029 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.602894788926603, 'node_seeds': [0.6024738516452064, 0.6010919535389108, 0.6026152249314444], 'champion_seeds': [0.6029525159912483, 0.601968666398895, 0.6033155626423989], 'delta_mean': -0.00069, 'se': 0.000444, 'z': -1.54, 'sigma_pooled': 0.000544, 'sigma_df': 12, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/006.patch` (27 changed lines) · duration 17s · tokens in/out 65347/5580 · intervention: False

### n=7 — node_007 (merge, parent 2, merge of [2, 5])
**Hypothesis:** Train five independently early-stopped same-user BPR FMs and average their tie-free normalized within-user ranks, retaining BPR's metric alignment while suppressing seed-specific ordering errors.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0009 (Nodes 002 and 005 each delivered about +0.0013 fresh-seed mean, while the prior exact-stack FM+BPR seed ensemble moved +0.0009, so overlap-adjusted expectation is +0.0009.)
**Result:** GAUC 0.6706 · nDCG@5 0.5373 · primary 0.6040 · realized Δ +0.0011 · ACCEPTED · seed confirmation {'node_seed0': 0.6039570725536132, 'node_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'champion_seeds': [0.6029525159912483, 0.601968666398895, 0.6033155626423989], 'delta_mean': 0.00163, 'se': 0.000429, 'z': 3.8, 'sigma_pooled': 0.000525, 'sigma_df': 14, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/007.patch` (72 changed lines) · duration 45s · tokens in/out 71400/4561 · intervention: False

### n=8 — node_008 (deepen, parent 2)
**Hypothesis:** Replace logistic BPR with a margin-0.5 hinge on the same within-user pairs so comfortably ordered pairs stop updating and capacity remains focused on weak orderings, including the dur>180s cohort where BPR helped most.
**Method:** loss-ranksvm-margin-pairs · target `loss` · expected Δ 0.0006 (The card range is 0.000–0.0015, and the diagnosis shows both a peaked BPR curve and a +0.0086 dur>180s group improvement, giving specific support for a conservative lower-third estimate.)
**Result:** GAUC 0.6699 · nDCG@5 0.5367 · primary 0.6033 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.6033038094249212, 'node_seeds': [0.6033466710004212, 0.6015471499800858, 0.6025937652245179], 'champion_seeds': [0.6029525159912483, 0.601968666398895, 0.6033155626423989], 'delta_mean': -0.00025, 'se': 0.000469, 'z': -0.53, 'sigma_pooled': 0.000575, 'sigma_df': 16, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/008.patch` (9 changed lines) · duration 14s · tokens in/out 66720/3764 · intervention: False

### n=9 — node_009 (deepen, parent 2)
**Hypothesis:** Raise L2 to 1e-5 only for latent embeddings while leaving linear field weights at 1e-6, reducing sparse-ID overfit without shrinking the useful tab and duration priors.
**Method:** regularization-embedding-only-l2 · target `regularization` · expected Δ 0.0005 (Global L2 was a positive FM+BPR near-miss of about +0.0003 fresh-seed mean, the champion mildly overfits, and node_003's critic identified the avoidable coupling between embedding and linear-weight penalties.)
**Result:** GAUC 0.6688 · nDCG@5 0.5364 · primary 0.6026 · realized Δ -0.0002 · rejected
**Diff:** `diffs/009.patch` (6 changed lines) · duration 14s · tokens in/out 64568/3292 · intervention: False

#### generation 2 closed — improved; streak 0; champion node_007; best 0.6044; tokens 436666/28961; 356s
_Diagnosis:_ Dynamics: mild overfit/plateau—primary first peaks at epoch 4 (0.6028), dips to 0.6019 at epoch 6, reties at epoch 7, then falls to 0.6025 at epoch 8.
Champion: GAUC 0.6692, nDCG@5 0.5365, mixed-user nDCG@5 0.7226.
Inverse-day weighting moved both halves down: GAUC −0.0033 and nDCG@5 −0.0019; it is dead here.
BPR moved mainly GAUC (+0.0021) versus nDCG@5 (+0.0007); L2 similarly gave +0.0014/+0.0007, while history aggregates barely moved either (+0.0006/+0.0003).
Five-seed averaging improved both halves, chiefly GAUC: +0.0021 GAUC and +0.0009 nDCG@5.
Next probe: merge five-seed rank averaging into the BPR champion, targeting ensembling/variance while retaining metric-aligned loss; Data Fact §7 supports same-user pair training because most training users are discriminative.
Validation-selection risk is currently moderate: streak 0, but two accepted wins were below +0.002 and therefore required seed confirmation.
Ignoring structurally weak dur=0/tab=0, dur>180s is weakest (0.5068 primary, 0.3784 nDCG); BPR moved it most (+0.0086), versus L2 +0.0061, history +0.0034, and seed averaging +0.0017, so the deepen slot should target robust long-duration ordering.
_Plan for next generation:_ `{"note": "Generation 2 improved decisively through the BPR plus five-seed rank ensemble; no second distinct component cleared the required +0.001 threshold, so no merge is justified.\nThe strongest parked retest is heterogeneous multiseed blending: the champion now supplies a strong five-member standard-BPR branch, while the complete field-aware/standard recipe previously produced the largest confirmed ensemble gain.", "plan": [{"type": "retest", "parent": "champion", "card": "ensembling-multiseed-heterogeneous-rank-blend", "hypothesis": "Retain the champion's five standard FM-BPR members, add five independently early-stopped field-aware FM-BPR members, and blend branch-averaged within-user ranks 0.6 field-aware to 0.4 standard to capture complementary ordering errors while preserving seed averaging.", "reason": "The stack has changed from a single BPR model to a confirmed five-seed BPR ensemble. Although one FM+BPR application was borderline at +0.0005 with z=1.76, the full heterogeneous multiseed recipe was previously confirmed at +0.0017, making this a higher-information retest than another loss or L2 variant."}]}`

### n=10 — node_010 (explore, parent 7)
**Hypothesis:** Retain the champion's global rank slots, but train long-only BPR specialists on same-user pairs where both rows exceed 180 seconds and use their seed-averaged order to permute only the long-video occupants of those slots, improving the weak cohort without disturbing cross-duration ordering.
**Method:** long-duration slot-preserving specialist · target `ensembling` · expected Δ 0.0006 (Data Fact §3 motivates distinct long-video behavior, while the current journal identifies dur>180s as the weakest actionable cohort at 0.5029 primary and reports that every generation-2 candidate worsened it.)
**Result:** GAUC 0.6712 · nDCG@5 0.5375 · primary 0.6044 · realized Δ +0.0004 · rejected · seed confirmation {'node_seed0': 0.6043737306391745, 'node_seeds': [0.6044934573613927, 0.604434679131614, 0.6050167548948051], 'champion_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'delta_mean': 0.00027, 'se': 0.000454, 'z': 0.6, 'sigma_pooled': 0.000556, 'sigma_df': 18, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/010.patch` (68 changed lines) · duration 64s · tokens in/out 71116/9890 · intervention: False

### n=11 — node_011 (deepen, parent 7)
**Hypothesis:** Replace 10% of each member's ordinary BPR samples with same-user positive-negative pairs where both videos exceed 180 seconds, directly improving ordering inside the weakest actionable duration cohort without disturbing most broad pairs.
**Method:** loss-bpr-pairwise-within-user — within-long-duration matched stream · target `loss` · expected Δ 0.0005 (The diagnosis identifies dur>180s as weakest at 0.5029 primary and says every generation-2 node worsened it; unlike the failed broad duration weighting, matched long-duration pairs cancel coarse duration effects and optimize within-cohort ordering.)
**Result:** GAUC 0.6705 · nDCG@5 0.5373 · primary 0.6039 · realized Δ -0.0000 · rejected
**Diff:** `diffs/011.patch` (16 changed lines) · duration 57s · tokens in/out 68106/5470 · intervention: False

### n=12 — node_012 (deepen, parent 7)
**Hypothesis:** Append one tab-crossed logarithmic duration field that is constant below 180 seconds but splits the >180-second tail into several bins, giving the ensemble finer row-varying structure only where its ranking is weakest.
**Method:** features-fine-duration-and-tab-cross — upper-tail-only refinement · target `features` · expected Δ 0.0005 (The generic duration refinement was an overall near-miss at -0.0001 on FM-BPR, while the current diagnosis specifically isolates dur>180s at only 0.3778 nDCG@5, supporting a narrower second iteration rather than another global rebucketing.)
**Result:** GAUC 0.6700 · nDCG@5 0.5367 · primary 0.6033 · realized Δ -0.0006 · rejected
**Diff:** `diffs/012.patch` (10 changed lines) · duration 52s · tokens in/out 65387/4637 · intervention: False

### n=13 — node_013 (deepen, parent 7)
**Hypothesis:** When an ensemble member has two validation stalls, halve its learning rate once and grant a fresh patience window so late low-step updates can improve its cached ranking instead of terminating at the first plateau.
**Method:** training-schedule-lr-decay-early-stop — per-member rescue decay · target `training-schedule` · expected Δ 0.0004 (Delayed decay previously reached a +0.0004 near-miss on FM-BPR, and the champion ensemble remains flat from epochs 7–12 without a late aggregate decline, making a cheap per-member rescue the most defensible schedule refinement.)
**Result:** GAUC 0.6707 · nDCG@5 0.5373 · primary 0.6040 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6039839304943033, 'node_seeds': [0.6039198710588058, 0.6039912098617719, 0.6047145798989604], 'champion_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'delta_mean': -0.00017, 'se': 0.000447, 'z': -0.38, 'sigma_pooled': 0.000548, 'sigma_df': 20, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/013.patch` (7 changed lines) · duration 65s · tokens in/out 65024/4003 · intervention: False

#### generation 3 closed — no improvement; streak 1; champion node_007; best 0.6044; tokens 456772/42130; 532s
_Diagnosis:_ Dynamics: primary rises through epoch 7 (0.6040) then stays flat at 0.6040 through epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6706/0.5373/0.7239, with no late decline.
Node 006 HOFM vs parent: GAUC ≈ unchanged, nDCG@5 +0.0001 and disc-nDCG +0.0002; neither half moved materially, so the cubic feature is dead here.
Node 007 seed ensemble: GAUC +0.0015, nDCG@5 +0.0008, disc-nDCG +0.0013; the accepted gain is chiefly broad pair ordering, with a smaller top-list benefit.
Node 008 RankSVM: GAUC +0.0007 but nDCG@5 only +0.0002; it mostly changed pair ordering rather than top-five quality, and fresh seeds reversed the gain.
Node 009 embedding-only L2: GAUC −0.0004 and nDCG@5 −0.0001; regularization moved neither half positively.
Next probe: add a five-seed field-aware BPR branch and blend it with the champion’s standard branch, targeting heterogeneous ensembling; Data Fact §1 supports well-estimated ID embeddings because the catalogue is closed.
Validation-selection risk is moderate: streak 0, but three accepted wins are below +0.002 and therefore depend on fresh-seed confirmation.
Ignoring dur=0/tab=0, dur>180s is weakest (primary 0.5029, nDCG 0.3778), and every node worsened it (006 −0.0009, 007 −0.0039, 008 −0.0011, 009 −0.0007); the deepen slot should target robust long-duration ordering.
_Plan for next generation:_ `{"note": "No generation-3 node improved the champion by >= 0.001, so no merge is justified; the long-duration specialist was the only positive branch but remained below confirmation thresholds.\nPrioritize the parked heterogeneous multiseed retest, then isolate the specialist to tabs where it helped and test one genuinely new legal sequence feature.", "plan": [{"type": "retest", "parent": "champion", "card": "ensembling-multiseed-heterogeneous-rank-blend", "hypothesis": "Retain the champion's five standard FM-BPR members, add five independently early-stopped field-aware FM-BPR members, average normalized within-user ranks within each branch, and blend them 0.6 field-aware to 0.4 standard.", "reason": "The context has changed from a single FM+BPR model, where this was only borderline, to a confirmed five-seed FM+BPR ensemble; the complete heterogeneous multiseed recipe previously delivered a confirmed +0.0017 gain and was planned but not executed this generation."}, {"type": "deepen", "parent": "champion", "card": "long-duration slot-preserving specialist", "hypothesis": "Use the long-duration specialist to permute >180-second slots only for tab 1 and tab 4 rows, where the prior specialist improved group primary by +0.0005 and +0.0012, while leaving tab 0, 2, and 6 ordering unchanged to avoid their measured losses."}, {"type": "explore", "parent": 7, "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and previous-same-author-by-tab categorical fields to every BPR ensemble member, targeting the measured within-tab fatigue signal from consecutive same-author exposures without using outcome history."}]}`

### n=14 — node_014 (explore, parent 7)
**Hypothesis:** Add a leakage-safe categorical field for each user's smoothed prior long-view affinity to the current video's tags, allowing semantic preference to generalize across the mostly unseen-to-user creators that raw user-author interactions cannot estimate.
**Method:** ordered user-tag semantic affinity · target `history` · expected Δ 0.0006 (Data Fact §10.3 says 96.6% of validation rows involve authors unseen by that user, while Data Fact §1 shows videos and their tags are in a closed catalogue, so tag-level history can share preference evidence where creator-level history is absent.)
**Result:** GAUC 0.6703 · nDCG@5 0.5369 · primary 0.6036 · realized Δ -0.0004 · rejected
**Diff:** `diffs/014.patch` (83 changed lines) · duration 65s · tokens in/out 69472/9741 · intervention: False

### n=15 — node_015 (retest, parent 7)
**Hypothesis:** Retain the champion's five standard FM-BPR members, add five independently early-stopped field-aware FM-BPR members, and blend branch-averaged within-user ranks 0.6/0.4 to capture complementary ordering errors.
**Method:** ensembling-multiseed-heterogeneous-rank-blend · target `ensembling` · expected Δ 0.001 (The complete heterogeneous multiseed recipe previously delivered a confirmed +0.0017 seed-mean gain, and the current stack now supplies a confirmed five-seed standard-BPR branch rather than the weaker single-model stack of the borderline retest.)
**Result:** GAUC 0.6700 · nDCG@5 0.5368 · primary 0.6034 · realized Δ -0.0006 · rejected
**Diff:** `diffs/015.patch` (43 changed lines) · duration 135s · tokens in/out 71475/7031 · intervention: False

### n=16 — node_016 (deepen, parent 7)
**Hypothesis:** Replace only 5% of ordinary pairs with same-user, same-tab long-duration pairs restricted to tabs 1 and 4, concentrating updates where the long-duration variants helped while avoiding tab 6 losses and easy cross-tab comparisons.
**Method:** loss-bpr-pairwise-within-user — same-tab long-duration stream · target `loss` · expected Δ 0.0004 (Node 011 improved dur>180s primary by +0.0027 despite being globally flat, while node 010 showed positive tab 1/tab 4 movements and a −0.0043 tab 6 movement, supporting a narrower second iteration.)
**Result:** GAUC 0.6703 · nDCG@5 0.5372 · primary 0.6038 · realized Δ -0.0002 · rejected
**Diff:** `diffs/016.patch` (19 changed lines) · duration 59s · tokens in/out 67426/5208 · intervention: False

### n=17 — node_017 (deepen, parent 7)
**Hypothesis:** Apply a milder 3e-6 L2 penalty only to user, video, and author embeddings while leaving tab, duration, and all linear weights at 1e-6, reducing sparse-ID variance without repeating the over-strong all-embedding penalty.
**Method:** regularization-embedding-dropout-l2 — mild ID-only shrinkage · target `regularization` · expected Δ 0.0003 (The BPR L2 probes were small near-misses around +0.0002 to +0.0003 fresh-seed mean, whereas node 009's 1e-5 all-embedding penalty lost only 0.0002, making a lower field-targeted strength the cheapest defensible refinement.)
**Result:** GAUC 0.6711 · nDCG@5 0.5374 · primary 0.6042 · realized Δ +0.0003 · rejected · seed confirmation {'node_seed0': 0.6042310624770972, 'node_seeds': [0.6044728917816915, 0.6044414664461526, 0.6046650262058673], 'champion_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'delta_mean': 0.00015, 'se': 0.00043, 'z': 0.35, 'sigma_pooled': 0.000527, 'sigma_df': 22, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/017.patch` (10 changed lines) · duration 56s · tokens in/out 66650/5105 · intervention: False

#### generation 4 closed — no improvement; streak 2; champion node_007; best 0.6044; tokens 482559/38953; 444s
_Diagnosis:_ Dynamics are flat: primary rises from 0.5945 at epoch 1 to 0.6040 at epoch 7, then remains 0.6040 through epoch 12; no late overfit.
Champion metrics are GAUC 0.6706, nDCG@5 0.5373, and mixed-user nDCG@5 0.7239.
Node 010 moved mainly GAUC (+0.0006), with only +0.0002 nDCG@5 and +0.0004 mixed-user nDCG; its gain is broad pair ordering, not strongly top-list.
Node 011 was flat (GAUC −0.0001, nDCG@5 ≈0); node 012 hurt both halves (about −0.0006 each), so its upper-tail feature is dead here.
Node 013 was also flat (GAUC +0.0001, nDCG@5 ≈0; fresh-seed mean −0.0002), giving no support for schedule rescue.
Next probe: heterogeneous multiseed field-aware/standard rank blending, targeting ensembling/model diversity; Data Fact §1 says the closed catalogue makes field-specific ID embeddings estimable.
Validation-overfitting risk is moderate: streak is 1/3, and three accepted wins were below +0.002 and required fresh-seed confirmation.
Ignoring structural dur=0/tab=0 weakness, dur>180s is weakest (0.5029 primary, 0.3778 nDCG): nodes 010/011 improved it +0.0036/+0.0027, while 012/013 moved −0.0008/−0.0003; deepen node 010 only on tabs 1 and 4 (+0.0005/+0.0012), avoiding tab 6 (−0.0043).
_Plan for next generation:_ `{"note": "No generation-4 node improved the champion by >= 0.001, so no merge is justified; heterogeneous blending and targeted long-duration loss changes were negative, while mild ID shrinkage was only a +0.00015 seed-mean no-win.\nWith the non-improving streak at 2, reserve one high-information slot for a genuinely new legal sequence signal and leave the other slots to the Selector.", "plan": [{"type": "explore", "parent": 7, "card": "history-same-author-run-features", "hypothesis": "Add leakage-safe run-so-far and previous-same-author\u00d7tab categorical fields to every BPR ensemble member, using only strictly earlier exposure features; this targets the measured within-tab same-author fatigue signal while introducing row-varying information absent from the static FM and the failed semantic-affinity history feature."}]}`

### n=18 — node_018 (explore, parent 7)
**Hypothesis:** Add a low-rank CP term sum(P[user] * Q[video] * R[tab]) to every BPR ensemble member so user-video preferences can change by surface rather than remaining shared across tabs.
**Method:** tab-conditioned user-video CP tensor · target `model` · expected Δ 0.0006 (Karatzoglou et al.'s Multiverse Recommendation motivates context-conditioned tensor factorization, while Data Facts §1 and §4 show a closed catalogue and extreme tab-dependent behavior that make a user×video×tab interaction estimable and relevant.)
**Result:** GAUC 0.6709 · nDCG@5 0.5371 · primary 0.6040 · realized Δ +0.0001 · rejected · seed confirmation {'node_seed0': 0.6040162597424729, 'node_seeds': [0.603347018301863, 0.6038222558304378, 0.6043758866036342], 'champion_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'delta_mean': -0.00053, 'se': 0.00043, 'z': -1.23, 'sigma_pooled': 0.000526, 'sigma_df': 24, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/018.patch` (27 changed lines) · duration 56s · tokens in/out 67757/4793 · intervention: False

### n=19 — node_019 (deepen, parent 7)
**Hypothesis:** Replace 2.5% of each ensemble member's ordinary BPR pairs with same-user tab-4 positive-negative pairs where both videos exceed 180 seconds, preserving node_016's strong tab-4 movement while avoiding its broader damage.
**Method:** loss-bpr-pairwise-within-user — tab-4-only long-duration matched stream · target `loss` · expected Δ 0.0004 (Node_016 was globally -0.0002 but improved tab=4 primary by +0.0047 and dur>180s by +0.0014, so restricting and halving that perturbation is the diagnosis-driven second iteration.)
**Result:** GAUC 0.6710 · nDCG@5 0.5373 · primary 0.6041 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.6041401551174728, 'node_seeds': [0.6041495240890606, 0.6045283268079515, 0.6043954976258115], 'champion_seeds': [0.6041188687289722, 0.6042451985221791, 0.6047650493502633], 'delta_mean': -2e-05, 'se': 0.000417, 'z': -0.04, 'sigma_pooled': 0.000511, 'sigma_df': 26, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/019.patch` (18 changed lines) · duration 43s · tokens in/out 67504/4489 · intervention: False

### n=20 — node_020 (deepen, parent 7)
**Hypothesis:** Allow node_010's long-duration specialists to permute only rank slots occupied by >180-second rows in tabs 1 or 4, leaving every other champion ordering unchanged.
**Method:** long-duration slot-preserving specialist — tab-1/tab-4 gate · target `ensembling` · expected Δ 0.0005 (Node_010 gained +0.00042 on seed 0 and +0.00027 over fresh seeds, while its group breakdown showed gains on tabs 1/4 and a -0.0043 loss on tab 6, making deterministic gating likely to remove its main harm.)
**Result:** ERROR at stage `implement`: no runnable script produced — recovery: None
**Diff:** `None` (None changed lines) · duration 0s · tokens in/out 215053/18841 · intervention: False

#### generation 5 closed — no improvement; streak 3; champion node_007; best 0.6044; tokens 580709/38411; 469s
_Diagnosis:_ Dynamics are flat: primary rises from 0.5945 at epoch 1 to 0.6040 at epoch 7, then remains 0.6040 through epoch 12.
Node 014 moved neither half positively: GAUC −0.0003, nDCG@5 −0.0004, mixed-user nDCG −0.0007; user-tag affinity is dead here.
Node 015 hurt both halves: GAUC −0.0006, nDCG@5 −0.0005, mixed-user nDCG −0.0008; heterogeneous blending is dead on this stack.
Node 016 mainly reduced GAUC (−0.0003), while nDCG@5 and mixed-user nDCG were essentially flat; its targeted loss did not improve top-list ordering.
Node 017 mainly improved GAUC (+0.0005), with only +0.0001 nDCG@5 and +0.0002 mixed-user nDCG; fresh-seed gain was just +0.00015.
Most informative next probe: leakage-safe same-author run-so-far and previous-same-author×tab fields, targeting history/sequence information supported by Data Fact §10.2’s within-tab fatigue effect.
Validation-selection risk is high-moderate: the non-improving streak is 2/3, and all three accepted wins were below +0.002 and required fresh-seed confirmation.
Ignoring structural dur=0/tab=0 weakness, dur>180s is weakest (primary 0.5029, nDCG 0.3778): nodes 014/016/017 moved it +0.0029/+0.0014/+0.0007, node 015 −0.0008; node 016 also moved tab=4 +0.0047, so deepen there.
_Plan for next generation:_ `{"note": "No node improved the champion by >= 0.001, so there is no eligible cross-component merge; the run has reached its three-generation convergence streak.\nIf another generation is permitted, recover the failed gated specialist once and test the still-untried legal same-author sequence signal; leave the remaining slot to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "long-duration slot-preserving specialist \u2014 tab-1/tab-4 gate", "hypothesis": "Recover node_020 and let node_010's long-duration specialists permute only slots occupied by >180-second impressions in tabs 1 or 4, preserving the champion ordering everywhere else.", "reason": "Node_020 failed before producing a runnable script, so the mechanism was never measured; the narrow gate was motivated by node_010's positive tab-1/tab-4 subgroup movements and avoids its tab-6 damage."}, {"type": "explore", "parent": 7, "card": "history-same-author-run-features", "hypothesis": "Add equal-time-safe run-so-far and previous-same-author\u00d7tab categorical fields to every BPR ensemble member, using only strictly earlier exposure features to capture the measured within-tab same-author fatigue signal."}]}`
