# Run journal — live_02

## Summary
```json
{
 "run_id": "live_02",
 "stop_reason": "converged: 3 generations without > 0.002 improvement",
 "generations": 5,
 "nodes": 16,
 "champion": 1,
 "champion_metrics": {
  "gauc": 0.6696755926496497,
  "ndcg5": 0.5364878968791964,
  "primary": 0.603081744764423,
  "ndcg5_disc": 0.7225608839404268
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00161,
 "top3_valid": [
  {
   "n": 5,
   "primary": 0.6036905218935378
  },
  {
   "n": 15,
   "primary": 0.6036749798374815
  },
  {
   "n": 6,
   "primary": 0.603591342679802
  }
 ],
 "designated": 15,
 "final_ranking": [
  {
   "n": 15,
   "valid_primary": 0.6036749798374815,
   "seeds": [
    0.6036749798374815,
    0.604173643311901,
    0.6039537705727057
   ],
   "mean": 0.6039341312406961,
   "std": 0.00020405161553125622
  },
  {
   "n": 6,
   "valid_primary": 0.603591342679802,
   "seeds": [
    0.603591342679802,
    0.6032550631951964,
    0.6029592540695468
   ],
   "mean": 0.6032685533148484,
   "std": 0.0002582253412554744
  },
  {
   "n": 5,
   "valid_primary": 0.6036905218935378,
   "seeds": [
    0.6036905218935378,
    0.6031319982343639,
    0.6027924981045094
   ],
   "mean": 0.6032050060774704,
   "std": 0.0003702335211561381
  }
 ],
 "usage": {
  "calls": 51,
  "tokens_in": 866254,
  "tokens_out": 99740,
  "cache_read": 682178,
  "cache_write": 0,
  "cost_usd": 4.253669
 },
 "wall_clock_s": 1780.5,
 "interventions": 0,
 "k": 3,
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

### n=1 — node_001 (explore, parent 0)
**Hypothesis:** Replace pointwise logloss with within-user BPR so training directly improves positive-negative ordering while cancelling metric-irrelevant user-constant effects.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0022 (The card projects 0.002–0.010, Fact 7 confirms many discriminative users, and project calibration maps a nominal 0.006 ranking-loss gain to about +0.0022.)
**Result:** GAUC 0.6697 · nDCG@5 0.5365 · primary 0.6031 · realized Δ +0.0016 · ACCEPTED · seed confirmation {'node_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00161, 'se': 0.0002, 't': 8.22, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/001.patch` (34 changed lines) · duration 15s · tokens in/out 34206/4668 · intervention: False

### n=2 — node_002 (explore, parent 0)
**Hypothesis:** Exponentially downweight early training rows so item and interaction estimates emphasize the late-April regime that most closely matches validation and hidden test.
**Method:** data-weighting-recency · target `data-weighting` · expected Δ 0.002 (The card range is 0.001–0.006, while Fact 5 shows severe volume and label drift plus a measured −0.007 FM valid-to-test gap, supporting a lower-third estimate near +0.002.)
**Result:** GAUC 0.6678 · nDCG@5 0.5360 · primary 0.6019 · realized Δ +0.0005 · rejected · seed confirmation {'node_seeds': [0.6019367741369501, 0.6020871274532423, 0.6013703133923629], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00036, 'se': 0.00024, 't': 1.5, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/002.patch` (13 changed lines) · duration 18s · tokens in/out 34051/3937 · intervention: False

### n=3 — node_003 (explore, parent 0)
**Hypothesis:** Assign duration-zero impressions a dedicated category so the FM can push these always-negative rows below genuinely short videos within each user.
**Method:** features-duration-unknown-flag · target `features` · expected Δ 0.0015 (The card range is 0.001–0.004 and Fact 3 provides deterministic mechanism evidence, but only about 1.9% of rows are affected, so a conservative lower-third estimate is appropriate.)
**Result:** GAUC 0.6671 · nDCG@5 0.5352 · primary 0.6012 · realized Δ -0.0003 · rejected
**Diff:** `diffs/003.patch` (4 changed lines) · duration 20s · tokens in/out 33927/3137 · intervention: False

#### generation 1 closed — no improvement; streak 1; champion node_001; best 0.6015; tokens 147487/13678; 248s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 7 (0.6015), then declines steadily to 0.5990 by epoch 11.
Champion: GAUC 0.6671, nDCG@5 0.5358; inferred mixed-user nDCG@5 ≈0.721 using Fact 7’s valid cohort shares.
Last generation had no child nodes, so there are no GAUC or nDCG@5 component deltas to diagnose.
Most informative next probe: within-user BPR pairwise loss, targeting ranking-loss alignment; Fact 7 shows 57.8% of valid users are discriminative.
This tests whether GAUC improves while nDCG@5 stays flat, which would indicate unresolved top-of-list ordering.
Validation-overfitting risk is currently low but will grow with repeated probing: non-improving streak 0 and zero sub-0.002 “wins” accepted.
_Plan for next generation:_ `{"note": "No merge is eligible: only BPR improved the prior champion by at least 0.001; recency and duration-unknown did not.\nRetest weak recency weighting on the changed BPR stack, and explore hard-negative sampling as the most direct refinement of the accepted ranking loss.", "plan": [{"type": "retest", "parent": "champion", "card": "data-weighting-recency", "hypothesis": "Combining seven-day recency weighting with within-user BPR will emphasize recent positive-negative preferences and improve transfer under drift more than recency-weighted pointwise training.", "reason": "The original pointwise result was weak but positive and uncertain (mean delta +0.00036, SE 0.00024); BPR is now the champion and changes which historical rows and pairs drive ranking gradients."}, {"type": "explore", "parent": "champion", "card": "loss-bpr-hard-negatives", "hypothesis": "Sampling three same-user negatives per positive will reduce wasted updates on already-correct pairs and improve GAUC and top-list ordering over single-negative BPR."}]}`

### n=4 — node_004 (retest, parent 1)
**Hypothesis:** Combining seven-day recency weighting with BPR should emphasize recent within-user preference pairs and transfer better under temporal drift than recency-weighted pointwise training.
**Method:** data-weighting-recency · target `data-weighting` · expected Δ 0.0008 (Node_002 was weak but positive at +0.0005 on pointwise FM, while Data Fact 5 documents strong traffic and label drift, so a changed BPR parent justifies a low-cost retest.)
**Result:** GAUC 0.6695 · nDCG@5 0.5367 · primary 0.6031 · realized Δ +0.0000 · rejected · seed confirmation {'node_seeds': [0.6031199975173567, 0.6030548263316666, 0.6029738269946883], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': -0.0, 'se': 0.00016, 't': -0.01, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/004.patch` (15 changed lines) · duration 12s · tokens in/out 34577/5216 · intervention: False

### n=5 — node_005 (explore, parent 1)
**Hypothesis:** Sampling three uniform same-user negatives per positive should spend more updates on informative ordering errors and improve both GAUC and top-list ranking over single-negative BPR.
**Method:** loss-bpr-hard-negatives · target `loss` · expected Δ 0.0013 (The card range is 0.000–0.004, and node_001's +0.0026 GAUC but only +0.0007 nDCG indicates room to refine pair sampling, though calibration places a first attempt in the range's lower third.)
**Result:** GAUC 0.6703 · nDCG@5 0.5371 · primary 0.6037 · realized Δ +0.0006 · rejected · seed confirmation {'node_seeds': [0.6036905218935378, 0.6031319982343639, 0.6027924981045094], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.00015, 'se': 0.00024, 't': 0.63, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/005.patch` (7 changed lines) · duration 20s · tokens in/out 69237/8140 · intervention: False

### n=6 — node_006 (improve, parent 1)
**Hypothesis:** Increasing embedding L2 to 1e-5 should reduce the clear post-epoch-8 overfitting of BPR while preserving its personalized ordering signal.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0012 (The card range is 0.000–0.004, and the champion declines steadily from 0.6031 at epoch 8 to 0.6020 at epoch 12, providing direct evidence that modest regularization may help.)
**Result:** GAUC 0.6697 · nDCG@5 0.5375 · primary 0.6036 · realized Δ +0.0005 · rejected · seed confirmation {'node_seeds': [0.603591342679802, 0.6032550631951964, 0.6029592540695468], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.00022, 'se': 0.00019, 't': 1.15, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/006.patch` (4 changed lines) · duration 16s · tokens in/out 69217/6933 · intervention: False

#### generation 2 closed — improved; streak 0; champion node_001; best 0.6037; tokens 221077/22486; 352s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 8 (0.6031) after 0.6030 at epoch 5, then declines steadily to 0.6020 by epoch 12.
node_001 BPR vs node_000: GAUC +0.0026, nDCG@5 +0.0007; gain is mostly pairwise/AUC, with limited top-list movement (ndcg5_disc 0.7226).
node_002 recency vs node_000: GAUC +0.0007, nDCG@5 +0.0002; both negligible (ndcg5_disc 0.7218), so fixed 7-day weighting is unconvincing.
node_003 duration-unknown vs node_000: GAUC +0.0000, nDCG@5 −0.0006 (ndcg5_disc 0.7203); this isolated feature is dead under pointwise FM.
Next probe: LambdaRank-weighted BPR targeting the loss/top-of-list component, because BPR moved GAUC far more than nDCG and Fact 7 shows evaluation lists are short but only 57.8% of valid users are discriminative.
Validation-overfitting risk is moderate: non-improving streak is 1, and one sub-0.002 win has been accepted after seed confirmation.
_Plan for next generation:_ `{"note": "No merge qualifies: none of nodes 4\u20136 improved the champion by >=0.001, and seed confirmation shows their gains were weak or null.\nUse the changed BPR stack to retest the mechanism-backed duration flag, and open one orthogonal history family; leave the third slot to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "features-duration-unknown-flag", "hypothesis": "A dedicated duration-zero category may become useful under within-user BPR because always-negative unknown-duration rows will receive direct pairwise pressure to rank below a user's positive impressions.", "reason": "The only prior test was on pointwise FM; the stack has changed to ranking-aligned BPR, and the feature encodes deterministic legal information."}, {"type": "explore", "parent": 1, "card": "history-user-aggregates", "hypothesis": "Leakage-safe per-user historical rates by author, tab, and duration bucket should add row-varying personalization beyond BPR's existing user-item interactions, especially for sparse users."}]}`

### n=7 — node_007 (retest, parent 1)
**Hypothesis:** A dedicated duration-zero category may help BPR directly push always-negative unknown-duration impressions below each user's positive rows.
**Method:** features-duration-unknown-flag · target `features` · expected Δ 0.001 (The card range is +0.001 to +0.004, but node_003 was flat on pointwise FM, so the changed ranking-aligned loss justifies only a lower-bound retest.)
**Result:** GAUC 0.6697 · nDCG@5 0.5365 · primary 0.6031 · realized Δ +0.0000 · rejected · seed confirmation {'node_seeds': [0.6030877383187754, 0.6032975650957224, 0.603313006772546], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.00018, 'se': 0.00016, 't': 1.11, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/007.patch` (4 changed lines) · duration 14s · tokens in/out 34444/3439 · intervention: False

### n=8 — node_008 (explore, parent 1)
**Hypothesis:** Leakage-safe per-user historical rates by author, tab, and duration bucket should add row-varying personalization beyond BPR's user-item interactions, particularly for sparse users.
**Method:** history-user-aggregates · target `history` · expected Δ 0.002 (The card range is +0.001 to +0.006, and Data Fact #2 shows a median of only 35 training interactions per valid user, making smoothed aggregates a plausible but modest source of new signal.)
**Result:** GAUC 0.6691 · nDCG@5 0.5368 · primary 0.6030 · realized Δ -0.0001 · rejected
**Diff:** `diffs/008.patch` (70 changed lines) · duration 22s · tokens in/out 35269/7203 · intervention: False

### n=9 — node_009 (improve, parent 1)
**Hypothesis:** Weighting BPR pairs by their current nDCG@5 swap impact should focus updates on top-ranked mistakes and convert the observed nDCG-specific movement into a larger primary gain.
**Method:** loss-lambdarank-pairs · target `loss` · expected Δ 0.002 (The card range is +0.001 to +0.006, while node_006 improved nDCG@5 by +0.0010 with no GAUC gain and Data Fact #7 reports short 5.6-row validation lists, providing specific evidence for top-of-list weighting.)
**Result:** GAUC 0.6685 · nDCG@5 0.5357 · primary 0.6021 · realized Δ -0.0010 · rejected
**Diff:** `diffs/009.patch` (36 changed lines) · duration 21s · tokens in/out 34927/7230 · intervention: False

#### generation 3 closed — no improvement; streak 1; champion node_001; best 0.6037; tokens 153483/20330; 284s
_Diagnosis:_ Dynamics: mild overfit; champion peaks at epoch 8 (0.6031) and declines steadily to 0.6020 by epoch 12.
Node 004 recency: GAUC −0.0002, nDCG@5 +0.0002 (disc 0.7229); effectively flat, so drift weighting adds no ranking signal.
Node 005 multi-negative BPR: GAUC +0.0006 and nDCG@5 +0.0006 (disc 0.7235); balanced but seed-unstable improvement.
Node 006 stronger L2: GAUC ≈0.0000, nDCG@5 +0.0010 (disc 0.7242); improvement is specifically top-of-list ordering.
Next probe: LambdaRank-weighted BPR targeting the loss/top-5 component, motivated by node 006’s nDCG-only movement and Data Fact #7’s short lists (5.6 valid impressions/user).
Validation-overfitting risk is elevated after several nearby probes and seed selection; streak is 0, but one sub-0.002 “win” has already been accepted.
_Plan for next generation:_ `{"note": "No merge qualifies: no node from this generation, or prior runner-up, improved the champion by the required >=0.001; duration/history/LambdaRank should remain parked.\nUse one runner-up-lineage slot to stabilize node_005's promising hard-negative gain and one untried auxiliary-target slot; leave the third slot to the Selector.", "plan": [{"type": "explore", "parent": 5, "card": "training-schedule-weight-averaging", "hypothesis": "EMA parameter averaging may stabilize node_005's +0.0006 hard-negative improvement across its peaked learning curve and turn the best runner-up lineage into a confirmable gain."}, {"type": "explore", "parent": "champion", "card": "aux-targets-is-click", "hypothesis": "A shared is_click auxiliary head should regularize sparse BPR embeddings with dense supervision from the same watch-time mechanism while retaining long_view BPR scores for ranking."}]}`

### n=10 — node_010 (explore, parent 1)
**Hypothesis:** EMA parameter averaging should stabilize node_005's promising hard-negative model around its validation peak and reduce late-epoch ordering degradation.
**Method:** training-schedule-weight-averaging · target `training-schedule` · expected Δ 0.001 (The card range is 0.000–0.003, and journal node_005 already gained +0.0006 while the champion curve shows a steady post-peak decline from 0.6031 to 0.6020.)
**Result:** GAUC 0.6659 · nDCG@5 0.5348 · primary 0.6003 · realized Δ -0.0027 · rejected
**Diff:** `diffs/010.patch` (29 changed lines) · duration 15s · tokens in/out 69713/11498 · intervention: False

### n=11 — node_011 (explore, parent 1)
**Hypothesis:** A shared is_click logistic head should provide dense near-miss supervision that regularizes sparse BPR embeddings while preserving the long_view BPR score used for ranking.
**Method:** aux-targets-is-click · target `aux-targets` · expected Δ 0.002 (The card range is 0.000–0.005, and Data Fact #6 gives unusually specific support: is_click has 0.76 correlation with long_view and occurs on 46% of training rows.)
**Result:** GAUC 0.6689 · nDCG@5 0.5367 · primary 0.6028 · realized Δ -0.0003 · rejected
**Diff:** `diffs/011.patch` (35 changed lines) · duration 17s · tokens in/out 34929/5472 · intervention: False

### n=12 — node_012 (explore, parent 1)
**Hypothesis:** A finer nonzero-duration encoding plus a tab-by-duration categorical cross should expose row-varying context interactions that the existing coarse FM fields may underrepresent under BPR.
**Method:** features-fine-duration-and-tab-cross · target `features` · expected Δ 0.001 (The card range is 0.000–0.004; Data Facts #3–4 show duration-bucket positive rates of 0.27–0.38 and tab rates from 0.04 to 0.49, while this interaction has not been measured on the champion.)
**Result:** GAUC 0.6692 · nDCG@5 0.5373 · primary 0.6033 · realized Δ +0.0002 · rejected · seed confirmation {'node_seeds': [0.6032636385461516, 0.6033115822562571, 0.6021449605887285], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': -0.00015, 'se': 0.00033, 't': -0.44, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/012.patch` (10 changed lines) · duration 14s · tokens in/out 34583/3920 · intervention: False

#### generation 4 closed — no improvement; streak 2; champion node_001; best 0.6037; tokens 188767/24181; 307s
_Diagnosis:_ Dynamics: mild overfit—primary peaks at epoch 8 (0.6031), then declines steadily to 0.6020 by epoch 12.
Node 007 (duration-unknown): GAUC and nDCG@5 are effectively unchanged (+0.0000/+0.0000; disc nDCG unchanged); this feature is dead under BPR.
Node 008 (history aggregates): GAUC falls ≈0.0006 while nDCG@5 rises ≈0.0003 (disc +0.0006); weak top-list movement offset by worse global ordering.
Node 009 (LambdaRank): both halves regress—GAUC −0.0012 and nDCG@5 −0.0008 (disc −0.0015); swap weighting did not improve top-of-list ordering.
Most informative next probe: auxiliary is_click head (aux-targets), motivated by Data Fact #6: correlation 0.76 with long_view and 46% prevalence, providing dense near-miss supervision.
Validation-overfitting risk is rising but still moderate: non-improving streak is 1, and one sub-0.002 win has already been accepted.
_Plan for next generation:_ `{"note": "No node delivered a reliable >=0.001 gain, and no two target components qualify for a merge; the fine duration cross reversed under seed confirmation.\nPrioritize new supervision and an alternative ranking objective, while retesting weight averaging because node_010\u2019s parent/stack description and severely depressed early curve suggest an implementation mismatch.", "plan": [{"type": "explore", "parent": "champion", "card": "loss-watchtime-censored", "hypothesis": "A censored watch-time auxiliary head can supply graded near-threshold supervision absent from binary BPR labels and improve within-user ordering, especially among short-video near-misses."}, {"type": "explore", "parent": "champion", "card": "loss-listwise-softmax-within-user", "hypothesis": "Replacing sampled BPR pairs with a per-user ListNet objective may improve whole-list and top-five ordering while remaining invariant to metric-irrelevant user-constant scores."}, {"type": "retest", "parent": "champion", "card": "training-schedule-weight-averaging", "hypothesis": "Correctly averaging only champion checkpoints around epochs 5\u20138 may reduce peak-epoch variance without the severe lag observed in node_010.", "reason": "Node_010 declared parent node_001 but claimed to preserve node_005\u2019s hard-negative stack, and its curve began near 0.558 versus the champion\u2019s 0.594; retest with isolated checkpoint averaging rather than the apparently lagging EMA implementation."}]}`

### n=13 — node_013 (explore, parent 1)
**Hypothesis:** Adding a censored watch-time auxiliary loss to the champion's shared embeddings will provide graded near-threshold supervision that binary BPR lacks and improve within-user ordering.
**Method:** loss-watchtime-censored · target `loss` · expected Δ 0.002 (The card range is +0.001 to +0.008, and Data Fact #3 specifically supports the mechanism because long_view is duration-defined and 17.2% of plays are censored completions.)
**Result:** GAUC 0.6698 · nDCG@5 0.5372 · primary 0.6035 · realized Δ +0.0004 · rejected · seed confirmation {'node_seeds': [0.6035028757484949, 0.6032081140601903, 0.6033533405085663], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.0003, 'se': 0.00016, 't': 1.85, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/013.patch` (50 changed lines) · duration 23s · tokens in/out 35136/6318 · intervention: False

### n=14 — node_014 (retest, parent 1)
**Hypothesis:** Averaging only the champion's epoch-5 through epoch-8 checkpoints should reduce peak-checkpoint variance without the severe early-training lag caused by node_010's apparent EMA or stack mismatch.
**Method:** training-schedule-weight-averaging · target `training-schedule` · expected Δ 0.0007 (The card range is +0.000 to +0.003, while the champion's measured peak-and-decline curve and node_010's anomalously low initial score justify a clean implementation retest.)
**Result:** GAUC 0.6695 · nDCG@5 0.5371 · primary 0.6033 · realized Δ +0.0002 · rejected · seed confirmation {'node_seeds': [0.6032869310374758, 0.6034374160160765, 0.6028863281868211], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.00015, 'se': 0.00018, 't': 0.86, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/014.patch` (13 changed lines) · duration 15s · tokens in/out 34669/4839 · intervention: False

### n=15 — node_015 (explore, parent 1)
**Hypothesis:** A five-seed average of the BPR champion should cancel seed-specific pair-ordering errors and stabilize validation performance as the search approaches convergence.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0012 (The card predicts +0.001 to +0.003, and the measured seed standard deviation is 0.0008, making seed averaging a low-risk closing move after two non-improving generations.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6037 · realized Δ +0.0006 · rejected · seed confirmation {'node_seeds': [0.6036749798374815, 0.604173643311901, 0.6039537705727057], 'champion_seeds': [0.603081744764423, 0.6032265276409501, 0.6028476475917082], 'delta_mean': 0.00088, 'se': 0.00016, 't': 5.35, 'rule': 'diff >= 0.001 and t >= 2.5'}
**Diff:** `diffs/015.patch` (49 changed lines) · duration 42s · tokens in/out 34655/5226 · intervention: False

#### generation 5 closed — no improvement; streak 3; champion node_001; best 0.6037; tokens 155440/19065; 436s
_Diagnosis:_ Dynamics: clear overfit—champion peaks at epoch 8 (0.6031), then declines steadily to 0.6020 by epoch 12.
Node 010 (EMA hard negatives): GAUC −0.0038, nDCG@5 −0.0017; both halves regressed, especially pairwise ordering (disc nDCG 0.7197).
Node 011 (is_click auxiliary): GAUC −0.0008, nDCG@5 +0.0002; near-miss supervision slightly shifts the top but harms overall ordering (disc nDCG 0.7230).
Node 012 (fine duration × tab): GAUC −0.0005, nDCG@5 +0.0008; improvement is top-of-list only, but three-seed mean delta is −0.00015 (disc nDCG 0.7240), so the feature is effectively flat.
Most informative next probe: censored watch-time auxiliary loss, targeting the watch-time/loss component; Data Fact #3 says long_view is duration-defined and 17.2% of plays are censored completions.
Validation-overfitting risk is rising: two consecutive non-improving generations and 13 nodes searched; one sub-0.002 “win” has been accepted.
_Plan for next generation:_ `{"note": "No merge is eligible: no two nodes with different target components improved the champion by at least 0.001.\nThe generation did not improve, so reserve one exploration slot for the untried architecture family and leave two slots to the Selector.", "plan": [{"type": "explore", "parent": 1, "card": "model-dcn-cross-head", "hypothesis": "Adding one regularized DCN-V2 cross layer to the BPR champion should capture higher-order user-by-tab-by-duration interactions that the FM cannot express directly, after feature, auxiliary-loss, and scheduling changes proved mostly flat."}]}`
