# Run journal — live_07

## Summary
```json
{
 "run_id": "live_07",
 "stop_reason": "converged: 3 generations without a >= 0.001 cumulative rise of the champion fresh-seed mean (ADR-0012)",
 "generations": 5,
 "nodes": 25,
 "champion": 9,
 "champion_metrics": {
  "gauc": 0.6709429808508166,
  "ndcg5": 0.5372849293965677,
  "primary": 0.6041139551236921,
  "ndcg5_disc": 0.723940356184314,
  "by_group": {
   "dur18-60s": {
    "rows": 33235,
    "gauc": 0.6466,
    "ndcg5": 0.4603,
    "primary": 0.5535
   },
   "dur60-180s": {
    "rows": 46565,
    "gauc": 0.6499,
    "ndcg5": 0.4879,
    "primary": 0.5689
   },
   "dur<18s": {
    "rows": 20101,
    "gauc": 0.7051,
    "ndcg5": 0.4057,
    "primary": 0.5554
   },
   "dur=0": {
    "rows": 2107,
    "gauc": 0.564,
    "ndcg5": 0.0861,
    "primary": 0.325
   },
   "dur>180s": {
    "rows": 22901,
    "gauc": 0.6306,
    "ndcg5": 0.3781,
    "primary": 0.5044
   },
   "tab=0": {
    "rows": 13726,
    "gauc": 0.5506,
    "ndcg5": 0.0629,
    "primary": 0.3068
   },
   "tab=1": {
    "rows": 92672,
    "gauc": 0.621,
    "ndcg5": 0.541,
    "primary": 0.581
   },
   "tab=2": {
    "rows": 3834,
    "gauc": 0.6402,
    "ndcg5": 0.5142,
    "primary": 0.5772
   },
   "tab=4": {
    "rows": 7877,
    "gauc": 0.5781,
    "ndcg5": 0.5478,
    "primary": 0.563
   },
   "tab=6": {
    "rows": 5170,
    "gauc": 0.6358,
    "ndcg5": 0.1057,
    "primary": 0.3708
   }
  }
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00265,
 "top3_valid": [
  {
   "n": 19,
   "primary": 0.6046074296225528
  },
  {
   "n": 13,
   "primary": 0.6045977116419363
  },
  {
   "n": 11,
   "primary": 0.6043075347501692
  }
 ],
 "designated": 19,
 "final_ranking": [
  {
   "n": 19,
   "valid_primary": 0.6046074296225528,
   "fresh_seeds": [
    0.60459485340838,
    0.604861559954103,
    0.60512336709234
   ],
   "accepted": false,
   "mean": 0.6048599268182744,
   "std": 0.00026426062681133715
  },
  {
   "n": 13,
   "valid_primary": 0.6045977116419363,
   "fresh_seeds": [
    0.6045661655807829,
    0.6046915899721115,
    0.6045834125887297
   ],
   "accepted": false,
   "mean": 0.6046137227138747,
   "std": 6.798416911137263e-05
  },
  {
   "n": 11,
   "valid_primary": 0.6043075347501692,
   "fresh_seeds": [
    0.6046567884077674,
    0.6046329639642818,
    0.6044638186702289
   ],
   "accepted": false,
   "mean": 0.6045845236807593,
   "std": 0.00010521015018626161
  },
  {
   "n": 9,
   "valid_primary": 0.6041139551236921,
   "fresh_seeds": [
    0.6040754830744365,
    0.6045684439544415,
    0.6044516995706408
   ],
   "accepted": true,
   "mean": 0.6043652088665062,
   "std": 0.00025761034282440273
  }
 ],
 "usage": {
  "calls": 72,
  "tokens_in": 3708647,
  "tokens_out": 295935,
  "cache_read": 2133434,
  "cache_write": 0,
  "cost_usd": 17.820832000000003
 },
 "wall_clock_s": 3573.9,
 "champion_seed_mean": 0.60437,
 "best_single_seed": 0.6046074296225528,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": {
  "best_single_seed": 0.6041139551236921,
  "streak": 3,
  "converged_at_generation": 5,
  "champion_at_stop": 9
 },
 "official_rule_submission": {
  "node": 9,
  "generation": 5,
  "valid_primary": 0.6041139551236921,
  "fresh_seed_mean": 0.60437,
  "fresh_seeds": 3
 },
 "convergence_switch": "confirmed",
 "tokens": {
  "in_total": 3708647,
  "in_cached": 2133434,
  "in_uncached": 1575213,
  "out": 295935
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

### n=1 — node_001 (explore, parent 0)
**Hypothesis:** Add candidate-match features against each user's most recent strictly earlier long-viewed video's tag, music, and video type, plus recency, to capture label-conditioned continuation while generalizing beyond exact authors.
**Method:** last-positive-attribute-recurrence · target `history` · expected Δ 0.0012 (Data Fact §10.1 reports a 0.78 long-view rate after a previously long-viewed same-author exposure, while §10.3 shows exact creator history is usually unavailable, motivating broader side-attribute matching.)
**Result:** GAUC 0.6682 · nDCG@5 0.5358 · primary 0.6020 · realized Δ +0.0005 · ACCEPTED · seed confirmation {'node_seed0': 0.601969254784846, 'node_seeds': [0.6028687869783466, 0.602174699972952, 0.6020127863639722], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.0009, 'se': 0.000289, 'z': 3.12, 'sigma_pooled': 0.000354, 'sigma_df': 4, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/001.patch` (86 changed lines) · duration 34s · tokens in/out 86231/9196 · intervention: False

### n=2 — node_002 (explore, parent 0)
**Hypothesis:** Add strictly prior session position, recent-impression density, and previous-gap fields so the FM can rank fresh-session impressions above rows encountered during attention-fatiguing scrolls.
**Method:** features-exposure-session · target `features` · expected Δ 0.001 (Data Fact §10.5 reports tab-1 long-view rates falling from 0.418 at session start to 0.137 beyond position 30, supporting a lower-third estimate from the card's [0.0005, 0.003] range.)
**Result:** GAUC 0.6669 · nDCG@5 0.5359 · primary 0.6014 · realized Δ -0.0001 · rejected
**Diff:** `diffs/002.patch` (65 changed lines) · duration 26s · tokens in/out 86936/8863 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with same-user positive-negative BPR so training directly optimizes the within-user ordering measured by GAUC and useful to nDCG@5.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0015 (The card was accepted six times on the official-FM stack, with fresh-seed mean gains generally between +0.0011 and +0.0017.)
**Result:** GAUC 0.6697 · nDCG@5 0.5365 · primary 0.6031 · realized Δ +0.0016 · ACCEPTED · seed confirmation {'node_seed0': 0.603081744764423, 'node_seeds': [0.6032265276409501, 0.6028476475917082, 0.6029940059970083], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00157, 'se': 0.000268, 'z': 5.87, 'sigma_pooled': 0.000328, 'sigma_df': 6, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/003.patch` (44 changed lines) · duration 19s · tokens in/out 85972/6297 · intervention: False

### n=4 — node_004 (improve, parent 0)
**Hypothesis:** Raise uniform FM L2 from 1e-6 to 1e-5 to reduce sparse-ID overfitting responsible for the steady validation decline after epoch 7.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0009 (The champion falls from 0.6015 at epoch 7 to 0.5990 at epoch 11, and this two-line L2 change previously delivered confirmed +0.0009 and +0.0010 gains on official FM.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6025029206269998, 'node_seeds': [0.6028729444838677, 0.6017090117891591, 0.6020694898356386, 0.6027857969882158, 0.6026234157272281], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00096, 'se': 0.000282, 'z': 3.41, 'sigma_pooled': 0.000386, 'sigma_df': 10, 'sigma_from_node_only': False, 'adaptive': True, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/004.patch` (2 changed lines) · duration 30s · tokens in/out 84392/2870 · intervention: False

### n=5 — node_005 (improve, parent 0)
**Hypothesis:** Average normalized within-user ranks from five independently seeded copies of the champion to cancel initialization noise and recurring small ordering errors.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0011 (Five-seed averaging on official FM previously measured gains of +0.0008 and a confirmed +0.0013, making +0.0011 a calibrated expectation.)
**Result:** GAUC 0.6692 · nDCG@5 0.5367 · primary 0.6029 · realized Δ +0.0015 · ACCEPTED · seed confirmation {'node_seed0': 0.6029498705162017, 'node_seeds': [0.6029798477500652, 0.6024102758902958, 0.6028004012335035], 'champion_seeds': [0.6017609746263709, 0.6010899806390073, 0.6015033559120361], 'delta_mean': 0.00128, 'se': 0.000306, 'z': 4.17, 'sigma_pooled': 0.000375, 'sigma_df': 12, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/005.patch` (60 changed lines) · duration 66s · tokens in/out 85836/7780 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_003; best 0.6030; tokens 592777/42758; 456s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 7 (0.6015), then falls steadily to 0.5990 by epoch 11; best metrics are GAUC 0.6671, nDCG@5 0.5358, mixed-user nDCG@5 0.7214.
Last generation had no candidate nodes, so there are no GAUC or nDCG@5 deltas to attribute.
Most informative next probe: features-exposure-session, targeting row-varying context; Data Fact §10.5 shows tab-1 long-view rate dropping from 0.418 at session start to 0.137 beyond position 30.
Validation-overfitting risk is currently low but nonzero: streak 0 and zero sub-0.002 wins accepted in this run; avoid selecting marginal single-seed gains against the same split.
Weakest meaningful duration cohort is >180s (primary 0.4982; GAUC 0.6196; nDCG 0.3768); tab 4 has especially weak GAUC (0.5636), while tab 0/dur=0 nDCG is structurally near zero.
No node moved any group this generation and no group is HARD; deepen session features first, then inspect whether they improve >180s and tab-4 ordering.
_Plan for next generation:_ `{"note": "BPR is the new champion, while L2 regularization and seed averaging each delivered confirmed gains above 0.001 in distinct components, so both deserve isolated merges with BPR.\nThe session feature was flat only under pointwise FM; retesting it on metric-aligned BPR is justified by the changed stack and its strong tab=4 and duration-tail subgroup gains.", "plan": [{"type": "merge", "merge_parents": [3, 4], "hypothesis": "Combine same-user BPR with uniform L2=1e-5; BPR improves objective alignment while stronger shrinkage may reduce the sharp post-peak overfitting still visible in the BPR learning curve."}, {"type": "merge", "merge_parents": [3, 5], "hypothesis": "Train five independently early-stopped BPR models and average their normalized within-user ranks, combining the confirmed BPR ordering gain with the confirmed variance reduction from seed ensembling."}, {"type": "retest", "parent": "champion", "card": "features-exposure-session", "hypothesis": "Add equal-time-safe session position, prior-10-minute density, and previous-gap fields to the BPR champion; pairwise training may use these row-varying attention signals more effectively than pointwise FM, especially in tab 4 and long-duration cohorts.", "reason": "The only test used pointwise FM and was nearly flat overall, but showed +0.0104 on tab 4 and +0.0026 for duration >180s; the loss stack has now changed materially to BPR."}]}`

### n=6 — node_006 (explore, parent 3)
**Hypothesis:** Add a leakage-safe Rocchio-style score contrasting each candidate's tag, music, and video-type matches against the user's strictly earlier positive versus negative content profile, allowing BPR to transfer preferences to unseen-author candidates.
**Method:** rocchio-contrastive-content-history · target `history` · expected Δ 0.0008 (Rocchio relevance feedback uses positive-minus-negative content profiles, while Data Facts §2 and §10.3 show median histories of 35 rows but 96.6% unseen candidate authors, and journal node_001's simpler latest-positive attribute match gained +0.0009 fresh-seed mean.)
**Result:** ERROR at stage `implement`: vetoed by critic: `rocchio_edges` is computed from all ordered-history scores. Later training scores incorporate earlier rows' labels, so a row's own `long_view` can influence the global bin thresholds subsequently applied to that row; the scored-row history feature is therefore not strictly earlier-time-safe. — recovery: None
**Diff:** `None` (None changed lines) · duration 0s · tokens in/out 89095/9378 · intervention: False

### n=7 — node_007 (explore, parent 3)
**Hypothesis:** Train a LightGBM LambdaRank model with query=user over legal continuous context and leakage-safe historical statistics so non-linear tab-duration-session interactions and top-list pair weights can improve on the plateauing second-order FM.
**Method:** model-lightgbm-lambdarank · target `model` · expected Δ 0.0015 (The card range is +0.001 to +0.006, and Data Fact §10.5 supplies unusually strong row-varying session effects while BPR moved nDCG@5 only +0.0007.)
**Result:** GAUC 0.6661 · nDCG@5 0.5356 · primary 0.6008 · realized Δ -0.0022 · rejected
**Diff:** `diffs/007.patch` (144 changed lines) · duration 189s · tokens in/out 90138/26134 · intervention: False

### n=8 — node_008 (merge, parent 3, merge of [3, 4])
**Hypothesis:** Combine same-user BPR with uniform L2=1e-5 so metric-aligned pair training retains its ordering gain while stronger shrinkage limits the champion's post-epoch-8 overfitting.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0005 (Node_004 gained +0.0010 fresh-seed mean on pointwise FM and the BPR champion falls from 0.6031 to 0.6020 after its peak, although prior BPR-side L2 gains warrant calibration down to +0.0005.)
**Result:** GAUC 0.6704 · nDCG@5 0.5369 · primary 0.6036 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.603605953442093, 'node_seeds': [0.6032591140834885, 0.602928993441006, 0.6031787507570261], 'champion_seeds': [0.6032265276409501, 0.6028476475917082, 0.6029940059970083], 'delta_mean': 0.0001, 'se': 0.000293, 'z': 0.34, 'sigma_pooled': 0.000358, 'sigma_df': 14, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/008.patch` (2 changed lines) · duration 16s · tokens in/out 87757/3232 · intervention: False

### n=9 — node_009 (merge, parent 3, merge of [3, 5])
**Hypothesis:** Train five independently early-stopped BPR models and average their normalized within-user ranks to combine BPR's objective alignment with the strongest observed reduction in top-list and seed variance.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0014 (Node_005 gained +0.00128 fresh-seed mean and the card previously confirmed about +0.0016 on an FM+BPR stack, including the generation's strongest nDCG@5 movement.)
**Result:** GAUC 0.6709 · nDCG@5 0.5373 · primary 0.6041 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seed0': 0.6041139551236921, 'node_seeds': [0.6040754830744365, 0.6045684439544415, 0.6044516995706408], 'champion_seeds': [0.6032265276409501, 0.6028476475917082, 0.6029940059970083], 'delta_mean': 0.00134, 'se': 0.000286, 'z': 4.7, 'sigma_pooled': 0.00035, 'sigma_df': 16, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/009.patch` (63 changed lines) · duration 51s · tokens in/out 91062/5116 · intervention: False

### n=10 — node_010 (retest, parent 3)
**Hypothesis:** Add equal-time-safe session position, prior-10-minute density, and previous-gap fields to the BPR champion so pairwise training can exploit attention-fatigue context that pointwise FM failed to use globally.
**Method:** features-exposure-session · target `features` · expected Δ 0.0008 (Node_002 was flat overall but improved tab=4 by +0.0104 and duration>180s by +0.0026, while Data Fact §10.5 shows tab-1 long-view rates falling from 0.418 at session start to 0.137 after position 30.)
**Result:** GAUC 0.6697 · nDCG@5 0.5371 · primary 0.6034 · realized Δ +0.0003 · ACCEPTED · seed confirmation {'node_seed0': 0.6033678044199775, 'node_seeds': [0.6041550778392764, 0.6038114764408065, 0.6037173893420826], 'champion_seeds': [0.6032265276409501, 0.6028476475917082, 0.6029940059970083], 'delta_mean': 0.00087, 'se': 0.000278, 'z': 3.14, 'sigma_pooled': 0.000341, 'sigma_df': 18, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/010.patch` (65 changed lines) · duration 21s · tokens in/out 89702/4662 · intervention: False

#### generation 2 closed — improved; streak 0; champion node_009; best 0.6044; tokens 662227/63511; 818s
_Diagnosis:_ Dynamics: overfit—primary peaks at epoch 8 at 0.6031, then declines steadily to 0.6020 by epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6697/0.5365/0.7226.
Node 1 history moved GAUC +0.0011 but nDCG@5 was flat (+0.0000), indicating broad pair ordering rather than top-list improvement.
Node 2 session features moved neither half: GAUC −0.0002, nDCG@5 +0.0001; dead under pointwise FM.
Node 3 BPR primarily moved GAUC +0.0026, with nDCG@5 +0.0007; Node 4 L2 moved +0.0014/+0.0007.
Node 5 seed averaging moved GAUC +0.0021 and nDCG@5 +0.0009, the strongest top-list movement of the generation.
Most informative next probe: retest exposure-session features on BPR, targeting row-varying context; Data Fact §10.5 shows attention decay from 0.418 at session start to 0.137 after position 30.
Validation-overfitting risk is material despite streak 0: four accepted wins are below 0.002, so further marginal validation gains need strict fresh-seed confirmation.
Ignoring structurally low tab=0/dur=0, dur>180s is weakest (primary 0.5043) and tab=4 has weakest meaningful GAUC (0.5737); session features moved tab=4 +0.0104 and dur>180s +0.0026, while BPR/L2 moved them +0.0057/+0.0061 and +0.0060/+0.0061; no groups are HARD.
_Plan for next generation:_ `{"note": "Five-seed BPR rank averaging produced the generation\u2019s only >=0.001 improvement and is now the champion; no eligible second component exists for a merge.\nSession features were seed-confirmed and strongest on tab 4 and 18\u201360s rows, while the Rocchio branch failed only because of a repairable leakage bug.", "plan": [{"type": "retest", "parent": "champion", "card": "features-exposure-session", "hypothesis": "Add the equal-time-safe session-position, prior-10-minute-density, and previous-gap fields to every member of the five-seed BPR champion; rank averaging may stabilize the session branch\u2019s confirmed but noisy +0.00087 seed-mean gain.", "reason": "The prior retest used a single BPR model, whereas the stack now contains five independently early-stopped BPR members and rank averaging, which materially changes variance and may preserve the session signal."}, {"type": "deepen", "parent": 10, "card": "features-exposure-session", "mechanism": "tab-duration-conditioned-session-fatigue", "target_group": "tab=4 and dur18-60s", "hypothesis": "Retain the three session fields and append compact categorical crosses of session-position/density with tab and broad duration cohort, focusing capacity on the groups where node_010 gained +0.0037 and +0.0035 rather than forcing one shared fatigue representation across all contexts."}, {"type": "retest", "parent": "champion", "card": "rocchio-contrastive-content-history", "hypothesis": "Add a leakage-safe positive-versus-negative content-profile field using fixed, label-independent score bins defined before outcome accumulation, with equal-time rows committed together.", "reason": "The branch never ran: the critic found that globally learned quantile edges could indirectly include a row\u2019s own label. Fixed semantic bins or bins derived only from feature/count structure remove that bug, and the five-seed BPR champion provides a more stable changed stack."}]}`

### n=11 — node_011 (explore, parent 9)
**Hypothesis:** Append equal-time-safe buckets for each candidate's tag, music, and video-type overlap with the user's five most recent same-session exposures, allowing every BPR ensemble member to learn short-term semantic repetition or novelty beyond static IDs and tab.
**Method:** session-attribute-novelty-fatigue · target `history` · expected Δ 0.0008 (STAMP-style short-term session memory is supported by Data Fact §10.2's within-tab penalty for consecutive same-author exposure and node_001's +0.0009 fresh-seed gain from outcome-conditioned attribute recurrence, making +0.0008 a conservative estimate for the label-free variant.)
**Result:** GAUC 0.6710 · nDCG@5 0.5376 · primary 0.6043 · realized Δ +0.0002 · rejected · seed confirmation {'node_seed0': 0.6043075347501692, 'node_seeds': [0.6046567884077674, 0.6046329639642818, 0.6044638186702289], 'champion_seeds': [0.6040754830744365, 0.6045684439544415, 0.6044516995706408], 'delta_mean': 0.00022, 'se': 0.000267, 'z': 0.82, 'sigma_pooled': 0.000327, 'sigma_df': 20, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/011.patch` (83 changed lines) · duration 123s · tokens in/out 90007/9700 · intervention: False

### n=12 — node_012 (explore, parent 9)
**Hypothesis:** Add an equal-time-safe FPMC transition term between each user's immediately previous exposed video and the current video to every member of the five-seed BPR ensemble, capturing continuation and fatigue absent from static FM fields.
**Method:** model-first-order-exposure-transition-fm · target `model` · expected Δ 0.0003 (The untried card range is [0.000, 0.0010], and Data Fact §10.2 shows strong exposure-sequence effects, but sparse video transitions place a first attempt in the range's lower third.)
**Result:** GAUC 0.6700 · nDCG@5 0.5372 · primary 0.6036 · realized Δ -0.0005 · rejected
**Diff:** `diffs/012.patch` (89 changed lines) · duration 104s · tokens in/out 91622/9651 · intervention: False

### n=13 — node_013 (retest, parent 9)
**Hypothesis:** Add equal-time-safe session position, prior-10-minute density, and previous-gap fields to every member of the five-seed BPR champion so rank averaging can stabilize the top-list-specific session signal.
**Method:** features-exposure-session · target `features` · expected Δ 0.0008 (Node_010 produced fresh-seed mean gain +0.00087 and moved nDCG@5 +0.0006, while Data Fact §10.5 reports tab-1 long-view rate declining from 0.418 at session start to 0.137 beyond position 30.)
**Result:** GAUC 0.6711 · nDCG@5 0.5381 · primary 0.6046 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.6045977116419363, 'node_seeds': [0.6045661655807829, 0.6046915899721115, 0.6045834125887297], 'champion_seeds': [0.6040754830744365, 0.6045684439544415, 0.6044516995706408], 'delta_mean': 0.00025, 'se': 0.000257, 'z': 0.97, 'sigma_pooled': 0.000315, 'sigma_df': 22, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/013.patch` (67 changed lines) · duration 102s · tokens in/out 92746/5555 · intervention: False

### n=14 — node_014 (deepen, parent 9)
**Hypothesis:** Diversify the five BPR members with antithetic negative-pool strata so they cover complementary same-user pairs rather than relying on correlated independent draws, then retain the existing normalized-rank average.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0005 (Node_009 gained +0.00134 fresh-seed mean from averaging five stochastic BPR members; deliberately reducing pair-sampling correlation should retain only a conservative fraction of that variance-reduction gain.)
**Result:** GAUC 0.6706 · nDCG@5 0.5377 · primary 0.6041 · realized Δ -0.0000 · rejected
**Diff:** `diffs/014.patch` (5 changed lines) · duration 86s · tokens in/out 89303/5964 · intervention: False

### n=15 — node_015 (deepen, parent 9)
**Hypothesis:** Replace epoch-wise with-replacement negative draws by independently seeded per-user cyclic negative queues so each BPR member covers its available negatives before repeating them without increasing the number of updates.
**Method:** cyclic-bpr-pair-coverage · target `training-schedule` · expected Δ 0.0004 (The champion samples only one random negative per positive per epoch and node_009's +0.00134 seed-ensemble gain demonstrates material stochastic ordering variance, but unchanged information and pair count cap the expected improvement near the seed-noise scale.)
**Result:** GAUC 0.6705 · nDCG@5 0.5369 · primary 0.6037 · realized Δ -0.0004 · rejected
**Diff:** `diffs/015.patch` (14 changed lines) · duration 84s · tokens in/out 85226/9130 · intervention: False

#### generation 3 closed — no improvement; streak 1; champion node_009; best 0.6044; tokens 702843/55702; 692s
_Diagnosis:_ Dynamics: primary rises from 0.5945 at epoch 1 to 0.6041 at epoch 8, then is flat through epoch 12; champion GAUC/nDCG@5/ndcg5_disc = 0.6709/0.5373/0.7239.
Node 6 failed implementation before scoring, so neither metric half was measured.
Node 7 LightGBM moved GAUC −0.0036 and nDCG@5 −0.0010 (disc −0.0017): both halves worsened; dead here.
Node 8 stronger L2 moved GAUC +0.0007 and nDCG@5 +0.0004 (disc +0.0006), but fresh-seed gain was only +0.0001.
Node 9 seed averaging moved GAUC +0.0013 and nDCG@5 +0.0008 (disc +0.0013): broad ordering and top-list ranking both improved.
Node 10 session features left GAUC flat (−0.0000) but moved nDCG@5 +0.0006 (disc +0.0009), indicating a top-list-specific signal.
Next probe: add exposure-session features to every five-seed BPR member, targeting features; Data Fact §10.5 shows attention decay from 0.418 at session start to 0.137 after position 30.
Validation risk is material despite streak 0: six accepted wins are below 0.002; weakest meaningful areas are dur>180s (0.5044) and tab=4 GAUC (0.5781), with node 10 moving them +0.0010/+0.0037—deepen tab-conditioned session fatigue; no groups are HARD.
_Plan for next generation:_ `{"note": "No merge is eligible: no generation-3 node improved the champion by at least 0.001, and the generation did not advance the champion.\nNode_013 is the strongest near-miss, with gains concentrated in dur>180s and tab=4 but harm on short videos and tab=6; prioritize a gated deepen, one repaired history retest, and one untried-family exploration.", "plan": [{"type": "deepen", "parent": 13, "card": "features-exposure-session", "mechanism": "cohort-gated-session-fatigue-crosses", "target_group": "dur>180s and tab=4", "hypothesis": "Replace the globally shared session fields with compact session-position and recent-density crosses activated only for dur>180s or tab=4 rows, preserving node_013's +0.0034 and +0.0020 subgroup gains while preventing its losses on dur<18s and tab=6."}, {"type": "retest", "parent": "champion", "card": "rocchio-contrastive-content-history", "hypothesis": "Add a positive-minus-negative candidate-content affinity field using fixed label-independent semantic bins and equal-time-safe accumulation, then train it in every member of the five-seed BPR ensemble.", "reason": "The earlier branch never ran because quantile thresholds indirectly depended on later labels; fixed bins remove that leakage bug, and the current five-seed BPR stack is materially different from its original single-model parent."}, {"type": "retest", "parent": "champion", "card": "last-positive-attribute-recurrence", "hypothesis": "Add latest strictly earlier positive tag, music, video-type match, and recency fields to every five-seed BPR member, then rank-average as in the champion.", "reason": "This history signal previously achieved a confirmed +0.0009 seed-mean gain on pointwise FM; BPR plus seed averaging is a changed stack that may stabilize and better exploit its broad GAUC improvement."}, {"type": "explore", "parent": "champion", "card": "model-din-history-attention", "hypothesis": "Add candidate-conditioned attention over strictly earlier positive history attributes to the five-seed BPR champion so tag, author, duration, and tab similarity can personalize candidates whose authors were unseen in the user's training history."}]}`

### n=16 — node_016 (explore, parent 9)
**Hypothesis:** Append equal-time-safe buckets for each video's and author's trailing-24-hour exposure count and previous-exposure gap so every BPR ensemble member can track current catalogue momentum that static identity embeddings cannot represent.
**Method:** online-item-exposure-momentum · target `features` · expected Δ 0.0006 (Data Fact 5 documents a roughly 10-fold traffic collapse and temporal label drift, while Koren's TimeSVD++ motivates time-varying item effects; the estimate is conservative because these label-free logging trends may partly reflect policy rather than relevance.)
**Result:** GAUC 0.6701 · nDCG@5 0.5379 · primary 0.6040 · realized Δ -0.0001 · rejected
**Diff:** `diffs/016.patch` (97 changed lines) · duration 157s · tokens in/out 92314/10421 · intervention: False

### n=17 — node_017 (explore, parent 9)
**Hypothesis:** Add a narrow candidate-conditioned attention residual over each user's strictly earlier positive tag, author, duration, and tab history to every BPR ensemble member so candidates with unseen authors can still match transferable interests.
**Method:** model-din-history-attention · target `model` · expected Δ 0.0007 (The card range is [0.000, 0.004] and Data Fact §10.3 says 96.6% of valid candidates have unseen user-author pairs, while the lower estimate reflects that simpler history features were nearly flat on BPR.)
**Result:** GAUC 0.6708 · nDCG@5 0.5372 · primary 0.6040 · realized Δ -0.0001 · rejected
**Diff:** `diffs/017.patch` (137 changed lines) · duration 302s · tokens in/out 95330/18743 · intervention: False

### n=18 — node_018 (retest, parent 9)
**Hypothesis:** Add an equal-time-safe signed candidate-content affinity field contrasting strictly earlier positive and negative tag, music, and video-type matches, using fixed levels that cannot depend on later labels.
**Method:** rocchio-contrastive-content-history · target `history` · expected Δ 0.0007 (The original branch expected +0.0008 but never ran because its quantile edges leaked later-label information; node_001's simpler latest-positive recurrence subsequently achieved +0.0009 fresh-seed mean.)
**Result:** GAUC 0.6705 · nDCG@5 0.5371 · primary 0.6038 · realized Δ -0.0003 · rejected
**Diff:** `diffs/018.patch` (85 changed lines) · duration 146s · tokens in/out 91949/9831 · intervention: False

### n=19 — node_019 (deepen, parent 13)
**Hypothesis:** Train the base champion beside node_013 and substitute the session model's normalized rank only for dur>180s or tab-4 rows, retaining the base ordering elsewhere.
**Method:** cohort-gated-session-parent-child-rank-fusion · target `ensembling` · expected Δ 0.0005 (Node_013 was a +0.00048 seed-0 near-miss with gains concentrated at +0.0034 for dur>180s and +0.0020 for tab 4, so prediction-level gating may recover an acceptance-scale fraction without its short-video and tab-6 losses.)
**Result:** GAUC 0.6715 · nDCG@5 0.5378 · primary 0.6046 · realized Δ +0.0005 · rejected · seed confirmation {'node_seed0': 0.6046074296225528, 'node_seeds': [0.60459485340838, 0.604861559954103, 0.60512336709234], 'champion_seeds': [0.6040754830744365, 0.6045684439544415, 0.6044516995706408], 'delta_mean': 0.00049, 'se': 0.000255, 'z': 1.94, 'sigma_pooled': 0.000312, 'sigma_df': 24, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/019.patch` (42 changed lines) · duration 203s · tokens in/out 90377/7789 · intervention: False

### n=20 — node_020 (deepen, parent 13)
**Hypothesis:** Add a weak auxiliary pair loss on node_013 that prefers earlier, lower-density impressions over later high-density impressions within the same session for dur>180s or tab-4 rows.
**Method:** session-monotonic-auxiliary-pair-loss · target `loss` · expected Δ 0.0005 (Data Fact §10.5 shows a monotonic tab-1 rate decline from 0.418 at session start to 0.137 beyond position 30, and node_013 moved nDCG@5 +0.0008 with its largest gains in the targeted cohorts.)
**Result:** GAUC 0.6673 · nDCG@5 0.5363 · primary 0.6018 · realized Δ -0.0023 · rejected
**Diff:** `diffs/020.patch` (51 changed lines) · duration 137s · tokens in/out 89818/11585 · intervention: False

- _event_ (generation 4): librarian (web search) added cards: ['loss-approxndcg-soft-ranks', 'model-attentional-factorization-machine']

#### generation 4 closed — no improvement; streak 2; champion node_009; best 0.6044; tokens 838387/78865; 1018s
_Diagnosis:_ Dynamics: flat—primary rises from 0.5945 at epoch 1 to 0.6041 at epoch 8, then stays 0.6041 through epoch 12; GAUC/nDCG@5/ndcg5_disc = 0.6709/0.5373/0.7239.
Node 11 moved GAUC +0.0001 and nDCG@5 +0.0003 (disc +0.0006), but its fresh-seed gain was only +0.0002; short-term attribute overlap is effectively dead here.
Node 12 moved both halves backward: GAUC −0.0009 and nDCG@5 −0.0001, so the FPMC transition term is dead on this stack.
Node 13 mainly improved top-list ordering: GAUC +0.0002 versus nDCG@5 +0.0008 (disc +0.0014), though fresh-seed gain was only +0.0003.
Node 14 traded GAUC −0.0003 for nDCG@5 +0.0004 and netted zero; node 15 reduced GAUC −0.0004 and nDCG@5 −0.0004.
Next probe: deepen the features-exposure-session component with cohort-gated session fatigue for dur>180s or tab=4; Data Fact §10.5 shows strong attention decay with session position and recent density.
Validation-overfitting risk is material: streak is 1, and six accepted wins in this run were all below 0.002, so subgroup-selected refinements require strict fresh-seed confirmation.
Ignoring structural dur=0/tab=0 weakness, dur>180s is weakest (primary 0.5044) and tab=4 has weakest meaningful GAUC (0.5781); node 13 improved them +0.0034 and +0.0020 respectively, with no HARD groups blocking this deepen.
_Plan for next generation:_ `{"note": "No merge is eligible: no generation-4 node improved the champion by >=0.001, and the strongest near-miss (node_019) targets a now-HARD group with a closed fusion mechanism.\nWith convergence one generation away, prioritize one changed-stack retest of the only confirmed history signal and one untried loss family aimed directly at the lagging nDCG half.", "plan": [{"type": "retest", "parent": "champion", "card": "last-positive-attribute-recurrence", "hypothesis": "Add leakage-safe latest strictly earlier positive tag, music, video-type match, and recency fields to every member of the five-seed BPR ensemble, then retain normalized within-user rank averaging.", "reason": "Node_001 achieved a confirmed +0.0009 seed-mean gain on a single pointwise FM, but it has never been tested on the materially changed metric-aligned BPR plus five-seed ensemble stack; its gain was broad GAUC signal rather than the session mechanisms closed this run."}, {"type": "explore", "parent": "champion", "card": "loss-approxndcg-soft-ranks", "hypothesis": "Add a low-weight ApproxNDCG auxiliary pass over sampled mixed-label user lists while retaining ordinary BPR as the main objective, using smooth ranks and a differentiable top-five cutoff to target nDCG@5 without replacing the stable pairwise loss."}]}`

### n=21 — node_021 (explore, parent 9)
**Hypothesis:** Add equal-time-safe previous-tab, current-tab transition, and same-tab streak fields so the ranker can distinguish entering a feed or profile surface from continuing within it, even when the current tab is identical.
**Method:** previous-tab-transition-state · target `features` · expected Δ 0.0006 (Data Fact §4 shows current-tab positive rates spanning 0.04–0.49, while §10.2 shows exposure behavior concentrated in profile/series tabs; a low-cardinality Markov state may capture residual interface context absent from the champion.)
**Result:** GAUC 0.6710 · nDCG@5 0.5372 · primary 0.6041 · realized Δ +0.0000 · rejected · seed confirmation {'node_seed0': 0.6041399720145135, 'node_seeds': [0.6041426361240668, 0.603869317924492, 0.6041127397853241], 'champion_seeds': [0.6040754830744365, 0.6045684439544415, 0.6044516995706408], 'delta_mean': -0.00032, 'se': 0.000248, 'z': -1.3, 'sigma_pooled': 0.000304, 'sigma_df': 26, 'sigma_from_node_only': False, 'adaptive': False, 'rule': 'fresh-seed mean gain >= 0.0005 and z >= 3.0 with the pooled seed SD'}
**Diff:** `diffs/021.patch` (69 changed lines) · duration 86s · tokens in/out 189481/13985 · intervention: False

### n=22 — node_022 (explore, parent 9)
**Hypothesis:** Add a low-weight ApproxNDCG auxiliary pass over sampled mixed-label user lists while retaining ordinary BPR, so smooth top-five rank gradients directly target the champion's lagging nDCG@5 half.
**Method:** loss-approxndcg-soft-ranks · target `loss` · expected Δ 0.0005 (The untried card ranges from 0.000 to 0.0012, and the diagnosis shows BPR's remaining weakness is top-list ranking while related wholesale loss replacements failed, supporting an acceptance-scale but conservative auxiliary-loss estimate.)
**Result:** GAUC 0.6703 · nDCG@5 0.5369 · primary 0.6036 · realized Δ -0.0005 · rejected
**Diff:** `diffs/022.patch` (58 changed lines) · duration 82s · tokens in/out 94851/12654 · intervention: False

### n=23 — node_023 (retest, parent 9)
**Hypothesis:** Add leakage-safe matches against each user's latest strictly earlier positive tag, music ID, and video type plus recency to every five-seed BPR member, then retain normalized within-user rank averaging.
**Method:** last-positive-attribute-recurrence · target `history` · expected Δ 0.0006 (Node_001 delivered a confirmed fresh-seed mean gain of 0.0009 on pointwise FM, while Data Fact 10.1 reports P(long_view)=0.78 following a positive same-author exposure; calibration is reduced because simpler history signals were flat under BPR.)
**Result:** GAUC 0.6706 · nDCG@5 0.5372 · primary 0.6039 · realized Δ -0.0002 · rejected
**Diff:** `diffs/023.patch` (88 changed lines) · duration 98s · tokens in/out 95380/7953 · intervention: False

### n=24 — node_024 (deepen, parent 9)
**Hypothesis:** Replace the champion's ordinary mean of five member ranks with a per-row trimmed mean that drops the highest and lowest member ranks, reducing the influence of seed-specific ordering outliers while preserving the same trained models.
**Method:** ensembling-seed-average · target `ensembling` · expected Δ 0.0003 (Node_009 gained 0.00134 fresh-seed mean from rank averaging and the pooled seed SD remains about 0.0003, so robust aggregation can plausibly recover only a small residual fraction without adding signal.)
**Result:** GAUC 0.6708 · nDCG@5 0.5373 · primary 0.6040 · realized Δ -0.0001 · rejected
**Diff:** `diffs/024.patch` (2 changed lines) · duration 74s · tokens in/out 91416/3873 · intervention: False

- _event_ (generation 5): librarian (web search) added cards: ['loss-warp-within-user-rank-weighting', 'model-neural-factorization-machine']

#### generation 5 closed — no improvement; streak 3; champion node_009; best 0.6044; tokens 912413/55099; 575s
_Diagnosis:_ Dynamics: flat—primary reaches 0.6041 at epoch 8 and remains 0.6041 through epoch 12; champion GAUC/nDCG@5/mixed-user nDCG = 0.6709/0.5373/0.7239.
Node 16 traded GAUC −0.0008 for nDCG@5 +0.0006 (mixed +0.0010); exposure momentum lacks a net gain here.
Node 17 moved neither half materially: GAUC −0.0001, nDCG@5 −0.0001; DIN attention is dead on this stack.
Node 18 reduced GAUC −0.0004 and nDCG@5 −0.0002; repaired Rocchio history is dead here.
Node 19 improved both halves, GAUC +0.0006 and nDCG@5 +0.0005, but fresh-seed Δ +0.00049 at z=1.94 failed acceptance; node 20 harmed GAUC −0.0036 and nDCG@5 −0.0010.
Most informative next probe: retest latest-positive attribute recurrence in the history component on the five-seed BPR stack; Data Fact §10.1 reports P(long_view)=0.78 after a prior positive same-author exposure.
Validation-overfitting risk is high: streak 2 of 3, with six accepted wins below 0.002; require strict fresh-seed confirmation.
Weakest meaningful groups are dur>180s and tab=4 GAUC; node 19 moved them +0.0034/+0.0020, but both are HARD and its fusion is closed—move on; among remaining cohorts dur18–60s is weakest, where node 20 gained +0.0018 despite failing globally.
_Plan for next generation:_ `{"note": "No generation-5 node improved the champion, and none qualifies for a merge; ApproxNDCG, recurrence history, transition features, and trimmed rank aggregation all regressed or were seed-negative.\nThe non-improving streak reached 3, so ADR-0012 convergence stops the run and no next generation should be scheduled.", "plan": []}`
