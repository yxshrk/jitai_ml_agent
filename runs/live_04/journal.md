# Run journal — live_04

## Summary
```json
{
 "run_id": "live_04",
 "stop_reason": "converged: 3 generations without > 0.002 improvement",
 "generations": 6,
 "nodes": 28,
 "champion": 15,
 "champion_metrics": {
  "gauc": 0.6715397677056579,
  "ndcg5": 0.5374037761703966,
  "primary": 0.6044717719380273,
  "ndcg5_disc": 0.7241460514629875
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.003,
 "top3_valid": [
  {
   "n": 27,
   "primary": 0.6045937128226722
  },
  {
   "n": 26,
   "primary": 0.6044768498785485
  },
  {
   "n": 15,
   "primary": 0.6044717719380273
  }
 ],
 "designated": 15,
 "final_ranking": [
  {
   "n": 15,
   "valid_primary": 0.6044717719380273,
   "fresh_seeds": [
    0.6044998700718957,
    0.6039909929412901
   ],
   "accepted": true,
   "mean": 0.6042454315065928,
   "std": 0.0003598304698419333,
   "tie_break": "within one SE (0.00022) of the best mean; accepted lineage preferred"
  },
  {
   "n": 26,
   "valid_primary": 0.6044768498785485,
   "fresh_seeds": [
    0.6045036884104502,
    0.6039955833890549
   ],
   "accepted": false,
   "mean": 0.6042496358997526,
   "std": 0.00035928450618356015
  },
  {
   "n": 27,
   "valid_primary": 0.6045937128226722,
   "fresh_seeds": [
    0.604334902493374,
    0.6039821565726058
   ],
   "accepted": false,
   "mean": 0.6041585295329899,
   "std": 0.00024942903261106675
  }
 ],
 "usage": {
  "calls": 99,
  "tokens_in": 1920190,
  "tokens_out": 392987,
  "cache_read": 1471664,
  "cache_write": 0,
  "cost_usd": 14.768072000000002
 },
 "wall_clock_s": 5796.7,
 "champion_seed_mean": 0.60425,
 "best_single_seed": null,
 "convergence_rule": "ADR-0012 (revised): 3 generations without a seed-confirmed champion change",
 "official_rule": null,
 "official_rule_submission": {
  "note": "the literal single-seed rule had not converged when the run ended",
  "node": null
 },
 "convergence_switch": "confirmed",
 "tokens": {
  "in_uncached": 1920190,
  "in_cached": 1471664,
  "out": 392987
 },
 "interventions": 0,
 "k": 5,
 "k_later": 5,
 "eps": 0.002,
 "n_converge": 3,
 "iteration_unit": "node",
 "iterations_used": 28
}
```

## Iterations

### n=0 — node_000 (reproduce_baseline, parent None)
**Hypothesis:** Reproduce the official FM baseline under the harness contract.
**Method:** official FM · target `None` · expected Δ 0.0 (published valid primary 0.6016)
**Result:** GAUC 0.6671 · nDCG@5 0.5358 · primary 0.6015
**Diff:** `None` (None changed lines) · duration 16s · tokens in/out 0/0 · intervention: False

### n=1 — node_001 (explore, parent 0)
**Hypothesis:** Replace each feature's shared FM embedding with field-specific embeddings so user-video, user-tab, and video-duration interactions can learn distinct representations without changing the ranking loss.
**Method:** field-aware FM embeddings · target `model` · expected Δ 0.0015 (Field-aware Factorization Machines (Juan et al., RecSys 2016) improve sparse interaction modeling, while Data Fact #1's closed catalogue makes field-specific user and video embeddings estimable, though the flat embedding-size ablation warrants a conservative estimate.)
**Result:** GAUC 0.6692 · nDCG@5 0.5368 · primary 0.6030 · realized Δ +0.0016 · ACCEPTED · seed confirmation {'node_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00122, 'se': 0.00023, 't': 5.22, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/001.patch` (46 changed lines) · duration 28s · tokens in/out 113838/18873 · intervention: False

### n=2 — node_002 (retest, parent 0)
**Hypothesis:** Replace pointwise logloss with within-user BPR pairs so optimization directly improves positive-negative ordering while cancelling metric-irrelevant user constants.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.0022 (The card range is 0.002-0.010, and prior official-FM retests realized +0.0016 to +0.0022 with strong seed significance.)
**Result:** GAUC 0.6692 · nDCG@5 0.5365 · primary 0.6028 · realized Δ +0.0014 · ACCEPTED · seed confirmation {'node_seeds': [0.6028286789333857, 0.6029525159912483, 0.601968666398895], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00114, 'se': 0.0003, 't': 3.83, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/002.patch` (37 changed lines) · duration 12s · tokens in/out 37768/6776 · intervention: False

### n=3 — node_003 (explore, parent 0)
**Hypothesis:** Add leakage-safe bucketized user-by-author, user-by-tab, and user-by-duration historical long-view rates to improve ordering for sparse users.
**Method:** history-user-aggregates · target `history` · expected Δ 0.0015 (The card range is 0.001-0.006 and Fact #2 reports a median of only 35 training rows per valid user, supporting smoothed aggregates but keeping the estimate in the lower third.)
**Result:** GAUC 0.6682 · nDCG@5 0.5360 · primary 0.6021 · realized Δ +0.0007 · ACCEPTED · seed confirmation {'node_seeds': [0.6021274265941343, 0.6026681982781319, 0.602684307383863], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00105, 'se': 0.00022, 't': 4.83, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/003.patch` (128 changed lines) · duration 26s · tokens in/out 116991/48176 · intervention: False

### n=4 — node_004 (retest, parent 0)
**Hypothesis:** Increase sparse embedding L2 modestly so the model retains its epoch-7 ranking peak instead of fitting id-specific noise through epoch 11.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.001 (The card range is 0.000-0.004, and the journal curve supplies specific overfitting evidence by falling from 0.6015 at epoch 7 to 0.5990 at epoch 11.)
**Result:** GAUC 0.6685 · nDCG@5 0.5365 · primary 0.6025 · realized Δ +0.0010 · ACCEPTED · seed confirmation {'node_seeds': [0.6025029206269998, 0.6028729444838677, 0.6017090117891591], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00092, 'se': 0.00032, 't': 2.86, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/004.patch` (2 changed lines) · duration 24s · tokens in/out 37543/2927 · intervention: False

#### generation 1 closed — no improvement; streak 1; champion node_001; best 0.6015; tokens 374014/84555; 814s
_Diagnosis:_ Dynamics: clear overfit; validation primary peaks at epoch 7 (0.6015), then falls steadily to 0.5990 by epoch 11.
Champion metrics: GAUC 0.6671, nDCG@5 0.5358, estimated mixed-user nDCG@5 ≈0.721 using Fact #7 cohort shares.
Last generation had no child nodes, so there are no GAUC-versus-nDCG movements to diagnose.
Next probe: within-user BPR pairwise loss, targeting ranking-loss alignment; Fact #7 shows most useful evaluation signal comes from mixed-label users.
Validation-overfitting risk is currently low: non-improving streak 0 and zero sub-0.002 “wins” accepted, though the peaked curve requires early stopping.
_Plan for next generation:_ `{"note": "Field-aware embeddings are the new champion, while BPR and stronger L2 each delivered confirmed gains from different target components.\nPrioritize orthogonal merges; retest history only because field-aware partner-specific embeddings may represent its added fields more effectively.", "plan": [{"type": "merge", "merge_parents": [1, 2], "hypothesis": "Combine field-aware embeddings with within-user BPR so the richer interaction representation is optimized directly for within-user ordering rather than pointwise calibration."}, {"type": "merge", "merge_parents": [1, 4], "hypothesis": "Apply stronger embedding L2 to the field-aware champion; its expanded parameterization peaks sharply at epoch 5 and then degrades, making targeted shrinkage especially relevant."}, {"type": "retest", "parent": "champion", "card": "history-user-aggregates", "hypothesis": "Add leakage-safe user-by-author, user-by-tab, and user-by-duration history fields to the field-aware champion, allowing each history signal to learn distinct interactions with row fields.", "reason": "The stack changed from shared FM embeddings to field-aware embeddings, and history had a confirmed seed-mean gain despite a weak single-seed delta; the new representation may remove its prior interaction bottleneck."}]}`

### n=5 — node_005 (explore, parent 1)
**Hypothesis:** Add strictly prior user-video exposure count and time-since-last-exposure fields so repeated impressions can receive different scores and the field-aware FM can learn familiarity versus fatigue effects.
**Method:** repeat-exposure fatigue encoding · target `history` · expected Δ 0.0015 (Data Fact #8 says 5.7% of valid rows belong to repeated user-video pairs, providing a small but directly affected cohort and therefore only a lower-third history-family gain is expected.)
**Result:** GAUC 0.6691 · nDCG@5 0.5360 · primary 0.6025 · realized Δ -0.0005 · rejected
**Diff:** `diffs/005.patch` (56 changed lines) · duration 44s · tokens in/out 38646/6990 · intervention: False

### n=6 — node_006 (merge, parent 1, merge of [1, 2])
**Hypothesis:** Combine field-aware embeddings with within-user BPR so partner-specific interaction capacity is optimized directly for positive-negative ordering rather than cross-user calibration.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.001 (The card range starts at +0.002, but live_04 node_002 realized only +0.0011 seed-mean and both parents reached GAUC 0.6692, indicating substantial overlap and warranting a calibrated +0.0010 merge estimate.)
**Result:** GAUC 0.6692 · nDCG@5 0.5368 · primary 0.6030 · realized Δ +0.0000 · rejected
**Diff:** `diffs/006.patch` (3 changed lines) · duration 33s · tokens in/out 40918/5034 · intervention: False

### n=7 — node_007 (merge, parent 1, merge of [1, 4])
**Hypothesis:** Apply node_004's modestly stronger embedding L2 to the expanded field-aware model because its validation primary peaks sharply at epoch 5 and then deteriorates every epoch.
**Method:** regularization-embedding-dropout-l2 · target `regularization` · expected Δ 0.0012 (The card's lower-third expectation is at most about +0.0013, node_004 delivered +0.0009 seed-mean, and node_001's sharper overfitting provides specific evidence for a small uplift toward +0.0012.)
**Result:** GAUC 0.6695 · nDCG@5 0.5370 · primary 0.6032 · realized Δ +0.0002 · rejected · seed confirmation {'node_seeds': [0.60323492065835, 0.6025816053625668, 0.6024817289375326], 'champion_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'delta_mean': 0.00011, 'se': 0.00026, 't': 0.43, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/007.patch` (2 changed lines) · duration 34s · tokens in/out 40673/3260 · intervention: False

### n=8 — node_008 (retest, parent 1)
**Hypothesis:** Halve the learning rate after the first stalled validation epoch so the sharply overfitting field-aware model can settle near its epoch-5 optimum instead of moving rapidly away from it.
**Method:** training-schedule-lr-decay-early-stop · target `training-schedule` · expected Δ 0.0006 (The schedule card ranges from +0.000 to +0.003 and previously lost 0.0004 on the official FM, but node_001's uninterrupted decline from 0.6030 to 0.5958 gives stack-specific evidence for a cheap lower-third retest.)
**Result:** GAUC 0.6692 · nDCG@5 0.5368 · primary 0.6030 · realized Δ +0.0000 · rejected
**Diff:** `diffs/008.patch` (4 changed lines) · duration 34s · tokens in/out 38124/3597 · intervention: False

#### generation 2 closed — no improvement; streak 2; champion node_001; best 0.6015; tokens 230022/25616; 372s
_Diagnosis:_ Dynamics: clear overfit—primary rises from 0.5944 (epoch 1) to 0.6030 (epoch 5), then falls every epoch to 0.5958 (epoch 9).
Field-aware FM: GAUC +0.00214 versus baseline, nDCG@5 +0.00104; improvement is mainly pairwise ordering (disc nDCG 0.7232).
BPR: GAUC +0.00205, nDCG@5 +0.00070; strongly GAUC-led, indicating limited top-of-list improvement (disc nDCG 0.7226).
History aggregates: GAUC +0.00110, nDCG@5 +0.00025; nearly flat at the top, so this feature design is weak here (disc 0.7218).
Higher L2: GAUC +0.00139, nDCG@5 +0.00071; modest variance control, not a solution to top-rank errors (disc 0.7226).
Next probe: ListNet-style within-user softmax on the field-aware champion, targeting the loss/top-of-list ordering; Fact #7 says valid lists average only 5.6 rows, making whole-list optimization practical and nDCG-relevant.
Validation-overfitting risk is high: after one non-improving generation, all 4 accepted nodes were sub-0.002 single-run “wins,” increasing selection bias despite seed confirmation.
_Plan for next generation:_ `{"note": "No node improved the champion meaningfully; the apparent L2 gain was only +0.00011 across seeds, while BPR and LR-decay were exactly flat.\nUse one exploration slot on the untried listwise-loss family, since field-aware capacity is established but its current pairwise objective may miss whole-list/top-of-list structure; leave four slots to the Selector.", "plan": [{"type": "explore", "parent": "champion", "card": "loss-listwise-softmax-within-user", "hypothesis": "Replace pure BPR with per-user ListNet softmax on the field-aware FM so each update optimizes the full within-user list, potentially improving nDCG@5 where merging BPR-related refinements and schedule changes was flat."}]}`

### n=9 — node_009 (explore, parent 1)
**Hypothesis:** Append leakage-safe, bucketized empirical long-view rates for video×tab and author×tab, using only strictly earlier training impressions, so the field-aware FM receives direct high-resolution estimates of context-specific item quality rather than learning every interaction through low-rank embeddings.
**Method:** ordered item-context target statistics · target `encoding` · expected Δ 0.002 (CatBoost ordered target statistics (Prokhorenkova et al., 2018) provide leakage-resistant categorical estimates, while Data Facts 1 and 4 show a closed catalogue and extreme tab-dependent behavior, making well-supported video×tab rates directly relevant to within-user ordering.)
**Result:** GAUC 0.6694 · nDCG@5 0.5360 · primary 0.6027 · realized Δ -0.0004 · rejected
**Diff:** `diffs/009.patch` (57 changed lines) · duration 57s · tokens in/out 38888/8119 · intervention: False

### n=10 — node_010 (explore, parent 1)
**Hypothesis:** Replace pointwise logloss with mixed-user ListNet softmax on the field-aware FM so each update optimizes the complete within-user ordering and can improve top-five placement where the BPR merge was flat.
**Method:** loss-listwise-softmax-within-user · target `loss` · expected Δ 0.0018 (The card range is +0.001 to +0.008; lower-third calibration gives about +0.0018, while Data Fact 7's 5.6-row average validation lists makes whole-list optimization directly relevant to nDCG@5.)
**Result:** GAUC 0.6653 · nDCG@5 0.5351 · primary 0.6002 · realized Δ -0.0029 · rejected
**Diff:** `diffs/010.patch` (64 changed lines) · duration 37s · tokens in/out 38332/8084 · intervention: False

### n=11 — node_011 (merge, parent 1)
**Hypothesis:** Port node_003's leakage-safe user-by-author, user-by-tab, and user-by-duration rate fields into the field-aware FM so its partner-specific embeddings can exploit the modest history signal more effectively.
**Method:** history-user-aggregates · target `history` · expected Δ 0.0008 (Journal node_003 showed a +0.0010 seed-mean gain on the baseline, but prior history results and merge attenuation justify discounting the expected transfer to +0.0008.)
**Result:** GAUC 0.6682 · nDCG@5 0.5352 · primary 0.6017 · realized Δ -0.0014 · rejected
**Diff:** `diffs/011.patch` (69 changed lines) · duration 71s · tokens in/out 38971/11315 · intervention: False

### n=12 — node_012 (merge, parent 1)
**Hypothesis:** Average within-user normalized ranks from the field-aware champion and the close BPR sibling so their structurally different ordering errors can cancel after two generations without a model-level improvement.
**Method:** heterogeneous-node-rank-average · target `ensembling` · expected Δ 0.001 (The ensembling card range is +0.001 to +0.003, and nodes 001 and 002 differ by only 0.0002 primary while using different model and loss choices, supporting a calibrated lower-bound estimate of +0.0010.)
**Result:** GAUC 0.6699 · nDCG@5 0.5373 · primary 0.6036 · realized Δ +0.0006 · rejected · seed confirmation {'node_seeds': [0.6035959202143739, 0.6029689836439411, 0.603092376144956], 'champion_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'delta_mean': 0.00056, 'se': 0.00023, 't': 2.43, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/012.patch` (77 changed lines) · duration 49s · tokens in/out 78659/22411 · intervention: False

### n=13 — node_013 (explore, parent 1)
**Hypothesis:** Add one DCN-V2 cross layer above the field-aware embeddings so bounded higher-order user-tab-duration interactions become representable beyond the champion's pairwise FM terms.
**Method:** model-dcn-cross-head · target `model` · expected Δ 0.001 (The card range is +0.000 to +0.005; lower-third calibration suggests about +0.0010 after discounting for the champion's sharp epoch-5 overfitting.)
**Result:** GAUC 0.6694 · nDCG@5 0.5355 · primary 0.6025 · realized Δ -0.0006 · rejected
**Diff:** `diffs/013.patch` (50 changed lines) · duration 55s · tokens in/out 77964/19560 · intervention: False

#### generation 3 closed — improved; streak 0; champion node_001; best 0.6036; tokens 346203/78158; 522s
_Diagnosis:_ Dynamics: clear overfit; primary peaks at epoch 5 (0.6030), then falls monotonically to 0.6011 at 6 and 0.5958 at 9.
Champion: GAUC 0.6692, nDCG@5 0.5368, mixed-user nDCG@5 0.7232.
Node 005 history: GAUC −0.00013, nDCG@5 −0.00088, mixed nDCG −0.00152; damage is mainly top-of-list, so repeat-fatigue features are dead here.
Node 006 BPR merge: GAUC ±0.00000, nDCG@5 ±0.00000, mixed nDCG ±0.00000; no effective loss change or ranking gain.
Node 007 stronger L2: GAUC +0.00024, nDCG@5 +0.00015, mixed nDCG +0.00025; tiny broad movement, seed-mean only +0.00011.
Node 008 LR decay: GAUC ±0.00000, nDCG@5 ±0.00000, mixed nDCG ±0.00000; schedule changes the tail but not the selected checkpoint.
Next probe: within-user ListNet softmax, targeting the loss/top-list ordering; Data Fact 7 says valid lists average only 5.6 rows, making whole-list optimization directly relevant to nDCG@5.
Validation-overfitting risk is high after two non-improving generations and repeated reuse of one split; four sub-0.002 “wins” have already been accepted.
_Plan for next generation:_ `{"note": "No merge qualifies: this generation produced no two >=0.001 gains on distinct components, and the earlier field-aware+BPR/history/regularization merges were already flat or harmful.\nPrioritize the borderline heterogeneous ensemble, one mechanism-backed retest on the changed field-aware stack, and one architecture probe from the close BPR lineage; leave two slots to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "heterogeneous-node-rank-average", "hypothesis": "Rebuild the field-aware plus standard-BPR within-user rank ensemble with five-seed averaging per branch and test a conservative 0.6/0.4 field-aware-to-BPR blend, reducing both seed variance and dilution from the weaker sibling.", "reason": "Node_012 improved every seed with mean delta +0.00056 but narrowly missed confirmation at t=2.43 versus 2.5; this is weak evidence rather than a disproven mechanism."}, {"type": "retest", "parent": "champion", "card": "loss-watchtime-censored", "hypothesis": "Add a lightly weighted censored log-watch-time auxiliary head to the field-aware model so its interaction-specific embeddings learn near-miss ordering information unavailable from binary long_view alone.", "reason": "The prior test was only on the standard-FM+BPR stack and was mildly positive (+0.0003); field-aware embeddings materially change the shared representation available to the auxiliary task."}, {"type": "explore", "parent": 2, "card": "model-dcn-cross-head", "hypothesis": "Add the low-rank DCN cross head to the close standard-FM BPR lineage, where explicit higher-order crosses may complement shared embeddings instead of duplicating the champion's already field-specific interaction capacity."}]}`

### n=14 — node_014 (explore, parent 1)
**Hypothesis:** Train auxiliary heads for reaching 25%, 50%, and 75% of min(duration_ms, 18000) so shared field-aware embeddings distinguish immediate bounces from near misses while the original long-view head remains the ranking score.
**Method:** threshold-normalized ordinal watch-depth auxiliary · target `aux-targets` · expected Δ 0.002 (Data Fact 3 establishes that long_view is generated by progress toward min(duration, 18 seconds), while the multi-task ordinal-regression idea of MTLR provides graded supervision without treating completed short videos as uncensored latent watch times.)
**Result:** GAUC 0.6696 · nDCG@5 0.5370 · primary 0.6033 · realized Δ +0.0002 · rejected · seed confirmation {'node_seeds': [0.6032822768050394, 0.6024645687479453, 0.6026935358635411], 'champion_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'delta_mean': 0.00016, 'se': 0.00026, 't': 0.6, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/014.patch` (50 changed lines) · duration 37s · tokens in/out 77765/16199 · intervention: False

### n=15 — node_015 (retest, parent 1)
**Hypothesis:** Rebuild the field-aware plus standard-BPR within-user rank ensemble with five-seed averaging per branch and a conservative 0.6/0.4 field-aware-to-BPR blend to reduce seed variance and dilution from the weaker sibling.
**Method:** heterogeneous-node-rank-average · target `ensembling` · expected Δ 0.0008 (Node_012 improved both metrics with seed-mean delta +0.00056 and narrowly missed confirmation at t=2.43, while the seed-averaging card reports gains up to 0.003.)
**Result:** GAUC 0.6715 · nDCG@5 0.5374 · primary 0.6045 · realized Δ +0.0014 · ACCEPTED · seed confirmation {'node_seeds': [0.6044717719380273, 0.6044998700718957, 0.6039909929412901], 'champion_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'delta_mean': 0.00166, 'se': 0.00022, 't': 7.65, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/015.patch` (122 changed lines) · duration 143s · tokens in/out 77869/23884 · intervention: False

### n=16 — node_016 (retest, parent 1)
**Hypothesis:** Add a lightly weighted censored log-watch-time auxiliary head to the field-aware model so interaction-specific embeddings learn graded near-miss information unavailable from binary long_view alone.
**Method:** loss-watchtime-censored · target `loss` · expected Δ 0.001 (The card range is 0.001–0.008, Fact 3 establishes watch time as the label-generating mechanism, and its earlier BPR-stack test was mildly positive at +0.0003.)
**Result:** GAUC 0.6667 · nDCG@5 0.5353 · primary 0.6010 · realized Δ -0.0021 · rejected
**Diff:** `diffs/016.patch` (51 changed lines) · duration 56s · tokens in/out 38929/8117 · intervention: False

### n=17 — node_017 (explore, parent 1)
**Hypothesis:** Add a low-rank DCN cross head to the standard-FM BPR lineage, where explicit higher-order crosses may complement shared embeddings instead of duplicating the champion's field-specific interactions.
**Method:** model-dcn-cross-head · target `model` · expected Δ 0.0005 (The DCN card range is 0.000–0.005, but node_013's nDCG loss on the field-aware parent warrants a lower-third estimate on the structurally different BPR parent.)
**Result:** GAUC 0.6694 · nDCG@5 0.5370 · primary 0.6032 · realized Δ +0.0001 · ACCEPTED · seed confirmation {'node_seeds': [0.6031744724494806, 0.6032517319119846, 0.6031675665286507], 'champion_seeds': [0.6030401988832401, 0.6023209423995108, 0.6026079598478907], 'delta_mean': 0.00054, 'se': 0.00021, 't': 2.63, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/017.patch` (62 changed lines) · duration 17s · tokens in/out 38722/8861 · intervention: False

### n=18 — node_018 (retest, parent 1)
**Hypothesis:** Give duration-zero impressions a dedicated categorical value in the field-aware model so their always-negative status can push them below genuinely short videos within each user.
**Method:** features-duration-unknown-flag · target `features` · expected Δ 0.0003 (Fact 3 says 1.9% of rows have unknown duration and are always negative, but the prior BPR-stack measurement was only +0.0002 despite the card's 0.001–0.004 family range.)
**Result:** GAUC 0.6688 · nDCG@5 0.5368 · primary 0.6028 · realized Δ -0.0002 · rejected
**Diff:** `diffs/018.patch` (5 changed lines) · duration 34s · tokens in/out 38046/3439 · intervention: False

#### generation 4 closed — no improvement; streak 1; champion node_015; best 0.6036; tokens 347755/68595; 755s
_Diagnosis:_ Dynamics: clear overfit—primary peaks at epoch 5 (0.6030), then falls steadily to 0.5958 by epoch 9; champion GAUC 0.6692, nDCG@5 0.5368.
Node 009 encoding: GAUC +0.0002 but nDCG −0.0008 (mixed-user nDCG 0.7217); item-context rates hurt top-list placement.
Node 010 ListNet: GAUC −0.0039 and nDCG −0.0017 (disc 0.7201); the listwise loss broadly degraded ranking.
Node 011 history: GAUC −0.0010 and nDCG −0.0016 (disc 0.7203); these aggregates are dead under field-aware FM.
Node 012 rank ensemble: GAUC +0.0007 and nDCG +0.0005 (disc 0.7239); complementary errors help both halves, but below acceptance.
Node 013 DCN: GAUC +0.0002 but nDCG −0.0013 (disc 0.7209); extra capacity preserves global pairs while worsening the top five.
Next probe: a censored watch-time auxiliary/ranking head targeting supervision/loss; Fact 3 says watch time generates the label and provides graded near-miss information.
Validation-overfitting risk is high after 14 probes despite streak resetting to 0: four sub-0.002 “wins” have already been accepted.
_Plan for next generation:_ `{"note": "No merge qualifies: node_015 is the only node this generation improving the prior champion by at least 0.001; node_017 has a stable but smaller model-component gain.\nThe new multi-branch ensemble changes node_017's context: its low-rank DCN-BPR branch may add complementary ordering errors even though its standalone delta was modest, while the other retests are clearly flat or harmful.", "plan": [{"type": "retest", "parent": "champion", "card": "model-dcn-cross-head", "hypothesis": "Replace the champion ensemble's standard-BPR branch with node_017's low-rank DCN-BPR branch, retaining the five-seed field-aware branch and conservative within-user rank blending; explicit crosses may improve ensemble diversity without adding a third weak branch.", "reason": "Node_017 showed a stable seed-mean gain (+0.00054, t=2.63) on the BPR lineage, and the context has changed from standalone scoring to a variance-reduced heterogeneous ensemble where complementary errors can be more valuable than standalone primary."}]}`

### n=19 — node_019 (explore, parent 15)
**Hypothesis:** Blend the champion with a smoothed per-user-by-tab empirical probability of reaching each candidate's min(duration_ms, 18000) threshold, using historical play times and duration-based risk sets, to improve ordering of the difficult short-video candidates.
**Method:** at-risk user-tab watch-survival reranker · target `history` · expected Δ 0.002 (Data Facts 3 and 4 show that the label is threshold-defined, short videos are hardest, and tab strongly changes viewing behavior, while a Kaplan-Meier-style risk-set estimate combines the previously separate watch-time and history mechanisms without altering BPR.)
**Result:** GAUC 0.6714 · nDCG@5 0.5374 · primary 0.6044 · realized Δ -0.0001 · rejected
**Diff:** `diffs/019.patch` (59 changed lines) · duration 318s · tokens in/out 40813/12418 · intervention: False

### n=20 — node_020 (retest, parent 15)
**Hypothesis:** Replace the champion's standard-BPR branch with node_017's low-rank DCN-BPR branch while retaining five-seed averaging and the 0.6/0.4 rank blend, because its complementary higher-order interactions may improve the ensemble without dilution from a third branch.
**Method:** model-dcn-cross-head · target `model` · expected Δ 0.0006 (Journal node_017 recorded a stable +0.0005 seed-mean gain on the BPR lineage, within the DCN card's 0.000–0.005 range, although its modest standalone effect warrants a conservative estimate.)
**Result:** GAUC 0.6711 · nDCG@5 0.5372 · primary 0.6042 · realized Δ -0.0003 · rejected
**Diff:** `diffs/020.patch` (52 changed lines) · duration 329s · tokens in/out 40458/9954 · intervention: False

### n=21 — node_021 (retest, parent 15)
**Hypothesis:** Replace uniform BPR gradients in the champion's ranking branch with warm-started delta-nDCG@5-weighted pair gradients so training concentrates on positive-negative swaps that affect the top of each user's list.
**Method:** loss-lambdarank-pairs · target `loss` · expected Δ 0.0015 (The LambdaRank card ranges from 0.001–0.006, and Data Fact 7's 5.6-row average validation lists plus node_015's much smaller nDCG than GAUC gain provide specific evidence for a lower-third but nontrivial effect.)
**Result:** GAUC 0.6693 · nDCG@5 0.5364 · primary 0.6028 · realized Δ -0.0016 · rejected
**Diff:** `diffs/021.patch` (33 changed lines) · duration 441s · tokens in/out 80966/15840 · intervention: False

### n=22 — node_022 (explore, parent 15)
**Hypothesis:** Sample BPR users from an equal mixture of positive-count-proportional and user-uniform distributions so the branch balances GAUC's positive weighting with nDCG@5's equal per-user weighting.
**Method:** metric-matched user-balanced pair weighting · target `data-weighting` · expected Δ 0.0012 (Data Fact 7 shows short, heterogeneous validation lists, while the official metric gives each nDCG user equal weight but existing one-pair-per-positive BPR weights users by positive count, creating a direct objective-weight mismatch.)
**Result:** GAUC 0.6687 · nDCG@5 0.5364 · primary 0.6026 · realized Δ -0.0019 · rejected
**Diff:** `diffs/022.patch` (9 changed lines) · duration 319s · tokens in/out 39651/5691 · intervention: False

### n=23 — node_023 (explore, parent 15)
**Hypothesis:** Replace the champion's linear Borda-style rank average with a fixed reciprocal-rank fusion that preserves the 0.6/0.4 branch preference but gives greater influence to agreement and disagreements near the top five.
**Method:** top-heavy reciprocal-rank fusion · target `ensembling` · expected Δ 0.0007 (Journal node_015 gained about 0.0023 GAUC but only 0.0006 nDCG from rank averaging, so a five-line top-heavy fusion is a low-cost probe of the remaining top-list error.)
**Result:** GAUC 0.6715 · nDCG@5 0.5375 · primary 0.6045 · realized Δ -0.0000 · rejected
**Diff:** `diffs/023.patch` (16 changed lines) · duration 316s · tokens in/out 39860/5698 · intervention: False

#### generation 5 closed — no improvement; streak 2; champion node_015; best 0.6036; tokens 320434/60303; 973s
_Diagnosis:_ Dynamics: flat/plateau, not overfit—primary rises 0.5963 (epoch 1) → 0.6041 (4), reaches 0.6045 at 8, then remains 0.6045 through 12.
Champion: GAUC 0.6715, nDCG@5 0.5374, mixed-user nDCG@5 0.7241; improvement is predominantly GAUC.
Node 14 auxiliary vs parent: GAUC +0.0004, nDCG +0.0002 (disc 0.7235)—neither moved materially.
Node 15 ensemble: GAUC +0.0023, nDCG +0.0006 (disc 0.7241)—clear broad-pair ordering gain, little top-list gain.
Node 16 watch-time: GAUC −0.0025, nDCG −0.0015 (disc 0.7205); node 17 DCN: roughly +0.0002/+0.0002 (disc 0.7234).
Node 18 duration-unknown: GAUC −0.0004, nDCG ≈0.0000 (disc 0.7231)—this feature is dead in the field-aware stack.
Next probe: nDCG-weighted pairwise/top-5 reranking on the ensemble, targeting the loss/reranking component; Fact 7 says valid lists average only 5.6 rows, making top-order errors directly diagnosable.
Validation-overfitting risk is high after 19 adaptive nodes despite only a one-generation stale streak: six accepted “wins” were below the frozen 0.002 threshold.
_Plan for next generation:_ `{"note": "No node improved node_015; reciprocal-rank fusion was effectively tied, while the loss and sampling changes clearly degraded both metrics.\nUse one runner-up exploration and one narrowly revised retest; leave the other three slots to the Selector rather than recombining already failed components.", "plan": [{"type": "explore", "parent": 23, "card": "top5-gated hybrid rank fusion", "hypothesis": "Apply reciprocal-rank fusion only when the two branches disagree on membership or ordering within the top five, while retaining the champion's linear rank blend elsewhere; this may preserve GAUC while capturing node_023's slight nDCG gain."}, {"type": "retest", "parent": "champion", "card": "at-risk user-tab watch-survival reranker", "hypothesis": "Restrict the survival signal to known-duration short videos near the completion threshold and use a conservative 3\u20135% rank blend, avoiding dilution on long videos where the champion is already strong.", "reason": "Node_019 was only -0.00008 despite a broad 15% blend, so evidence against the underlying signal is weak; the suspected issue is excessive weight and application outside the hard short-video cohort."}]}`

### n=24 — node_024 (explore, parent 15)
**Hypothesis:** Replace the standard-BPR branch's raw user-video interaction with a one-layer LightGCN interaction so sparse user and video factors borrow signal from positive graph neighborhoods while the existing FM terms retain tab and duration context.
**Method:** one-layer LightGCN ID propagation · target `model` · expected Δ 0.002 (LightGCN (He et al., SIGIR 2020) shows that degree-normalized interaction-graph propagation strengthens collaborative filtering, and Data Facts 1–2 provide a closed catalogue with enough short user histories for one-hop smoothing but not reliable independent ID estimates.)
**Result:** GAUC 0.6711 · nDCG@5 0.5374 · primary 0.6043 · realized Δ -0.0002 · rejected
**Diff:** `diffs/024.patch` (42 changed lines) · duration 423s · tokens in/out 81278/23498 · intervention: False

### n=25 — node_025 (explore, parent 15)
**Hypothesis:** Apply reciprocal-rank fusion only to rows where the two champion branches disagree within the top five, retaining the linear rank blend elsewhere to seek the slight nDCG gain without disturbing GAUC.
**Method:** top5-gated hybrid rank fusion · target `ensembling` · expected Δ 0.0002 (Node_023 changed GAUC by -0.00004 and nDCG@5 by +0.00007, so a narrower gate is expected to retain only a small fraction of that top-list signal.)
**Result:** GAUC 0.6713 · nDCG@5 0.5373 · primary 0.6043 · realized Δ -0.0002 · rejected
**Diff:** `diffs/025.patch` (30 changed lines) · duration 256s · tokens in/out 40001/9006 · intervention: False

### n=26 — node_026 (retest, parent 15)
**Hypothesis:** Apply the survival reranker only to known-duration videos at or below 18 seconds with a 3–5% rank blend, avoiding the dilution from broadly reranking long videos.
**Method:** at-risk user-tab watch-survival reranker · target `history` · expected Δ 0.0003 (Node_019 was only -0.00008 with a broad 15% blend, while Fact 3 identifies short-video completion as the hard duration-specific regime, supporting a conservative targeted retest.)
**Result:** GAUC 0.6715 · nDCG@5 0.5374 · primary 0.6045 · realized Δ +0.0000 · rejected · seed confirmation {'node_seeds': [0.6044768498785485, 0.6045036884104502, 0.6039955833890549], 'champion_seeds': [0.6044717719380273, 0.6044998700718957, 0.6039909929412901], 'delta_mean': 0.0, 'se': 0.00019, 't': 0.02, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/026.patch` (36 changed lines) · duration 256s · tokens in/out 40383/10297 · intervention: False

### n=27 — node_027 (explore, parent 15)
**Hypothesis:** Add author_id only to the field-aware branch so user-author preference can pool evidence across an author's videos while the unchanged BPR branch preserves complementary item-level ordering.
**Method:** field-aware author-id feature · target `features` · expected Δ 0.0008 (Facts 1 and 2 show authors have no validation cold start and users have only a median 35 training rows, but the estimate is discounted because the organizer's static-field ablation was flat under the standard FM.)
**Result:** GAUC 0.6717 · nDCG@5 0.5375 · primary 0.6046 · realized Δ +0.0001 · rejected · seed confirmation {'node_seeds': [0.6045937128226722, 0.604334902493374, 0.6039821565726058], 'champion_seeds': [0.6044717719380273, 0.6044998700718957, 0.6039909929412901], 'delta_mean': -2e-05, 'se': 0.0002, 't': -0.09, 'rule': 'diff >= 0.0005 and t >= 2.5'}
**Diff:** `diffs/027.patch` (26 changed lines) · duration 304s · tokens in/out 39795/10752 · intervention: False

#### generation 6 closed — no improvement; streak 3; champion node_015; best 0.6036; tokens 301762/75760; 1446s
_Diagnosis:_ Dynamics: flat plateau, not overfit—primary rises 0.5963 at epoch 1 to 0.6041 at 4, then reaches 0.6045 at 8 and stays there through 12.
Champion: GAUC 0.6715, nDCG@5 0.5374; champion ndcg5_disc was not reported (near-identical node_023: 0.72425).
Node_019 history: GAUC −0.00013, nDCG +0.00001—neither half moved; survival history is dead here.
Node_020 DCN branch: GAUC −0.00039, nDCG −0.00018; node_021 LambdaRank: GAUC −0.00225, nDCG −0.00096—loss weighting damaged both, especially GAUC.
Node_022 user-balanced pairs: GAUC −0.00281, nDCG −0.00099—metric-matched sampling strongly harmed both halves.
Node_023 reciprocal-rank fusion: GAUC −0.00004, nDCG +0.00007—top-heavy fusion only redistributed negligible noise.
Next probe: sweep the field-aware/BPR blend weight using mixed-user nDCG as a diagnostic, targeting ensembling; Fact 7 says only 57.8% of valid users are discriminative, so raw nDCG masks top-list effects.
Validation-overfitting risk is high: the non-improving streak is 2, and six accepted nodes had gains below 0.002; require a full-threshold win rather than another marginal selection.
_Plan for next generation:_ `{"note": "No merge qualifies: every generation-6 node improved the champion by less than 0.001, and the two seed-confirmed deltas were flat.\nThe search is converged, so allocate only two focused slots to changed-context optimization and leave three slots to the Selector.", "plan": [{"type": "retest", "parent": 17, "card": "regularization-embedding-dropout-l2", "hypothesis": "Apply modest cross-head L2 and field-vector dropout to the low-rank DCN-BPR model so its complementary higher-order interactions survive without overfitting, then reassess it as an ensemble branch.", "reason": "The regularization card failed or was flat on ordinary FM variants, but node_017 introduced a parameter-rich DCN head and showed a positive seed-mean delta; this changed architecture is the context where targeted regularization is most justified."}, {"type": "explore", "parent": 2, "card": "loss-listwise-softmax-within-user", "hypothesis": "Keep standard within-user BPR as the dominant objective and add a low-weight user-normalized ListNet term, rather than replacing BPR, to improve top-list organization without sacrificing pairwise GAUC."}]}`
