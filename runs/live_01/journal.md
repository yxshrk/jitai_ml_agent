# Run journal — live_01

## Summary
```json
{
 "run_id": "live_01",
 "stop_reason": "generation cap 2",
 "generations": 2,
 "nodes": 7,
 "champion": 1,
 "champion_metrics": {
  "gauc": 0.6703073475190711,
  "ndcg5": 0.5369853914191499,
  "primary": 0.6036463694691105,
  "ndcg5_disc": 0.7234219277427735
 },
 "baseline_valid_primary": 0.6014687563529677,
 "delta_vs_baseline_valid": 0.00218,
 "top3_valid": [
  {
   "n": 4,
   "primary": 0.6039248002601583
  },
  {
   "n": 1,
   "primary": 0.6036463694691105
  },
  {
   "n": 5,
   "primary": 0.6031393484617817
  }
 ],
 "designated": 4,
 "final_ranking": [
  {
   "n": 4,
   "valid_primary": 0.6039248002601583,
   "seeds": [
    0.6039248002601583,
    0.6029761772835946,
    0.6026772156077824
   ],
   "mean": 0.6031927310505117,
   "std": 0.0005318449126612243
  },
  {
   "n": 1,
   "valid_primary": 0.6036463694691105,
   "seeds": [
    0.6036463694691105,
    0.6031184279276939,
    0.6027235946155616
   ],
   "mean": 0.6031627973374554,
   "std": 0.00037802543210522675
  },
  {
   "n": 5,
   "valid_primary": 0.6031393484617817,
   "seeds": [
    0.6031393484617817,
    0.6026377747156546,
    0.6020181854925893
   ],
   "mean": 0.6025984362233419,
   "std": 0.00045855733091596337
  }
 ],
 "usage": {
  "calls": 23,
  "tokens_in": 370499,
  "tokens_out": 63719,
  "cache_read": 236071,
  "cache_write": 0,
  "cost_usd": 2.7017455
 },
 "wall_clock_s": 967.4,
 "interventions": 0,
 "k": 3,
 "eps": 0.002,
 "n_converge": 3,
 "iteration_unit": "node",
 "iterations_used": 7
}
```

## Iterations

### n=0 — node_000 (reproduce_baseline, parent None)
**Hypothesis:** Reproduce the official FM baseline under the harness contract.
**Method:** official FM · target `None` · expected Δ 0.0 (published valid primary 0.6016)
**Result:** GAUC 0.6671 · nDCG@5 0.5358 · primary 0.6015
**Diff:** `None` (None changed lines) · duration 15s · tokens in/out 0/0 · intervention: False

### n=1 — node_001 (improve, parent 0)
**Hypothesis:** Replace pointwise logloss with within-user BPR so training directly improves positive-negative ordering and avoids spending capacity on user-level calibration that cannot affect either metric.
**Method:** loss-bpr-pairwise-within-user · target `loss` · expected Δ 0.006 (The menu identifies ranking-aligned loss as the organizers' lead #1 with an expected gain of 0.002–0.010, and GAUC is exactly the fraction of correctly ordered within-user positive-negative pairs.)
**Result:** GAUC 0.6703 · nDCG@5 0.5370 · primary 0.6036 · realized Δ +0.0022 · ACCEPTED
**Diff:** `diffs/001.patch` (433 changed lines) · duration 14s · tokens in/out 32255/5356 · intervention: False

### n=2 — node_002 (improve, parent 0)
**Hypothesis:** Apply exponential recency weights with a seven-day half-life so the FM learns more from the late-training regime that most closely resembles validation and test.
**Method:** data-weighting-recency · target `data-weighting` · expected Δ 0.004 (Data fact §5 shows that early high-volume days dominate training while positive rates drift toward validation, and the menu estimates a 0.001–0.006 gain from recency weighting.)
**Result:** GAUC 0.6678 · nDCG@5 0.5360 · primary 0.6019 · realized Δ +0.0005 · rejected · grey-zone confirmation {'node_seeds': [0.6019367741369501, 0.6020871274532423, 0.6013703133923629], 'champion_seeds': [0.6014687563529677, 0.6017609746263709, 0.6010899806390073], 'delta_mean': 0.00036}
**Diff:** `diffs/002.patch` (452 changed lines) · duration 18s · tokens in/out 47873/7776 · intervention: False

### n=3 — node_003 (improve, parent 0)
**Hypothesis:** Encode duration_ms=0 in a dedicated category so the model can consistently rank these always-negative impressions below genuinely short videos.
**Method:** features-duration-unknown-flag · target `features` · expected Δ 0.003 (Data fact §3 establishes that duration-zero rows are always negative but the baseline mixes them with a short-video bucket having roughly 0.28 positive rate; the menu estimates a 0.001–0.004 gain.)
**Result:** GAUC 0.6671 · nDCG@5 0.5352 · primary 0.6012 · realized Δ -0.0003 · rejected
**Diff:** `diffs/003.patch` (275 changed lines) · duration 20s · tokens in/out 31801/3496 · intervention: False

#### generation 1 closed — improved; streak 0; champion node_001; best 0.6036; tokens 153708/18058; 277s
_Diagnosis:_ Training dynamics: mild overfit; validation peaks at epoch 7 (0.6015) then declines steadily to 0.5990 while training continues.
Evidence points first to the pointwise loss/training objective: evaluation is strictly within-user ranking, so BPR/ListNet could avoid wasting capacity on user-level calibration.
Data also supports recency weighting (early high-volume days dominate despite drift) and an explicit duration-unknown category (duration_ms=0 is always negative).
Last generation taught only that the official FM is faithfully reproduced; no new mechanism was tested, so none can yet be credited or rejected.
Validation-overfitting risk is currently low (zero non-improving generations), but repeated selection among many small, noisy gains will raise it quickly; require ≥0.002 or multi-seed confirmation.
_Plan for next generation:_ `{"note": "BPR produced the only acceptance and a clear +0.0022 gain; no two distinct target components improved by at least +0.001, so no merge is justified.\nRecency showed a small but seed-consistent +0.00036 on pointwise FM and merits one retest because the champion now uses a materially different pairwise loss; leave two slots to the Selector.", "plan": [{"type": "retest", "parent": "champion", "card": "data-weighting-recency", "hypothesis": "Applying seven-day exponential recency weighting to within-user BPR pairs may combine metric-aligned ordering with adaptation to the late-training regime.", "reason": "The original pointwise test was below acceptance but positive across three paired seeds; replacing logloss with BPR changes how row weights affect sampled positive-negative comparisons, so the prior evidence does not settle performance on the champion stack."}]}`

### n=4 — node_004 (retest, parent 1)
**Hypothesis:** Apply seven-day exponential recency weighting to within-user BPR pairs so metric-aligned learning emphasizes the late-training regime most similar to validation.
**Method:** data-weighting-recency · target `data-weighting` · expected Δ 0.001 (The pointwise three-seed test was consistently positive by +0.00036, and the menu notes that pairwise weighting can interact differently with measured temporal drift.)
**Result:** GAUC 0.6707 · nDCG@5 0.5372 · primary 0.6039 · realized Δ +0.0003 · rejected · grey-zone confirmation {'node_seeds': [0.6039248002601583, 0.6029761772835946, 0.6026772156077824], 'champion_seeds': [0.6036463694691105, 0.6031184279276939, 0.6027235946155616], 'delta_mean': 3e-05}
**Diff:** `diffs/004.patch` (127 changed lines) · duration 12s · tokens in/out 67125/12840 · intervention: False

### n=5 — node_005 (improve, parent 1)
**Hypothesis:** Weight the champion's BPR gradients by each pair's current nDCG@5 swap impact so learning concentrates on positive-negative errors near the top of each user's ranking.
**Method:** loss-lambdarank-pairs · target `loss` · expected Δ 0.003 (The LambdaRank menu entry expects +0.001 to +0.006, and the champion's smaller nDCG gain (+0.0012) than GAUC gain (+0.0032) leaves a clear top-ranking weakness to target.)
**Result:** GAUC 0.6699 · nDCG@5 0.5364 · primary 0.6031 · realized Δ -0.0005 · rejected
**Diff:** `diffs/005.patch` (112 changed lines) · duration 18s · tokens in/out 33592/5426 · intervention: False

### n=6 — node_006 (improve, parent 1)
**Hypothesis:** Add leakage-safe historical user rates by author, tab, and duration bucket to BPR so rows are ordered using each user's prior response to similar impressions.
**Method:** history-user-aggregates · target `history` · expected Δ 0.004 (The history menu entry expects +0.001 to +0.006 because users have a median 35 training interactions and these row-varying aggregates provide personalization not captured reliably by sparse embeddings alone.)
**Result:** GAUC 0.6681 · nDCG@5 0.5364 · primary 0.6023 · realized Δ -0.0014 · rejected
**Diff:** `diffs/006.patch` (792 changed lines) · duration 13s · tokens in/out 71905/25618 · intervention: False

#### generation 2 closed — no improvement; streak 1; champion node_001; best 0.6036; tokens 216791/45661; 593s
_Diagnosis:_ Training dynamics: clear overfitting; validation peaks at epoch 8 (0.6036) then falls steadily to 0.6006 by epoch 12.
Evidence points primarily to the loss: within-user BPR matches GAUC’s positive-negative pair ordering and removes useless user-constant calibration.
BPR moved both GAUC (+0.0032) and nDCG@5 (+0.0012) over the reproduced FM, yielding the only accepted gain (+0.0022 primary).
Seven-day recency weighting was directionally positive but negligible (+0.00036 mean over three seeds), despite measured temporal drift; weighting alone is insufficient.
The duration-unknown category did not help (−0.0003), likely because duration-zero rows are only 1.9% and affect few within-user comparisons.
The peaked curve suggests early stopping/schedule or regularization could preserve the ranking gain; future changes should compose with BPR rather than branch from pointwise FM.
Validation-overfitting risk is currently moderate: only one accepted selection and streak 0, but the narrow +0.0022 win and epoch selection warrant multi-seed confirmation.
_Plan for next generation:_ `{"note": "No merge qualifies: recency was seed-confirmed neutral, while LambdaRank and history aggregates both reduced primary.\nUse the changed BPR stack to retest the mechanism-backed duration flag, and explore auxiliary click supervision as a new family.", "plan": [{"type": "retest", "parent": "champion", "card": "features-duration-unknown-flag", "hypothesis": "A dedicated duration-zero category will let BPR directly push always-negative unknown-duration impressions below positives within the same user.", "reason": "The original test was only on pointwise FM and had weak single-seed evidence (-0.0003); the accepted BPR objective changes how this rare but deterministic feature affects ordering."}, {"type": "explore", "parent": 1, "card": "aux-targets-is-click", "hypothesis": "Adding a modest pointwise is_click auxiliary head with shared embeddings will regularize sparse user/item representations while BPR remains the primary ranking objective."}]}`
