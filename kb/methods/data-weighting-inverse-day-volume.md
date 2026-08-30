---
id: data-weighting-inverse-day-volume
family: data-weighting
target_component: data-weighting
source: kb/data/facts.md §5 (training-day volume varies by roughly 10×); inverse-frequency importance weighting
applies_when:
  - training dates have sharply unequal row counts, so high-volume logging days dominate ordinary empirical risk
  - each training row exposes its date as legal show-time metadata
  - the learner supports per-example weights in its pointwise loss and gradient
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated pointwise-FM probe supplied negative rather than positive evidence, so equal-day
  gradient mass has no attributable expected gain until a materially changed stack demonstrates otherwise
cost: ~20 changed lines; measured runtime ~1.5x the official FM; one float32 weight per training row; numpy only
composes_with: [features-duration-unknown-flag, aux-targets-is-click, model-dcn-cross-head,
  regularization-embedding-dropout-l2]
conflicts_with: [data-weighting-recency]
status: dead_under [official FM x1 (best Δ -0.0026)]
evidence: [live_06:node_001]
---
## Claim
Weight each training row inversely to the number of impressions on its date, normalized to mean one, so every
training day contributes equal total mass to pointwise FM logloss.

## Mechanism (why it moves within-user ranking)
Ordinary row-average loss gives high-volume dates proportionally more influence. Inverse daily frequency weighting
instead estimates an equal mixture over dates, changing the learned video, author, tab, duration, and interaction
parameters. Those row-varying terms can alter within-user order even though date itself is not added as a feature.

## How to implement on node_000
1. Read `date` from `train.csv`, shifting the existing tab, duration, and label column indices accordingly.
2. Compute `tr_day` with `np.unique(tr_dates, return_inverse=True)` and `day_counts = np.bincount(tr_day)`.
3. Set `wtr = len(train) / (n_days * day_counts[tr_day])` as float32, giving weights mean one.
4. Change `FM.step(X, y)` to `FM.step(X, y, weight)`.
5. Use `g = (sigmoid(z) - y) * weight / B` before the existing FM gradient scatter and Adam update.
6. Report the weighted mean logloss and pass the correspondingly shuffled `wtr` slice in every minibatch.
7. Preserve the model, feature encoding, validation-primary early stopping, and prediction path.

## Risks / failure modes
- Equal-day weighting strongly amplifies low-volume dates and can increase variance or overfit logging anomalies.
- Unequal volume need not imply biased conditional preferences; discarding the natural row distribution may hurt.
- This is not exponential recency weighting: early and late dates receive equal aggregate mass regardless of age.
- Combining it with another date weighting rule obscures which target distribution the optimizer represents.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM x1 (best Δ -0.0026)
- live_06:node_001 on [official FM]: primary 0.5989, single-seed Δ -0.0026 — rejected; 20 changed lines
