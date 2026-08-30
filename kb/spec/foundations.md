# Foundations — the task-specific mathematics every role should reason from

These are derivations, not textbook material: what the metric rewards, what therefore cannot help, and how large
the real effects are. Numbers come from `kb/data/facts.md` and the run journals.

## 1. What the metric measures
- Validation: 124,909 rows, ~7 impressions per user. `primary = (GAUC + nDCG@5) / 2`, both computed per user
  from the ORDER of that user's impressions by predicted score.
- GAUC: per user with both labels present, the AUC of their impressions (P[score(pos) > score(neg)]), weighted by
  the user's number of positives; users with a single label class are excluded (valid: 30.3 % no-positive users,
  11.9 % all-positive, 57.8 % mixed — only the mixed 57.8 % count).
- nDCG@5 with binary gains: DCG = Σ_{i≤5} rel_i / log2(i+1) over the top-5 by score, divided by the ideal DCG. Users
  with no positive score 0 AND count, so nDCG@5 is capped near 0.70; the oracle primary is 0.8484 (test 0.8645).
- Ladder: random 0.4757 · item popularity 0.5715 · official FM 0.6016 · oracle 0.8484. All of personalisation over
  popularity is worth ≈ +0.03; realistic single additions are +0.001 to +0.005.

## 2. Invariances — what cannot move the score
- Both metrics are invariant to any strictly monotone transform of a user's scores and to adding ANY quantity that
  is constant within a user. Therefore: user bias, first-order user weights, user-side features (activity level,
  register days, fan counts …) as additive terms, calibration, and the global intercept change nothing. They only
  matter through INTERACTIONS with item/context fields.
- Any feature constant across a user's ~7 impressions is invisible; features that vary within the impression set
  (video, author, tab, duration, position in the day, the user's history with that author/tab) are the levers.
- In the FM `s = b + Σ w_i + Σ_{i<j} <v_i, v_j>` over the 5 fields (user, video, author, tab, duration bucket), the
  user vector acts only through its 4 dot products with the other fields; the 6 item/context pairs are shared by all
  users of that item and carry "popularity in context".

## 3. Loss versus metric
- Pointwise logloss fits P(long_view | row) for every row, including the 30 % of users whose rows are all
  negatives; capacity spent on user-level calibration is wasted for ranking. A pairwise loss within a user
  (BPR: −log σ(s_pos − s_neg) over that user's positive/negative pairs) has gradients that depend only on within-user
  score differences — the same quantity GAUC measures. Measured: +0.0016 (seed-confirmed, t 8.2) on this data.
- Listwise softmax over ~7 items per user is a noisier estimate of the same thing; LambdaRank-style weighting of
  pairs by nDCG change targets the top-5 half of the metric.
- Early stopping is on the validation PRIMARY, not on the loss: the best epoch by loss and by ranking differ.

## 4. Noise, confirmation and the winner's curse
- Seed-to-seed SD of the primary is ≈ 0.0003 (baseline seeds 0, 1, 2 → 0.60147 / 0.60176 / 0.60109). A single-seed
  Δ below ≈ 0.0005 is inside the noise.
- Picking the best of k single-seed branches is biased upward (order statistics): +0.0022 on one seed was +0.0017
  over three; +0.0005/+0.0006/+0.0005 were +0.0000/+0.0001/+0.0002. Hence acceptance uses three FRESH seeds (seed 0,
  the selected screen, is excluded) and a z-test with the seed SD pooled over the run: fresh-seed mean gain ≥ 0.0005
  and z ≥ 3 (two more seeds when 2 ≤ z < 3). The run converges after 3 generations without a ≥ 0.001 cumulative rise
  of the champion's fresh-seed mean (the organizers' ε rescaled to the seed-mean's noise), and the literal rule
  (single-seed best, ε = 0.002, N = 3) is tracked and reported alongside.

## 5. Learning dynamics on this data
- Closed catalogue (0 % unseen videos in validation), median 35 training rows per user: the FM memorises
  user × video pairs; training loss keeps falling while validation primary peaks at epoch 5–7, then declines.
  Remedies act on capacity (k), regularisation (L2, dropout), the schedule (lr decay, weight averaging) or the loss.
- Volume and positive rate drift 10× across the training window; the validation week is closest to the last
  training days — recency weighting is the natural test, and it measured flat here (+0.0003 to +0.0005).
- `tab` dominates the positive rate; short videos (duration < 18 s must be watched in full) are the hard case;
  `duration_ms = 0` rows are always negative.
