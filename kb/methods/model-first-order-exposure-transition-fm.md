---
id: model-first-order-exposure-transition-fm
family: sequential-model
target_component: model
source: Rendle, Freudenthaler, and Schmidt-Thieme, "Factorizing Personalized Markov Chains for Next-Basket
  Recommendation," WWW 2010 (https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf) ([archives.iw3c2.org](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf?utm_source=openai))
applies_when:
  - `time_ms` permits each user's impressions to be ordered without using outcomes (task specification)
  - sequence effects are measured (facts §10.2: consecutive same-author rows have 0.142 positive rate versus 0.337)
  - current static FM scores cannot distinguish exposure context, while 5.7% of valid rows involve repeated pairs
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0005 over 1 measurement(s), so the promise is capped at the record; was: factorized transitions add genuinely new row-varying information, but logged exposures are
  not chosen next items and individual video transitions are sparse; expect at most a small acceptance-scale gain
cost: ~40 changed lines; two video-sized embedding tables and one extra dot product, runtime ~1.2x; numpy only
composes_with: [loss-bpr-pairwise-within-user, history-same-author-run-features, regularization-embedding-dropout-l2]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)]
evidence: [live_07:node_012, ceiling:oracle]
---
## Claim
Add an FPMC-style latent transition score between the user's immediately previous exposed video and the current
video, using only strictly earlier impression features.

## Mechanism (why it moves within-user ranking)
The transition term `dot(P[previous_video], Q[current_video])` varies across a user's rows and therefore can change
both metrics. Unlike a same-author flag, factorization can learn graded continuation or fatigue patterns shared
across different video pairs while the existing FM retains long-term user preference and context effects.

## How to implement on node_000
1. Map videos to dense indices and reserve one sentinel for users with no strictly earlier exposure.
2. Lexsort rows by `(user_id, time_ms, original_row)` and compute each row's previous video, committing equal-time
   rows together so they cannot become one another's history.
3. For validation, initialize from each user's final training exposure; then advance through earlier validation rows.
4. For score-extra, process train plus feature-only valid history before advancing through extra rows chronologically.
5. Add `P,Q` arrays shaped `(n_video+1,k)` and score `sum(P[prev] * Q[current], axis=1)`.
6. Scatter `g*Q[current]` and `g*P[prev]` in both BPR gradient passes; include their Adam states and checkpoints.

## Risks / failure modes
- Logged exposure order is not a user-selected sequence, so generic video-to-video transitions may mostly be noise.
- Pair-specific observations are sparse despite the closed catalogue; strong L2 on `P,Q` is required.
- Computing a row's context from later or equal-time impressions would leak future exposure information.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0005)
- live_07:node_012 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6036, single-seed Δ -0.0005 — rejected; 89 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0010 for the signal family 'session-context' — facts §11.1: pairs within 10 min are 2 % of the GAUC pair mass; measured +0.0009 on BPR (z 3.1), +0.0002 on the seed blend; attribute continuation from the most recent earlier positive (history-last-positive-attribute-recurrence) +0.0009 on the plain FM (live_07 node_001), additive 0 on node_009 (kb/data/screens/BEHAVIOUR.md) (facts §11, kb/data/screens/CEILING.md)
