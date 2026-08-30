---
id: loss-session-monotonic-auxiliary-pairs
family: ranking-loss
target_component: loss
source: kb/data/facts.md §10.5 (long-view rate declines with session position and recent-impression density);
  pairwise logistic ranking
applies_when:
  - leakage-safe session position and prior-10-minute density features are already available
  - same-user BPR training supports an additional pair stream
  - duration_ms and tab are available to restrict the heuristic to long-duration or tab-4 impressions
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0023 over 1 measurement(s), so the promise is capped at the record; was: the exact 0.1-weight auxiliary stream reduced single-seed primary by 0.00234 on a five-seed
  FM-BPR ensemble; no fresh-seed or positive evidence supports an attributable gain
cost: 51 changed lines; an extra logistic pair stream per BPR batch; measured runtime 137 s; numpy only
composes_with: [features-exposure-session, loss-bpr-pairwise-within-user, ensembling-seed-average]
conflicts_with: [loss-lambdarank-pairs, loss-approxndcg-soft-ranks, loss-warp-within-user-rank-weighting]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0023)]
evidence: [live_07:node_020]
---
## Claim
Add a weak pseudo-preference loss that ranks the first targeted impression in a session above later targeted
impressions having both greater session position and greater recent-exposure density.

## Mechanism (why it moves within-user ranking)
The auxiliary logistic pairs encode measured attention fatigue as an ordering constraint without using outcomes:
earlier, less-dense rows are pushed above later, denser rows. Pairs are restricted to rows with duration above
180 seconds or tab 4, where the parent session model had shown its strongest subgroup movement.

## How to implement on node_000
1. First add `features-exposure-session` and same-user BPR.
2. Implement `monotonic_session_pairs(rows, session)`: lexsort by user and time and start sessions after gaps above
   1,800,000 ms.
3. Mark rows where `duration_ms > 180000` or `tab == 4`; find the first marked row in each session.
4. Pair it with each strictly later marked row only when both session-position and density buckets increase.
5. Extend `FM.step(Xp, Xn, Xe=None, Xl=None, aux_weight=0.1)`.
6. For auxiliary differences `da = score(Xe) - score(Xl)`, use
   `ga = -0.1 * sigmoid(-da) / len(Xe)` and scatter opposite gradients into early and late features.
7. Seed a separate auxiliary permutation with `seed + 10000 + member`; distribute all auxiliary pairs across the
   ordinary BPR minibatches using `np.linspace` cuts.
8. Preserve ordinary BPR, member-specific early stopping, and normalized-rank seed averaging.

## Risks / failure modes
- Session position is a logging-context correlation, not a guaranteed row-level preference; forcing monotonic
  pseudo-labels can overwhelm genuine relevance, as the measured large loss suggests.
- The implementation sends every auxiliary pair through the optimizer each epoch; weight 0.1 is not necessarily
  weak after accounting for total pair count.
- The archived node inherited exposure-session fields, BPR, and five-seed averaging; none of those mechanisms is
  attributable to this card, and the auxiliary edit itself performed worse than both its parent and champion.
- Selection of long-duration and tab-4 cohorts came from validation subgroup movement and may overfit that split.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0023)
- live_07:node_020 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6018, single-seed Δ -0.0023 — rejected; 51 changed lines
