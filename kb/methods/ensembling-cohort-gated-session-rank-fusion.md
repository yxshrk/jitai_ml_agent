---
id: ensembling-cohort-gated-session-rank-fusion
family: ensembling
target_component: ensembling
source: live_07:node_019; kb/data/facts.md §10.5 (session fatigue); conditional expert rank fusion
applies_when:
  - a seed-averaged BPR ranker and a matching branch with features-exposure-session are both available
  - session features improve specific legal cohorts but are flat or harmful globally
  - duration_ms and tab are available at scoring time to define the fixed gate
expected_delta: [0.0, 0.0005]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0005 over 2 measurement(s), so the promise is capped at the record; was: the complete base/session two-branch gate measured fresh-seed mean Δ +0.00049 over the
  five-seed BPR champion, but z=1.94; part of that movement may come from the session branch rather than fusion alone
cost: 42 changed lines on the session parent; ten FM-BPR training phases; measured runtime 203 s; numpy only
composes_with: [features-exposure-session, loss-bpr-pairwise-within-user, ensembling-seed-average]
conflicts_with: [ensembling-top5-gated-hybrid-rank-fusion, ensembling-long-duration-slot-specialists]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0005); official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)]
evidence: [live_07:node_019, live_08:node_015]
---
## Claim
Train parallel five-seed base and session-feature BPR ensembles, then substitute the session ensemble's within-user
rank only for impressions with duration above 180 seconds or tab 4, retaining base ranks elsewhere.

## Mechanism (why it moves within-user ranking)
The fixed row-level gate restricts a noisy session-context expert to cohorts where session features previously
improved ordering. Both branch outputs are converted to within-user ranks before substitution, avoiding score-scale
mismatch; a final rank transform restores a deterministic, tie-free ordering.

## How to implement on node_000
1. Add the proven BPR sampler, five-member seed ensemble, normalized ranks, and features-exposure-session fields.
2. Keep the session matrices as `Xtr` and `Xva`; create base matrices with their first five columns.
3. Set `base_dim = sum(dims[:5])`.
4. Define the gate as `(duration_ms > 180000) | (tab == 4)` for valid and score-extra rows.
5. Train five session FMs and five base FMs with seeds `seed+s`, independent samplers, stopping, and checkpoints.
6. Define `gated_ranks`: rank-average each branch, replace base rank by session rank where the gate is true,
   then call `normalized_ranks(..., tiebreak=base_rank)`.
7. Evaluate history from the gated predictions and stop only after both branches become inactive.
8. Apply the identical gate and fusion to `predictions_extra.csv`, rebuilding session state from train only.

## Risks / failure modes
- The measured comparison bundled the session branch with the gate; node_013 alone had fresh-seed Δ +0.00025,
  so the full +0.00049 cannot be attributed solely to fusion.
- The duration/tab gate was selected from validation subgroup results and may not transfer under cohort drift.
- Hard substitution can disturb cross-cohort ordering within a user even when each branch ranks its cohort well.
- Ten models roughly double the five-seed champion's runtime and validation-selected checkpoints.
- Do not carry valid exposure state into score-extra; rebuild each scored split independently from train state.

## Measured
_Verdict:_ never accepted in 2 measurements on 2 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0005); official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend x1 (best Δ -0.0000)
- live_07:node_019 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6046, single-seed Δ +0.0005, seed-mean Δ +0.0005 (z 1.94) — rejected; 42 changed lines
- live_08:node_015 on [official FM + ensembling-seed-average + ensembling-multiseed-heterogeneous-rank-blend] (variant: short-threshold directional gating): primary 0.6041, single-seed Δ -0.0000 — rejected; 32 changed lines
