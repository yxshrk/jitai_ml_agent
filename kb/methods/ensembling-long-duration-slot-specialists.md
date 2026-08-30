---
id: ensembling-long-duration-slot-specialists
family: ensembling
target_component: ensembling
source: live_06:node_010; kb/data/facts.md §3; within-user rank invariance and conditional expert ensembling
applies_when:
  - a seed-averaged within-user BPR ranker already provides the global ordering
  - duration_ms is available at show time and the >180 s cohort has weak within-user ranking
  - enough same-user positive-negative training pairs exist with both durations above 180 s
expected_delta: [0.0, 0.0003]
expected_delta_basis: measured (ADR-0018): best seed-mean gain +0.0004 over 1 measurement(s), so the promise is capped at the record; was: the exact three-specialist recipe measured fresh-seed mean Δ +0.00027 over a five-seed
  FM-BPR ensemble (seed-0 Δ +0.00042), but z = 0.60 and the gain was not accepted
cost: ~68 changed lines; three additional BPR training phases; measured runtime 64 s versus 45 s (~1.4x); numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, features-fine-duration-and-tab-cross]
conflicts_with: []
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0003)]
evidence: [live_06:node_010, ceiling:oracle]
---
## Claim
Train three BPR specialists only on same-user pairs whose positive and negative rows both exceed 180 seconds, then
use their averaged conditional ranking to permute only the long-video slots of an existing global rank ensemble.

## Mechanism (why it moves within-user ranking)
The specialists learn ordering within the weak long-duration cohort. Slot-preserving fusion assigns the global
ensemble's existing long-row score values according to specialist order, so all short-versus-long rank slots and
all ordering among non-long rows remain unchanged.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user` and `ensembling-seed-average` with five base members.
2. Build `long_tr = duration_ms > 180000`, then grouped long-negative arrays and eligible long-positive indices.
3. Add `reorder_slots(users, base_scores, specialist_scores, selected)` using per-user `np.lexsort`.
4. Train three fresh `FM` instances with seeds `seed+5..seed+7` on uniformly sampled same-user long-only pairs.
5. For each specialist epoch, rank only validation rows above 180 s and evaluate after `reorder_slots`.
6. Early-stop and checkpoint each specialist independently on full validation primary.
7. Average the three specialists' normalized ranks and permute the base ensemble's long-row slots.
8. Repeat the identical long-row selection, specialist ranking, averaging, and slot permutation for score-extra.

## Risks / failure modes
- The measured fresh-seed gain was only +0.00027 with z = 0.60, below the acceptance threshold.
- This card assumes the existing BPR loss and five-seed ensemble; their gains are not attributable to this method.
- Validation-selected specialists can overfit a small cohort, while users with fewer than two long rows cannot move.
- The fixed 180-second threshold may not transfer under temporal cohort drift.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ +0.0003)
- live_06:node_010 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6044, single-seed Δ +0.0004, seed-mean Δ +0.0003 (z 0.6) — rejected; 68 changed lines
- ceiling:oracle on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: BOUNDED <= +0.0016 for the signal family 'ensembling' — facts §11.3 + blend009 calibration: seed averaging +0.0013–0.0016 over a single model (live_06 node_005/007); on the seed-averaged champion 20 seeds 0.6044, two lineages 0.6047, a GBDT member +0.0006 — re-weightings of the same information add ≤ +0.0006 beyond seed averaging (facts §11, kb/data/screens/CEILING.md)
