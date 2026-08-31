---
id: ensembling-pseudo-dwell-transition-expert
family: ensembling
target_component: ensembling
source: kb/data/facts.md §3 and §10.1; live_09 node_013
applies_when:
  - impression timestamps, durations, and author IDs are available at scoring time
  - a same-user BPR model is competitive and can serve as an unchanged anchor
  - repeated or consecutive-author exposures may carry short-term behavioral state
expected_delta: [0.0, 0.0000]
expected_delta_basis: measured (ADR-0018): best seed-mean gain -0.0001 over 1 measurement(s), so the promise is capped at the record; was: attribution-calibrated archival prior; the complete independent-expert plus 15% rank-fusion
  mechanism was flat on one seed, so no positive gain is assigned without changed-stack evidence
cost: ~124 changed lines; two independently stopped BPR phases; measured runtime about 28 s; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-field-aware-fm-embeddings]
conflicts_with: [model-first-order-exposure-transition-fm]
status: dead_under [official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0001)]
evidence: [live_09:node_013]
---
## Claim
Train a separate BPR-FM expert with categorical proxies for the previous impression's dwell state and its
same-author transition, then contribute 15% of a tie-free within-user rank blend with an unchanged BPR anchor.

## Mechanism (why it moves within-user ranking)
For each user, the expert compares the gap from the previous impression with that previous video's
`min(duration_ms, 18000)` threshold. It encodes no predecessor, long break, below-threshold gap, or
threshold-reaching gap, crossed with whether the candidate shares the predecessor's author. These row-varying
states can alter within-user order, while the anchor limits damage from the noisy pseudo-label.

## How to implement on node_000
1. First apply `loss-bpr-pairwise-within-user` to create the unchanged anchor branch and sampler.
2. Load `time_ms`; join `author_id` from `video_features_basic.csv`.
3. Add `pseudo_sequence`: stable-sort by user and time, process equal-time rows as one group, and carry
   `(previous_time, previous_duration, previous_author)` per user.
4. Encode state 0 for no predecessor, 1 for gap over 30 minutes, 2 when duration is unknown or
   `gap < min(previous_duration, 18000)`, and 3 otherwise.
5. Add a second categorical value `state * 2 + same_author`; append both fields to the expert matrix.
6. Build validation and extra states from final train state, then advance causally within that scored split.
7. Train the base and expert BPR FMs with seeds `seed` and `seed+1000`, independently stopping on primary.
8. Blend normalized within-user ranks as `0.85*base + 0.15*expert`, using base rank as the final tie-break.
9. Record every executed epoch for both phases, padding stopped phases while retaining raw phase diagnostics.

## Risks / failure modes
- Inter-impression gap is not observed play time, so the inferred dwell class is a noisy behavioral proxy.
- Equal-time groups cannot identify a unique predecessor; the archived implementation advances state using the
  group's last file-order row after all group features are emitted.
- The diff bundles pseudo-state features, a second independently seeded BPR model, and rank fusion; any future gain
  cannot be attributed to pseudo-dwell alone without a matched expert lacking those fields.
- This duplicates BPR training already present in the anchor and may add correlated rather than complementary errors.
- `time_ms` and `duration_ms` must use the same millisecond units, and valid state must never carry into extra scoring.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user x1 (best Δ -0.0001)
- live_09:node_013 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6036, single-seed Δ -0.0001 — rejected; 124 changed lines
