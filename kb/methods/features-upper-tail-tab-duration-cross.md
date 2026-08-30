---
id: features-upper-tail-tab-duration-cross
family: features
target_component: features
source: kb/data/facts.md §3–4 (duration mechanism and tab effects); live_06:node_012
applies_when:
  - duration_ms and tab are legal show-time features
  - videos above 180 seconds are a weak ranking cohort
  - the model accepts additional categorical fields
expected_delta: [0.000, 0.000]
expected_delta_basis: the isolated probe lost 0.00061 primary on the five-seed FM-BPR ensemble and received no
  fresh-seed confirmation, so no positive attributable gain is supported
cost: 10 changed lines; one small categorical field; runtime remained approximately 1x parent; numpy only
composes_with: [loss-bpr-pairwise-within-user, ensembling-seed-average, model-field-aware-fm-embeddings, model-dcn-cross-head]
conflicts_with: [features-fine-duration-and-tab-cross]
status: dead_under [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0006)]
evidence: [live_06:node_012]
---
## Claim
Append a tab-crossed duration-tail field with levels at 180, 240, 360, and 600 seconds, preserving one shared
duration level at or below 180 seconds while refining the weak upper tail.

## Mechanism (why it moves within-user ranking)
The crossed value varies across a user's rows and can therefore change both metrics. It gives the FM direct
parameters for tab-specific upper-tail duration regimes that would otherwise be represented only through the
existing tab-duration embedding interaction.

## How to implement on node_000
1. Append `tab_dur_tail` to `FIELDS`.
2. Define `tail_edges = np.array([180_000, 240_000, 360_000, 600_000], dtype=np.float64)`.
3. In `raw`, compute `tail = int(np.searchsorted(tail_edges, float(dur), side='left'))`.
4. Append `f'{tab}|{tail}'` to the returned categorical values.
5. Let the existing vocabulary construction, offset encoding, FM training, checkpointing, and scoring paths consume
   the sixth field unchanged.

## Risks / failure modes
- The field duplicates information already available through tab and the ordinary duration bucket, increasing
  variance without adding new observations.
- `side='left'` places exactly 180 seconds in the lower bucket; changing boundary semantics creates a different method.
- The negative measurement was on a five-seed FM-BPR ensemble; gains from BPR and seed averaging are not
  attributable to this feature.
- Do not combine with the generic fine-duration/tab-cross card without an ablation because their encodings overlap.

## Measured
_Verdict:_ never accepted in 1 measurements on 1 stack(s); official FM + loss-bpr-pairwise-within-user + ensembling-seed-average x1 (best Δ -0.0006)
- live_06:node_012 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6033, single-seed Δ -0.0006 — rejected; 10 changed lines
