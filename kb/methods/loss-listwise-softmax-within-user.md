---
id: loss-listwise-softmax-within-user
family: ranking-loss
target_component: loss
source: kb/literature/losses/cao2007_listnet.pdf (ListNet, top-one probability); organizer README "unexplored" #1
applies_when:
  - the metric is within-user ranking (task.md) and users have several rows each (facts: train mean 43.5 rows/user)
  - users have mixed labels (facts §7: 92.7 % of train users are discriminative)
expected_delta: [0.001, 0.008]
expected_delta_basis: organizers list listwise next to pairwise as lead #1; ListNet optimises the whole per-user
  ordering, closest in spirit to nDCG, but plain softmax is not top-heavy — expect BPR-like gains, not more
cost: ~70 lines (per-user grouping + softmax gradient); runtime ~1x if segment ops are vectorised; numpy only
composes_with: [features-duration-unknown-flag, data-weighting-recency, aux-targets-is-click, model-dcn-cross-head]
conflicts_with: [loss-bpr-pairwise-within-user, loss-lambdarank-pairs]
status: untried
evidence: []
---
## Claim
For each user, turn the row scores into a probability distribution with a softmax and minimise the cross-entropy
against the label distribution (positives share the mass); the model learns to put the user's positives on top.

## Mechanism (why it moves within-user ranking)
The softmax over one user's rows is invariant to any user-constant term (task.md), so — like BPR — all capacity goes
to ordering. Unlike BPR it looks at the whole list at once, which matches nDCG's list-level view; ListNet shows the
top-one probability loss is a proper surrogate for the permutation-level objective.

## How to implement on node_000
1. Sort train rows by user once; keep `starts` of each user's block (np.unique with return_index).
2. Per epoch, iterate over batches of *users* (e.g. 400 users per batch, all their rows): compute logits z for the
   rows, subtract the per-user max, exponentiate, normalise within user with `np.add.reduceat`.
3. Target t = y / (sum of y in the user's block); users with zero positives contribute nothing (skip).
4. Gradient d(loss)/dz = P − t; feed it into the existing `np.add.at` gradient accumulation exactly as `g` is now.
5. Optional temperature tau (divide z by tau, try 0.5–2) and a small logloss term to avoid exact ties.

## Risks / failure modes
- Users with hundreds of rows dominate the gradient — normalise each user's contribution to 1.
- Per-user Python loops are too slow: use sorted blocks + reduceat, never a loop over users inside the epoch.
- Plain softmax rewards ordering deep in the list too; nDCG@5 only cares about the top — see loss-lambdarank-pairs.

## Measured
_Verdict:_ no measurement yet

