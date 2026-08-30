---
id: ensembling-seed-average
family: ensembling
target_component: ensembling
source: kb/literature/agent-design-notes.md (MLE-STAR: ensembling +6 points; AIRA: single-seed selection is noisy); scoring.md (seed std 0.0008)
applies_when:
  - a champion exists whose remaining error is partly seed variance (std 0.0008 per seed on this dataset)
  - runtime allows N x training (baseline 15 s x 5 = 75 s) — always true here; use as a CLOSING move once the search has plateaued
expected_delta: [0.001, 0.003]
expected_delta_basis: averaging N seeds removes ~sqrt(N) of the seed noise from the score and slightly improves the
  ordering; with std 0.0008 the ceiling is small but nearly free
cost: ~20 lines (loop over seeds, average logits or within-user ranks); runtime N x; numpy only
composes_with: [loss-bpr-pairwise-within-user, loss-listwise-softmax-within-user, loss-lambdarank-pairs, loss-watchtime-censored, features-duration-unknown-flag, features-fine-duration-and-tab-cross, data-weighting-recency, aux-targets-is-click, history-user-aggregates, model-dcn-cross-head, regularization-embedding-dropout-l2, training-schedule-lr-decay-early-stop]
conflicts_with: []
status: proven — accepted on [official FM], [official FM + loss-bpr-pairwise-within-user]
evidence: [live_02:node_015, live_05:node_005, live_06:node_005, live_06:node_007, live_07:node_005, live_07:node_009, live_07:node_014, live_07:node_024]
---
## Claim
Train the champion's script N = 5 times with seeds seed..seed+4 and average the prediction scores (or average
within-user ranks, which is scale-free); submit the average.

## Mechanism (why it moves within-user ranking)
Each seed's model makes different small ordering mistakes on the same pairs; averaging cancels the uncorrelated
part. This is variance reduction, not new signal — hence the small but reliable gain, and why it belongs at the end.

## How to implement on node_000
1. Wrap training in `for s in range(N): model_s = train(seed + s)`; keep each model's best state.
2. score = mean_s(logits_s) — or, for scale-free averaging, mean of per-user ranks computed with lexsort.
3. SMOKE_EPOCHS must cap every member; `--seed` still controls determinism (members use seed + s).
4. Node ensembles: the same code with the two best *different* scripts is a legitimate merge node (ADR-0009).

## Risks / failure modes
- Runtime N x — fine for FM (75 s), check for heavier heads (DCN 2–3 min each).
- Averaging a good model with a clearly worse one hurts; only ensemble nodes within ~0.002 of each other.

## Measured
_Verdict:_ ACCEPTED 4x (live_06:node_005 on [official FM] Δ +0.0013; live_06:node_007 on [official FM + loss-bpr-pairwise-within-user] Δ +0.0016; live_07:node_005 on [official FM] Δ +0.0013; live_07:node_009 on [official FM + loss-bpr-pairwise-within-user] Δ +0.0013)
- live_02:node_015 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6037, single-seed Δ +0.0006, seed-mean Δ +0.0009 (t 5.35) — rejected; 49 changed lines
- live_05:node_005 on [official FM]: primary 0.6025, single-seed Δ +0.0011, seed-mean Δ +0.0008 (z 2.85) — rejected; 65 changed lines
- live_06:node_005 on [official FM]: primary 0.6029, single-seed Δ +0.0015, seed-mean Δ +0.0013 (z 3.22) — ACCEPTED; 68 changed lines
- live_06:node_007 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6040, single-seed Δ +0.0011, seed-mean Δ +0.0016 (z 3.8) — ACCEPTED; 72 changed lines
- live_07:node_005 on [official FM]: primary 0.6029, single-seed Δ +0.0015, seed-mean Δ +0.0013 (z 4.17) — ACCEPTED; 60 changed lines
- live_07:node_009 on [official FM + loss-bpr-pairwise-within-user]: primary 0.6041, single-seed Δ +0.0010, seed-mean Δ +0.0013 (z 4.7) — ACCEPTED; 63 changed lines
- live_07:node_014 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6041, single-seed Δ -0.0000 — rejected; 5 changed lines
- live_07:node_024 on [official FM + loss-bpr-pairwise-within-user + ensembling-seed-average]: primary 0.6040, single-seed Δ -0.0001 — rejected; 2 changed lines
