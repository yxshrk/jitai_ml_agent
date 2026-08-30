# Project 2 branch guide

This is the shared explanation of work on `codex/project-2-kuairand-agent`.
It is intentionally limited to this branch. The adjacent `mle-agent/`
directory is a separate Git worktree and none of its implementation was copied
into this branch.

## Goal and evaluation contract

The task is to rank each user's logged impressions by `long_view`. The primary
validation metric is the mean of **GAUC** and **nDCG@5**, using the organizer's
unchanged `kuairand-starter-kit/evaluate.py`.

- Training window: 2022-04-08 through 2022-04-21.
- Validation window: 2022-04-22 through 2022-04-28.
- Test labels are not used to choose research changes. The original organizer
  baseline can print public test metrics, but all experiments below select only
  on validation.
- The official FM reference is approximately **0.6016** validation primary;
  our seed-0 reproduction was **0.60150**.

## Current status

The strongest independently developed architecture is the corrected causal
**Sequence DeepFM**.

| result | GAUC | nDCG@5 | primary | interpretation |
|---|---:|---:|---:|---|
| Sequence DeepFM, seed 5 | 0.671280 | 0.538190 | **0.604735** | Best observed single-model validation result |
| Sequence DeepFM, seed 6 | 0.671516 | 0.537671 | **0.604594** | Fixed-seed confirmation |
| Sequence DeepFM, seed 7 | 0.671093 | 0.538018 | **0.604555** | Fixed-seed confirmation |
| Sequence DeepFM, seed 8 | 0.671206 | 0.537896 | **0.604551** | Fixed-seed confirmation |
| Sequence DeepFM, seeds 5–8 mean | — | — | **0.604609** | Current credible estimate; no DeepFM ensemble has been selected yet |
| Contextual FM mean-logit ensemble, seeds 0–2 | 0.668811 | 0.536968 | 0.602890 | Earlier stable leader |
| 70% contextual-FM + 30% contextual-BPR, seed 3 | 0.669634 | 0.537047 | 0.603341 | Best non-neural blend, but validation-selected and provisional |
| Same fixed blend, BPR seed 4 | 0.669030 | 0.536838 | 0.602934 | Confirmation; two-seed blend mean 0.603137 |

The next sensible action is a predeclared multi-seed DeepFM ensemble, followed
by a fresh-seed confirmation. It has **not** been run yet.

## Architecture used by the leading Sequence DeepFM

Source: `kuairand-starter-kit/sequence_deepfm.py`.

Inputs are categorical user ID, video ID, author ID, tab, duration-decile,
hour, weekday, and randomized-exposure flag. Every field has its own embedding
table. The score combines:

1. a first-order linear term;
2. an FM second-order interaction term over the eight field embeddings;
3. a two-layer MLP (`Linear → ReLU → Dropout(0.1) → Linear`); and
4. a causal, mean-pooled representation of up to eight previously seen authors
   for the user, plus a learned history-to-current-video/author match.

The model is trained with binary cross-entropy on `long_view`. Histories are
causal: training rows at the same timestamp cannot see one another's outcome,
and validation rows receive only the user's completed training-window history.
No validation label becomes a model feature.

The key implementation correction was initializing embedding weights with
standard deviation 0.01, matching the organizer FM. PyTorch's default
unit-scale embedding initialization made the FM interaction logits explode;
the uncorrected run reached only 0.53871, while the corrected model reached
0.6047.

## How iterations are coordinated

This branch uses a disciplined research loop, not a claim of fully autonomous
agent orchestration:

1. Reproduce the official baseline and lock the evaluator/split contract.
2. State one testable modelling hypothesis.
3. Train only on the fixed training window and evaluate each epoch on the fixed
   validation window.
4. Keep the best validation checkpoint with patience-based early stopping.
5. Record the full learning curve, configuration, and conclusion in `runs/`.
6. Reject regressions rather than stacking them into the leader.
7. Use a new seed or a fixed-weight confirmation before promoting a small gain.
8. Commit coherent batches of code and artifacts so the decision trail is
   inspectable.

Each experiment has a dedicated JSON/JSONL record. The results table is also
maintained in `runs/RESULTS.md`.

## What has been tried

| family | best validation primary | conclusion |
|---|---:|---|
| Organizer NumPy FM, seed 0 | 0.60150 | Reference reproduction |
| Context FM: hour + weekday + is_rand | 0.60246 / 0.60253 / 0.60209 (seeds 0–2) | Keep; mean-logit ensemble 0.60289 |
| Context FM, hour + is_rand | 0.60150 | Reject |
| Context FM, weekday only | 0.60168 | Too small alone |
| Context FM, hour + weekday | 0.60143 | Reject |
| Context FM + video age | 0.60122 | Reject |
| Warm-start BPR FM, base features | 0.60139 | Reject |
| Context FM + click multitask head | 0.60189 | Reject |
| Causal history-rate blend | 0.60246 | Best blend weight was zero; no added value |
| Causal sequence FM | 0.60162 | Reject |
| Positive-class weighted context FM | 0.60181 | Reject |
| Field-dropout/LR-decay context FM | 0.59751 | Over-regularized |
| Context FM with 32-dimensional embeddings | 0.60221 | Extra capacity overfit |
| Explicit wide context crosses | 0.59973 | Overfit |
| Per-user-rank Context-FM ensemble | 0.60276 | Below mean-logit aggregation |
| Contextual BPR refinement, seed 3 | 0.60290 | Near tie alone; useful diverse signal |
| Contextual BPR blend | 0.60334 best observed | Provisional; seed-4 fixed-weight confirmation 0.60293 |
| Sequence DeepFM before init correction | 0.53871 | Diagnosed unstable logit scale |
| Corrected Sequence DeepFM | 0.604735 / 0.604594 | Current leading architecture |
| Four-member Sequence DeepFM ensemble | 0.604483 logits / 0.604559 per-user ranks | Reject; averaging did not add signal |
| Rolling earlier-validation author metadata | 0.603923 (seed 5) | Reject; strictly causal metadata updates were weaker than frozen train history |
| Recency-weighted author pooling | 0.604776 / 0.604661 / 0.604452 / 0.604577 | Reject; four-seed mean 0.604616 is a noise-level tie |
| BCE then within-user pairwise DeepFM | 0.604773 (seed 5) | Reject; direct pairwise refinement was only a near-tie |
| Multi-task DeepFM (click/profile/like/follow) | 0.604308 (seed 5) | Reject; this first shared-head weighting hurt |
| Positive-only causal author history | 0.604659 (seed 5) | Reject; below all-exposure author history |

## Commit checkpoints

| commit | purpose |
|---|---|
| `c76e993` | Recorded causal-history, sequence-FM, and class-weighting negative controls |
| `765a119` | Added contextual ranking refinements and documented rejected capacity/cross/dropout tests |
| `26a0dd3` | Added heterogeneous Context-FM/BPR ensemble scoring and its confirmation artifacts |
| `aea6dc4` / `d8846c6` | Documented and confirmed corrected Sequence DeepFM across four seeds |
| `f157ae4` | Added the bounded API-guided configuration-search agent and its pilot artifacts |
| pending next commit | Record the negative DeepFM ensemble and rolling-metadata controls |

## Useful answers for team questions

**What architecture did it choose?** A DeepFM: categorical embeddings feed a
linear term, FM interactions, a small MLP, and a causal mean-pooled recent
author-history feature.

**How are iterations coordinated?** One hypothesis per run, fixed
train/validation split, checkpoint selection on validation, full run logs,
new-seed checks for small gains, and logical Git commits. This branch's process
is disciplined but not represented as a single fully autonomous harness.

**What did not work?** Naive pairwise loss on the base FM, click multitask
training, hand-built rate histories, class weighting, heavy regularization,
larger FM embeddings, and explicit wide feature crosses. Their exact scores
are above and in `runs/RESULTS.md`.
