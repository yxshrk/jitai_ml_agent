# PRACTICES — every dimension, options tested, winner, evidence

Every choice in the final system is backed by a logged experiment with seeds.
Evidence links: zoo/EXPERIMENTS*.md (E-numbers), logs/RUNS.md, logs/run_ab_*.
Protocol: seed 42 explore; wins confirmed at seeds 42/43/44 (mean±std) vs floor
ε=0.002 over baseline 0.6016. ~170 measured cells total (see DASHBOARD.md).

## The frozen model stack (candidate for official runs)
DCN-lite (1 cross layer + MLP128) · official 5 fields ONLY · hybrid 0.5 within-user
BPR + 0.5 logloss (1 uniform negative/positive) · recency weighting (7-day half-life)
· strong regularization (dropout 0.3 + wd 1e-3 family, per ablate_fields --regularized)
· AdamW · selection on PRIMARY at half-epoch · NO auxiliaries.
**Confirmed: 0.6047 ± 0.0003 (ablation) / 0.6048 ± 0.0005 (dims re-confirmation).**

| dimension | options measured | winner | margin / note | evidence |
|---|---|---|---|---|
| architecture | FM, DCN-lite(1-3 cross), DeepFM-ish, FinalMLP, k∈{8,16,32} | DCN-lite, 1 cross, k16 | +0.0025 over FM; FinalMLP failed 3-seed confirm; capacity↑ overfits | E, FINAL |
| input fields | ablation curve L0→L5 kitchen sink | **L0 (official 5 only)** | more fields LOSE (L5 −0.002 vs L0-strong); regularization > information | ABLATION |
| loss | logloss, BPR, hybrid {0/.3/.5/.7/1}, listwise, ΔnDCG-lambda | 0.5/0.5 hybrid | pure either ≈ −0.001; listwise & lambda hurt | E5, AUDIT |
| BPR sampling | 1/3/5 negs, popularity^0.75, hard negatives | 1 uniform negative | others ≤ control and up to 4× slower | DIMS |
| auxiliaries | none, click, like, follow, comment, forward, play-frac; combos; w∈{.1,.2,.3} | **none** | best combo (click+like .1) still < control | DIMS |
| optimizer | Adam, AdamW, Adagrad{.01,.05}, SGD+mom, split emb/MLP | AdamW | Adagrad ties (1 seed); SGD unusable | DIMS |
| regularization | single-dose dropout/wd (dead) → joint grid | strong combo + rapid step LR decay | only schedule family beating plateau | SWEEP, ABLATION |
| data weighting | uniform, recency half-life {3,7,14} | 7-day recency | +0.0027 ± 0.0012 confirmed | HIST E3 |
| features (item/user/context) | affinities, user-stat crosses, metadata crosses, content, aggregates, session, freshness | none survive | all ≤ ε or hurt; session +0.0003 (noise) | HIST, AUDIT, FINAL |
| duration handling | 10 vs 50 buckets, ≤18s flag, regime heads | buckets in L0 dur_bucket10 | 50-bucket variants in-noise; regime heads HURT | E3, AUDIT |
| ensembles | seed-ens, arch-ens, cross-run rank-avg, LGBM blend, specialist gating | seed ensemble only (final step) | variance reducer ~best-single; others ≤ noise or hurt | E7/E8, SWEEP, analyzer |
| embeddings init | random, co-visitation SVD | random | SVD flat | HIST E4 |
| selection metric | GAUC/epoch vs PRIMARY/half-epoch | PRIMARY at half-epoch | organizer metric; part of all confirmed configs | AUDIT |

## Agent-design dimensions (A/B'd on live runs)
| dimension | options | winner | evidence |
|---|---|---|---|
| proposer context | compact journal vs full history (~3×) tokens | **compact**: 0.6034 @ 12.9k tok vs 0.6018 @ 39.6k | run_ab_compact vs run_ab_full2 |
| proposal format | whole-file vs diffs | whole-file (research + zero parse failures across all runs) | agent-design.md; journals |
| search policy | greedy improve-best + forced branch | kept (all runs converge cleanly) | RUNS.md |
| acceptance | raw delta vs σ-calibrated + reseed grey zone | σ-calibrated (killed FinalMLP false positive) | FINAL |
| method selection | freeform proposals vs diagnose→select from card library | selector (built; dress rehearsal tonight) | agent/METHODS.md |
| stopping | official ε=0.002/N=3 all-iterations | official rule verbatim | loop.py |

## Closed with reasoning (not measured)
LLM-as-ranker (GenRec — no semantics in anonymized IDs; wall-clock cost) ·
evaluate.py zip quirk (contract violation) · log_random (awaiting organizer ruling;
temporal overlap) · val-fitting for final model (organizer: don't — train on train only).

## Honest bottom line
~170 cells, 3 surviving levers (DCN-lite, recency, strong regularization+schedule),
everything else measured dead or in-noise on this dataset. Plateau ≈ 0.6045-0.6055
valid primary appears structural for hackathon-scale methods; the organizer's own
warning about validation overfitting predicts test deltas below validation deltas
for all teams — our conservative acceptance protocol is the defense.
