# Lever ledger — every idea, its status. Goal: nothing untried by end of 28 Aug.

Legend: DEAD = measured below epsilon (log ref) · ALIVE = measured win kept ·
C1/C2/C3/C4 = running campaign (hist / sweep / audit / final) · PARKED = rules risk,
webinar question · N/A = ruled out with reasoning, not testable.

## Losses & objectives
- pointwise logloss — baseline
- within-user BPR hybrid — ALIVE in best stack (EXPERIMENTS.md E5)
- pure BPR / pure logloss — DEAD (E5)
- per-user listwise softmax (full history) — DEAD (run 04)
- dnDCG lambda, top-5 sized groups — C3
- ordinal watch-ratio aux — DEAD as FM-aux (run 04) / C3-adjacent variants closed
- CWM censored watch-time — DEAD as aux at hackathon dose (run 04/05)
- user-uniform vs positive-weighted specialist mix — C4

## Architectures
- FM (k16) — baseline · k32 DEAD · k8 — C2
- DCN-lite — ALIVE (+0.0025, core of best stack)
- DeepFM/MTL shared-bottom — tied/kept at aux 0.1 (EXPERIMENTS.md)
- FFM (k8), FinalMLP — C4 (formal closure cells)
- xDeepFM/AutoInt — N/A (research: overfit at this scale; consistent with every
  capacity increase measured so far)
- DIN-lite sequence attention — C1 (gated; prerequisite failed, correctly skipped)
- duration-regime two-head — C3
- hierarchical user-embedding shrinkage — C4

## Features
- 50 dur buckets + <=18s flag + dur x tab + hour + dow — ALIVE (in best stack)
- item/author Bayesian aggregates — DEAD (E4)
- video content features — DEAD (E6)
- user-history affinities (author/tab/duration) — DEAD (C1 E1)
- user-stats x item crosses — DEAD (C1 E2)
- user-metadata (coarse) x item crosses — C3
- session features from time_ms — C3
- item freshness (upload age) + causal popularity velocity — C4
- explicit sparse cross IDs w/ frequency backoff — C4
- co-visitation SVD embedding init — C1
- user-constant features alone — N/A (cannot move a within-user metric)

## Training & schedules
- early stop on GAUC — baseline behavior; PRIMARY selection + sub-epoch — C3
  (zoo/common.py still selects on GAUC — fix queued post-campaigns)
- single-dose dropout 0.15 / AdamW 1e-4 — DEAD (runs 02)
- joint dropout x wd x lr-schedule grid, embedding-specific reg — C2
- batch/lr scaling, SWA/EMA — C2
- recency weighting of train days — C1
- Optuna polish of winner — C4

## Ensembles
- seed ensemble — measured: variance reducer to ~best-single-seed (E7)
- diverse-architecture rank ensemble — C2
- specialist gating (GAUC-model + nDCG-model) — C4
- LightGBM lambdarank + blends — DEAD (E8)

## Data & rules-edge
- log_random rows (dates overlap val/test) — PARKED: webinar legality question
- train on train+val for final test model — PARKED: webinar question
- evaluate.py zip-truncation quirk — N/A: rejected as contract violation (audit)
- pseudo-labeling val — N/A: circular on val; no legal test-time use identified
- KuaiRand-1k/27k — bonus benchmarks, separate decision (not a Pure lever)
- LLM-as-ranker (GenRec, Netflix, arXiv:2608.10257) — N/A with reasoning: needs
  verbalizable semantics (titles/text); KuaiRand-Pure is anonymized IDs, and
  LLM-per-impression inference conflicts with scored wall-clock. Cite in Devpost
  as a surveyed-and-rejected direction; its context-engineering emphasis and
  multi-objective reward weighting validate our agent-context design and the
  C4 specialist-gating experiment.

## Harness/agent levers (points, not score)
- learning-curve observability — DONE (28 Aug)
- method-selector call + convergence pressure + stagnation reflector — Codex building
- cross-run memory — partially via MENU measured annotations; full version optional
- final test-submission script (outside agent world) — TODO before 1 Sep
