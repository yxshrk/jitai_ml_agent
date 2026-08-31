# MLE Agent — autonomous research for KuaiRand long-view prediction

*TikTok TechJam 2026, Track 2*

## What we built

We built an autonomous ML research pipeline that turns each iteration into an auditable solution-tree node: a hypothesis, runnable script, diff, learning curve, official metrics, and any recovery event. The loop is **diagnose → select from cited method cards → implement → sigma-gated acceptance → official convergence**. A selector diagnoses the parent curve, chooses one eligible card from `agent/METHODS.md`, cites it, and records a rejected alternative. The proposer implements the smallest coherent whole-file change; the harness owns execution, rollback, and stopping.

Seeds are mechanical: seed 42 explores, promising changes are confirmed at seeds 42/43/44, and closes are agent-designed — the designated run trained 7 ensemble members and validation-selected 3 (seeds 42-44, per-user rank average). Compact cross-run memory prevents repeated failures; a reflector runs during stagnation and stores a final self-critique without automatically applying it. Recorded autonomous runs had **zero human interventions**.

## How it addresses the problem statement

The system maps directly to Track 2's three tasks. First, every run reproduces the official FM baseline; the designated run reproduced 0.601838 against the published 0.6016. Second, it iterates across architecture, objectives, features, regularization, data weighting, optimization, and ensembling using cited methods on train and validation only. Third, it designates one converged winner for the single final test submission and improves validation primary over baseline.

KuaiRand-Pure runs end to end to the official convergence rule; interventions are counted; and smoke tests, timeouts, output validation, one fixer attempt, and route-around logic handle failures. Each journal contains the required hypothesis, parent diff, GAUC, nDCG@5, primary, and recovery record. Reports include iterations, wall-clock time, tokens, and GPU-hours.

## Results

The progression is concrete: official Pure baseline **0.6016** → compact single model **0.6047 ± 0.0003** over three seeds → a FULLY UNSEEDED agent run reaching **0.60558** (+0.0040 validation-primary over baseline): the agent ran its own two-stage random hyperparameter search inside one iteration (finding non-obvious dials — dropout 0.18, weight decay 9e-5, LR decay x0.57), then designed its own ensemble, training 7 seed variants and validation-selecting 3 combined by per-user rank average. Earlier seeded runs (0.60513) and a val-greedy seed pool (0.60602) are disclosed as development evidence only. On bonus KuaiRand-1K the agent line kept climbing: a 48-cell factorial run discovered a regime inversion vs Pure (pure logloss, no recency) and designated **0.6524**; a later run then discovered causal session features (gap-since-last-impression, session position, hour and weekday crosses) worth **+0.019**, reaching **0.66892** (triple-audited; row-shuffle invariance confirmed at 0.66517, fresh-seed replications 0.674/0.677). On KuaiRand-27K the same recipe reaches **0.67263** on GPU as an out-of-protocol scaling demo.

The auditable ledger contains **~250 measured cells across 139 completed disclosed runs (snapshot 31 Aug)** (logs/RUNS.md + per-run journals). Three levers survived: **DCN-lite**, **seven-day recency weighting**, and **strong joint regularization plus rapid learning-rate decay**. Sequence modeling was refuted; in the history campaign, its affinity prerequisite scored 0.6035001, only +0.0019001 over baseline, so DIN-lite was correctly gated off. Watch-time objectives also stayed below epsilon: ordinal watch ratio reached 0.6033 and the CWM-style censored auxiliary 0.6022.

The field curve showed “less is more.” At seed 42, the official five-field L0 model led at 0.604335; kitchen-sink L5 fell to 0.601740. With strong regularization, confirmed L0 reached **0.604660 ± 0.000309**, while strong L5 reached only 0.602991 at seed 42 and did not qualify for confirmation. User affinities, sparse user crosses, and co-visitation initialization likewise failed to beat the controlled stack. The useful signal was modest temporal distribution-shift correction, not more identities or capacity.

## Methodology rigor

We use a fixed **seed-42 explore → seeds 42/43/44 confirm** protocol. A win must clear the **0.002 epsilon floor**. Three baseline seeds calibrate sigma; changes at or above two sigma are accepted, grey-zone changes receive one reseed, and regressions are reverted. Convergence is three completed iterations without improvement above 0.002, with a 50-iteration or six-hour backstop.

Leakage prevention is structural: hidden test data is physically absent from the train/validation workspace and available only to the private final-submission step. Final models are retrained on **train only**, following organizer guidance. Transfer was triangulated with full validation, a discounted short late-validation slice, and an evaluation-only random-exposure window. There the ensemble scored 0.381004285 and seed 42 scored 0.381385181; only relative ordering is meaningful because the exposure policy differs from hidden test.

## Autonomy & feasibility

The intervention count is machine-verifiable: every journal record has an `intervention` boolean and each run report aggregates it. The designated runs record **0 interventions** (official definition: only behavior-changing actions count); runs that were tainted by mid-run knowledge edits were discarded and disclosed rather than argued. Recorded converged runs took approximately **10–25 minutes** on CPU for Pure. The designated Pure run (bigclock_07, six iterations) used **115,315 tokens and 17.0 minutes wall-clock**; the designated 1K run (omega_1k, eight iterations) used **320,048 tokens and 344.8 minutes** with ~5.7 RTX-4090 GPU-hours (wall-clock is the scored measure; the Pure run used no GPU). Cross-run aggregate: **≈9.9M tokens, ≈140 run-hours** across 139 completed runs.

`gpt-5.6-sol` serves as selector, proposer, and reflector; `gpt-5.4-mini` is the fixer. The harness owns seeding, timeouts, acceptance, best-node selection, and convergence.

## Dev tools, APIs, libraries, and datasets used

- **Development/runtime:** Python 3.11, `uv`, Git, and the project harness.
- **ML and analysis:** PyTorch, NumPy, Optuna, and Matplotlib.
- **API:** OpenAI Responses API with direct HTTP integration and per-role token metering.
- **Data and evaluation:** KuaiRand-Pure required benchmark, KuaiRand-1K bonus benchmark, and the organizer starter kit's official evaluator and submission schema. No external training data or pretrained weights were used.

## Limitations & what's next

The plateau around 0.6045–0.6055 appears structural at this scale: added fields, capacity, sequences, watch-time auxiliaries, and blends mostly overfit or land inside noise. Next we would test larger KuaiRand variants, recalibrate sigma for different model families, and broaden search beyond greedy improve-best.

The division of labor is plain: the agent is an executor, not a director. A human designed the method-card space, search levers, safety constraints, and acceptance policy. The agent diagnosed runs, selected among those cited choices, wrote and repaired code, executed experiments, rejected unsupported gains, converged, and critiqued itself. Its autonomy is real within that deliberately human-authored research boundary.


## The knowledge loop (added 31 Aug)

The system is not one run but a research loop across runs: an unseeded run invented a
temporal pair-sampling kernel (untried speculative card, realized +0.0014 — double its
predicted gain); that measurement was distilled back into the method library; later
clean runs draw on it as cited evidence. The same loop worked on 1K: one run's causal
session-feature discovery (0.66892) was audited three ways (independent
tie-aware re-evaluation exact match; fresh-seed replication 0.674/0.677; within-hour
row-shuffle invariance confirmed, 0.66517), carded, and measured on Pure by a
subsequent run (+0.0002 — an honest negative: Pure's sessions are too sparse, exactly
as predicted by a consulted reviewer). Negative results are first-class: watch-time
objectives, listwise losses, attention, feature crosses, hard-negative BPR, and
post-hoc run blending are all measured dead and disclosed.

## Final designations (frozen 31 Aug)

- Pure: bigclock_07 0.605575 (clean, unseeded, official convergence; no final-wave run exceeded it;
  novel_l1 confirmed at 0.605496 — statistically equal, discussed in README}}
- 1K: omega_1k 0.66892 (faithful A-form replay CSV; recorded run value claimed, replay caveat disclosed in SUBMISSION_RECIPE.md)
