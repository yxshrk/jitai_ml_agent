# MLE Agent — autonomous research for KuaiRand long-view prediction

*TikTok TechJam 2026, Track 2*

## What we built

We built an autonomous ML research pipeline that turns each iteration into an auditable solution-tree node: a hypothesis, runnable script, diff, learning curve, official metrics, and any recovery event. The loop is **diagnose → select from cited method cards → implement → sigma-gated acceptance → official convergence**. A selector diagnoses the parent curve, chooses one eligible card from `agent/METHODS.md`, cites it, and records a rejected alternative. The proposer implements the smallest coherent whole-file change; the harness owns execution, rollback, and stopping.

Seeds are mechanical: seed 42 explores, promising changes are confirmed at seeds 42/43/44, and the final pass trains a fixed five-seed ensemble. Compact cross-run memory prevents repeated failures; a reflector runs during stagnation and stores a final self-critique without automatically applying it. Recorded autonomous runs had **zero human interventions**.

## How it addresses the problem statement

The system maps directly to Track 2's three tasks. First, every run reproduces the official FM baseline; the designated run reproduced 0.601838 against the published 0.6016. Second, it iterates across architecture, objectives, features, regularization, data weighting, optimization, and ensembling using cited methods on train and validation only. Third, it designates one converged winner for the single final test submission and improves validation primary over baseline.

KuaiRand-Pure runs end to end to the official convergence rule; interventions are counted; and smoke tests, timeouts, output validation, one fixer attempt, and route-around logic handle failures. Each journal contains the required hypothesis, parent diff, GAUC, nDCG@5, primary, and recovery record. Reports include iterations, wall-clock time, tokens, and GPU-hours.

## Results

The progression is concrete: official Pure baseline **0.6016** → compact single model **0.6047 ± 0.0003** over three seeds → a FULLY UNSEEDED agent run reaching **0.60558** (+0.0040 validation-primary over baseline): the agent ran its own two-stage random hyperparameter search inside one iteration (finding non-obvious dials — dropout 0.18, weight decay 9e-5, LR decay x0.57), then designed its own ensemble, training 7 seed variants and validation-selecting 3 combined by per-user rank average. Earlier seeded runs (0.60513) and a val-greedy seed pool (0.60602) are disclosed as development evidence only. On bonus KuaiRand-1K the agent itself scaled the ensemble from five to ten seeds during its run, finishing at **0.63874 (validation)** from a 0.6208 single-model start; on KuaiRand-27K the same recipe reaches **0.67263** on GPU as an out-of-protocol scaling demo.

The auditable ledger contains about **170 measured cells**; the broader requested experiment total is **{MEASURED_EXPERIMENTS}** until reconciled. Three levers survived: **DCN-lite**, **seven-day recency weighting**, and **strong joint regularization plus rapid learning-rate decay**. Sequence modeling was refuted; in the history campaign, its affinity prerequisite scored 0.6035001, only +0.0019001 over baseline, so DIN-lite was correctly gated off. Watch-time objectives also stayed below epsilon: ordinal watch ratio reached 0.6033 and the CWM-style censored auxiliary 0.6022.

The field curve showed “less is more.” At seed 42, the official five-field L0 model led at 0.604335; kitchen-sink L5 fell to 0.601740. With strong regularization, confirmed L0 reached **0.604660 ± 0.000309**, while strong L5 reached only 0.602991 at seed 42 and did not qualify for confirmation. User affinities, sparse user crosses, and co-visitation initialization likewise failed to beat the controlled stack. The useful signal was modest temporal distribution-shift correction, not more identities or capacity.

## Methodology rigor

We use a fixed **seed-42 explore → seeds 42/43/44 confirm** protocol. A win must clear the **0.002 epsilon floor**. Three baseline seeds calibrate sigma; changes at or above two sigma are accepted, grey-zone changes receive one reseed, and regressions are reverted. Convergence is three completed iterations without improvement above 0.002, with a 50-iteration or six-hour backstop.

Leakage prevention is structural: hidden test data is physically absent from the train/validation workspace and available only to the private final-submission step. Final models are retrained on **train only**, following organizer guidance. Transfer was triangulated with full validation, a discounted short late-validation slice, and an evaluation-only random-exposure window. There the ensemble scored 0.381004285 and seed 42 scored 0.381385181; only relative ordering is meaningful because the exposure policy differs from hidden test.

## Autonomy & feasibility

The intervention count is machine-verifiable: every journal record has an `intervention` boolean, each run report aggregates it, and `logs/RUNS.md` records **0 interventions for all runs**. Recorded converged runs took approximately **10–25 minutes** (11.6–23.5 minutes in the run ledger) on CPU. Typical later runs used roughly **30–50k tokens per run**; the designated four-iteration journal records 9,758 input and 12,063 output tokens, or 21,821 total. Cross-run aggregate: **{TOKEN_TOTAL}**. GPU-hours were **0 (CPU-only)**.

`gpt-5.6-sol` serves as selector, proposer, and reflector; `gpt-5.4-mini` is the fixer. The harness owns seeding, timeouts, acceptance, best-node selection, and convergence.

## Dev tools, APIs, libraries, and datasets used

- **Development/runtime:** Python 3.11, `uv`, Git, and the project harness.
- **ML and analysis:** PyTorch, NumPy, Optuna, and Matplotlib.
- **API:** OpenAI Responses API with direct HTTP integration and per-role token metering.
- **Data and evaluation:** KuaiRand-Pure required benchmark, KuaiRand-1K bonus benchmark, and the organizer starter kit's official evaluator and submission schema. No external training data or pretrained weights were used.

## Limitations & what's next

The plateau around 0.6045–0.6055 appears structural at this scale: added fields, capacity, sequences, watch-time auxiliaries, and blends mostly overfit or land inside noise. Next we would test larger KuaiRand variants, recalibrate sigma for different model families, and broaden search beyond greedy improve-best.

The division of labor is plain: the agent is an executor, not a director. A human designed the method-card space, search levers, safety constraints, and acceptance policy. The agent diagnosed runs, selected among those cited choices, wrote and repaired code, executed experiments, rejected unsupported gains, converged, and critiqued itself. Its autonomy is real within that deliberately human-authored research boundary.
