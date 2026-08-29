# Cross-run memory

## Run logs/run_official_1k_01d
dataset: 1k
stop_reason: converged
best_primary: 0.620817
- node_001 | method: regularization-schedule | hypothesis: The parent is severely overfitting because validation primary | primary: n/a | verdict: failed
- node_002 | method: regularization-schedule | hypothesis: An aggressive package of MLP dropout 0.30, accessed-row | primary: 0.601891 | verdict: rejected
- node_003 | method: session-time-features | hypothesis: Because the parent peaks at epoch 0.5 and | primary: n/a | verdict: failed
self_critique:
Overall assessment

The run did not demonstrate convergence; it found a strong baseline and then failed to execute or validate two broad hypotheses. “Budget exhausted/no viable improvement” would be a more accurate stop characterization than “converged.”

What the harness/policy did suboptimally

- It bundled too many changes. Node_001/002 simultaneously changed dropout, embedding L2, AdamW decay, LR decay, checkpointing, and batch size. The 0.6019 result gives no attribution and cannot identify which change caused the large regression.
- It continued from a failed node. Node_002 inherited an unmeasured node_001 rather than branching from the last successful implementation, increasing debugging and confounding risk.
- Execution reliability was poor: two of three proposals produced no metric. The policy should have required a short smoke test and runtime estimate before committing a full trial.
- The hypotheses overreached the evidence. The curve supports early overfitting, but not specifically that sparse static IDs are the cause. Adding multiple crosses could increase sparsity and worsen exactly that problem.
- Expected gains of 0.0025–0.003 were only modest relative to sigma=0.0015. Near-threshold improvements would require repeated seeds, yet the run spent budget on large packages instead.
- The search did not isolate the most obvious finding: the baseline peaks around epoch 0.5. Finer checkpoint/early-stop tuning was cheaper and better supported than architectural or feature expansion.
- “Validation-GAUC checkpointing” may also create selection noise unless checkpoint frequency and the final primary metric are standardized across trials.

What I would change in the scaffold

- Require atomic experiments: one regularizer, one optimizer change, or one feature family per node.
- Always branch experimental work from the latest successful node; treat failed nodes as debugging records only.
- Add mandatory compile/data checks plus a timed mini-run before full evaluation.
- Record metric curves at fixed training fractions, including train loss, validation primary, and best versus final checkpoint.
- Re-run the baseline across seeds before treating differences near 1–2 sigma as real.
- Require explicit rollback/fallback plans for expensive feature pipelines.
- Separate implementation changes such as batching from scientific changes so throughput fixes do not confound model comparisons.
- Use a stop label that distinguishes true search convergence from repeated execution failure or exhausted budget.

What the next run should try first

1. Reproduce the baseline for several seeds and checkpoint more finely around its apparent optimum, such as 0.25, 0.5, and 0.75 epoch. This verifies whether 0.6208 is stable and whether earlier stopping alone improves it.
2. If overfitting is reproducible, test exactly one low-risk intervention first—preferably modest embedding/ID regularization or reduced embedding capacity—while keeping optimizer, LR schedule, batching, and checkpointing unchanged.
3. Only after an atomic regularization result should causal session-position/gap features be tested, one feature family at a time and without high-cardinality crosses initially.

The highest-value next action is not another feature package; it is establishing a reliable baseline distribution and optimizing the early checkpoint region.

## Run logs/run_official_1k_02d
dataset: 1k
stop_reason: converged
best_primary: 0.620817
- node_001 | method: embedding-dim-down | hypothesis: The validation collapse after epoch 0.5 diagnoses severe | primary: n/a | verdict: failed
- node_002 | method: embedding-dim-down | hypothesis: Reducing the parent model's embedding dimension to k=8 | primary: 0.614862 | verdict: rejected
- node_003 | method: duration-regime-heads | hypothesis: The parent’s validation peak at epoch 0.5 followed | primary: n/a | verdict: failed
self_critique:
Run critique

What was suboptimal
- The run declared convergence after only one baseline and one successfully evaluated alternative. Two of four nodes produced no metric, so this was more an execution-limited run than a converged search.
- The k=8 hypothesis was overly specific and weakly supported. A post-epoch-0.5 validation decline indicates training-time overfitting, but does not uniquely implicate embedding width. The result, 0.6149 versus 0.6208, strongly rejects this particular reduction.
- The proposed gains (+0.002 to +0.003) were close to the reported sigma of 0.0015. Single evaluations cannot reliably distinguish gains of that size.
- node_002 was attached to a failed/no-metric node rather than cleanly represented as a repaired execution of a hypothesis branching from node_000. This obscures lineage and failure accounting.
- The censoring-boundary residual-head idea added architectural complexity before simpler controls—checkpoint timing, weight decay, dropout, and intermediate dimensions—were tested.
- “No metric” failures were not diagnosed or retried, yet apparently contributed to stopping.

Scaffold changes
- Separate scientific rejection from infrastructure failure. A no-metric node should receive an automatic minimal repair/retry and should not count toward convergence.
- Require a minimum number of valid trials or explored hypothesis classes before stopping.
- Use repeated seeds or paired reruns when expected improvements are near validation noise; report uncertainty on deltas.
- Avoid asserting precise expected gains without evidence. Record the diagnosis, intervention, and falsifiable outcome separately.
- Preserve clean lineage: repaired runs should branch from the last valid parent and be labeled as retries.
- Add lightweight sweeps around plausible settings rather than jumping from k=24 directly to k=8.
- Log epoch-level train/validation metrics and checkpoint scores so “overfitting” can be distinguished from optimizer instability or noisy validation.

What the next run should try first
1. Reproduce the baseline across several seeds and verify that the epoch-0.5 peak is stable.
2. Keep k=24 and directly regularize training: tune weight decay and embedding dropout, while retaining early checkpoint selection.
3. Test intermediate embedding sizes, especially k=16, rather than retrying k=8.
4. Tune evaluation/checkpoint cadence around the early peak; the best improvement may come from selecting a more precise early checkpoint rather than changing architecture.
5. Only after these controls, retry the 18-second regime feature as a simple indicator or interaction before introducing a separate residual head.

The immediate best first experiment is a small, paired sweep of baseline k=24 with stronger embedding regularization and finer early checkpointing.

## Run logs/run_desig_full_02
dataset: pure
stop_reason: converged
best_primary: 0.601838
- node_001 | method: regularization-schedule | hypothesis: Because validation primary peaks at epoch 8 and | primary: 0.598423 | verdict: rejected
- node_002 | method: seed-ensemble | hypothesis: Because validation primary peaks at epoch 8 and | primary: 0.602756 | verdict: suspect_implementation
- node_003 | method: seed-ensemble | hypothesis: Replacing noisy per-member validation-selected checkpoints with the champion’s | primary: 0.602661 | verdict: rejected
self_critique:
The run converged prematurely. It explored only one substantive modeling direction—overfitting around epoch 8—and then spent two nodes on an ensemble implementation issue. That is not enough evidence that the baseline FM is locally optimal.

Harness/policy weaknesses:
- node_001 bundled embedding dropout, accessed-row L2, AdamW decay, and LR decay. Its failure is uninterpretable because four interventions changed at once, and “aggressive” regularization was poorly matched to a small observed overfit gap.
- node_002 reported the best score, 0.6028, but was marked SUSPECT_IMPLEMENTATION. The policy then spent another expensive five-seed run on a narrow checkpoint correction rather than first building a minimal equivalence test.
- node_003 still scored 0.6027, above baseline, yet was rejected. The journal does not expose the rejection threshold, uncertainty, or whether ensemble cost is part of acceptance. This makes the decision hard to audit.
- The baseline sigma of 0.0001 appears overconfident unless it came from repeated seeds. A single-seed baseline cannot support reliable judgments about gains of roughly 0.001.
- Per-user rank averaging, checkpoint selection, and “measured seed-ensemble implementation” were not specified rigorously enough before the experiment. The suspect result was therefore avoidable.
- Declaring convergence after four nodes, with only one clean non-baseline experiment, was too aggressive.

Scaffold changes:
- Require one-factor-at-a-time ablations before allowing compound regularization recipes.
- Establish a repeated-seed baseline and report mean, standard error, paired deltas, and compute cost.
- Add deterministic unit checks for ensemble member loading, fixed versus validation-selected epochs, user grouping, rank direction, tie handling, and prediction coverage.
- Separate scientific acceptance from implementation status and resource tradeoffs. A valid but expensive improvement should be recorded as such rather than simply rejected.
- Make stopping require coverage of several independent hypothesis families, not just exhaustion of one branch.
- Prefer cheap diagnostics before full retraining: evaluate saved checkpoints across epochs and test ensemble code on cached predictions.

What the next run should try first:
1. Reproduce node_000 over at least five fixed seeds, saving predictions for epochs around the peak, such as 6–10.
2. Using cached predictions, cleanly compare:
   - single-seed fixed epoch 8,
   - best epoch per seed,
   - raw-score averaging,
   - per-user rank averaging.
3. If the approximately +0.0009 ensemble gain remains statistically credible, retain it as a variance-reduction result and document its inference cost.
4. Then return to single-model improvements with isolated, modest changes: tune latent dimension, learning rate, embedding L2, and dropout separately. Early stopping or checkpoint averaging should be tested before another compound regularization schedule.

The immediate priority is not another model idea; it is obtaining a trustworthy estimate of seed and checkpoint variance so that gains near 0.001 can be distinguished from noise.

## Run logs/run_desig_full_01
dataset: pure
stop_reason: converged
best_primary: 0.601838
- node_001 | method: regularization-schedule | hypothesis: Because validation GAUC peaks around epochs 6-8 and | primary: 0.595355 | verdict: rejected
- node_002 | method: session-time-features | hypothesis: Although the parent’s epoch-8 validation peak followed by | primary: 0.602435 | verdict: rejected
- node_003 | method: seed-ensemble | hypothesis: Because the parent peaks near epoch 8 and | primary: 0.602885 | verdict: suspect_implementation
self_critique:
Overall

The run was too shallow to justify “converged.” It tested only three children, two of which bundled several changes, and then retained the baseline. Reporting best_primary=0.601838 is internally consistent with acceptance status, but the run found potentially useful signal in node_002 that was not adequately followed up.

What the harness/policy did suboptimally

- Premature convergence: three trials are insufficient, especially when node_002 improved raw primary from 0.6018 to 0.6024. That result warranted replication and ablation rather than termination.
- Weak experimental isolation:
  - node_001 simultaneously changed embedding dropout, row-wise L2, AdamW weight decay, and LR decay. Its failure reveals little about which intervention was harmful.
  - node_002 added multiple temporal/session features and crosses together, so the source of its small gain is unknown.
- Hypotheses used unjustifiably precise effect-size promises (“+0.0025,” “at least +0.002,” “approximately +0.004”). These estimates were not grounded in prior measurements and appear to have influenced binary accept/reject decisions unnecessarily.
- Acceptance seems too rigid. A +0.0006 candidate should not be promoted automatically, but it should enter a “promising; replicate” state rather than be discarded.
- Uncertainty handling is unclear. Baseline sigma=0.0001 is implausibly precise unless it came from repeated full training runs or bootstrap analysis. Validation-row bootstrap uncertainty would not capture seed and checkpoint variance.
- node_003 was a poor early allocation of compute. Five-seed ensembling tests variance reduction, not a better underlying model, and complicates fair comparison. The implementation concern should have been resolved with prediction-shape, ordering, per-user ranking, and leakage checks before scoring.
- The overfitting diagnosis was repeatedly asserted from one learning curve, but the scaffold did not first test simpler responses such as checkpoint selection, mild regularization, or one-factor sweeps.

What I would change in the scaffold

- Add trial states: rejected, promising-needs-replication, accepted, and invalid/suspect.
- Require paired multi-seed evaluation for gains near the noise floor, using identical splits and reporting mean, standard deviation, and per-seed deltas.
- Separate hypothesis generation from expected gain; use directional predictions unless a calibrated prior supports a numeric target.
- Enforce one conceptual change per trial. Permit bundles only after components have independently shown value.
- Automatically trigger ablations or replications when a candidate beats the champion numerically but misses the promotion threshold.
- Add minimum exploration requirements before convergence: several distinct hypothesis families and at least one follow-up on every promising result.
- For ensembles, validate prediction alignment, ranking direction, user grouping, and absence of validation-tuned weights; compare both raw-score and rank averaging.
- Track best observed, best valid single model, and best accepted model separately.

What the next run should try first

Replicate node_002 across at least three matched seeds, then ablate its feature groups:

1. Session gap and session position only.
2. Hour/position cross only.
3. Weekday/gap cross only.
4. Best individual group plus the next-best group.

Verify strict causality and feature construction at sequence boundaries. If the mean paired gain remains positive and stable, promote the simplest variant. In parallel, run a small one-factor regularization sweep—mild embedding dropout or weight decay, not the aggressive node_001 bundle—with unchanged optimizer and LR schedule. Defer ensembling until a stronger single model is established and node_003’s implementation issue is diagnosed.

## Run logs/run_desig_1k
dataset: 1k
stop_reason: converged
best_primary: 0.620817
- node_001 | method: regularization-schedule | hypothesis: Because validation peaks at epoch 0.5 and then | primary: n/a | verdict: failed
- node_002 | method: regularization-schedule | hypothesis: Because validation peaks at epoch 0.5 and then | primary: 0.609535 | verdict: rejected
- node_003 | method: session-time-features | hypothesis: Because validation primary peaks at epoch 0.5 and | primary: n/a | verdict: failed
self_critique:
Run critique

What the harness/policy did suboptimally
- It declared convergence after effectively one completed challenger. That is far too little exploration to support a convergence claim.
- node_001 bundled four major changes—dropout, embedding L2, AdamW weight decay, and rapid LR decay. When node_002 regressed from 0.6208 to 0.6095, the run learned almost nothing about which change caused the damage.
- The 0.5-per-epoch LR decay was especially aggressive and insufficiently motivated. It confounded regularization with optimization failure.
- node_002 debugged an experiment that produced no metric, but apparently reran the same hypothesis rather than first isolating the execution failure or simplifying the patch.
- node_003 also failed without a recorded recovery attempt. Feature-generation failures should trigger a minimal repair/debug path, not termination.
- The policy correctly returned to node_000 for node_003, but it spent its limited budget on a multi-feature package rather than cheap, atomic ablations.
- The claimed improvements of 0.003–0.004 are only about 2–3 baseline sigma. No repeated-seed confirmation or uncertainty-aware comparison was attempted.
- “Fine-grained checkpoint selection” risks adapting to validation noise, especially on a 1k dataset. The run does not document a fixed checkpoint rule or an untouched final evaluation split.
- The journal lacks enough diagnostics: exact epoch-wise validation values, train/validation gap, failure traces, parameter counts, category coverage, and seed stability.

What I would change in the scaffold
- Require a minimum exploration budget before allowing “converged,” such as 5–10 successful atomic trials across at least two hypothesis families.
- Enforce one-factor experiments by default. Bundles should require prior evidence for each component.
- Separate build/runtime debugging from scientific evaluation. Preserve the intended experiment, record the traceback, and test the smallest executable repair.
- Add automatic recovery for failed feature pipelines: schema checks, cardinality checks, missing-value handling, and a one-batch smoke test before full training.
- Always branch scientific experiments from the best valid node, while keeping debug descendants separate.
- Record per-epoch metrics and select checkpoints using a fixed rule. Confirm promising changes over several seeds; compare gains against paired uncertainty.
- Use small sweeps for continuous controls rather than one aggressive setting.
- Add guardrails against validation overfitting from repeated checkpoint and hypothesis selection.

What the next run should try first
1. Reproduce node_000 over 3–5 seeds to establish whether 0.6208 is stable.
2. Diagnose the early peak with an atomic learning-rate/early-stopping experiment. Keep the model and regularization unchanged; try a lower initial LR, such as 0.5x and 0.25x, with more frequent checkpoints around the first epoch. This directly tests whether the collapse is optimization overshoot rather than insufficient regularization.
3. If overfitting remains, test one regularizer at a time with mild values: embedding L2 first, then dense weight decay, then dropout. Avoid the 0.5-per-epoch decay bundle.
4. After optimization is stable, retry exactly one robust context feature—preferably session position or log-scaled inter-impression gap—with a smoke-tested preprocessing path. Do not introduce all time/session crosses simultaneously.

The strongest immediate lesson is not that regularization failed; it is that an aggressive, confounded package failed. The run ended before identifying a useful causal direction.

## Run logs/run_desig_full_03
dataset: pure
stop_reason: converged
best_primary: 0.601838
- node_001 | method: none | hypothesis: (proposal failed) | primary: n/a | verdict: failed
- node_002 | method: none | hypothesis: Applying epoch-level exponential moving averaging to the baseline | primary: 0.601358 | verdict: rejected
- node_003 | method: embedding-dim-down | hypothesis: Because validation primary peaks at epoch 8 and | primary: 0.600950 | verdict: rejected
self_critique:
Overall assessment

The run converged prematurely around an already strong baseline. Neither tested modification directly targeted the most plausible failure mode, and one proposal-generation failure wasted a substantial fraction of the search budget.

What the harness/policy did suboptimally

- It explored only two valid alternatives before declaring convergence. That is insufficient evidence that the local design space is exhausted.
- node_001 failed without producing an executable experiment, yet node_002 was nominally descended from that failed node. Failed nodes should not become parents; proposal failures should be retried or repaired without consuming an experimental branch.
- The policy overcommitted to a single interpretation of the learning curve: “capacity-driven overfit.” A post-peak validation decline can also reflect insufficient regularization, learning-rate behavior, checkpoint selection, or ordinary validation noise.
- EMA was poorly matched to the diagnosis. Averaging parameters across later overfit epochs can hurt unless EMA start time and decay are carefully chosen. The observed rejection was therefore not very informative.
- Reducing k directly from 16 to 8 was a coarse capacity change. It confounds reduced overfitting with reduced representational power; intermediate dimensions or explicit regularization would have been more diagnostic.
- Claimed gains of 0.0015–0.002 were not justified by prior evidence. The baseline sigma of 0.0001 also needs scrutiny: if it was not estimated from multiple independent seeds, acceptance and convergence decisions were overconfident.
- The journal lacks seed-level results, best epoch, stopping epoch, runtime, and train/validation curves for each candidate. Without these, it is difficult to distinguish a real regression from noise or altered optimization dynamics.

What I would change in the scaffold

- Require every proposal to compile and pass a smoke test before it consumes a node.
- Permit ancestry only from successful, metric-bearing nodes.
- Retry malformed proposals automatically with the concrete error and configuration schema.
- Separate screening from confirmation: run broad single-seed screens, then rerun promising candidates and the baseline over identical seeds.
- Base acceptance on paired seed differences or confidence intervals, not a possibly synthetic single-run sigma.
- Require hypotheses to specify one controlled change, expected mechanism, and diagnostic outcome.
- Avoid convergence until several distinct intervention classes have been tested, such as regularization, optimization, capacity, and feature/model structure.
- Record full curves and checkpoint behavior so the policy can tell whether a change improved the peak, delayed overfitting, or merely changed stopping dynamics.

What the next run should try first

First run a small embedding-regularization sweep while keeping k=16 and all other settings fixed. Increase L2/weight decay specifically on FM embeddings, for example baseline, 2×, 4×, and 8× the current value. This directly tests the overfitting hypothesis without discarding capacity. Compare candidates using the same seeds and best-checkpoint rule.

If that fails, the next priorities should be:

1. Test intermediate dimensions such as k=12 and possibly k=20, rather than only k=8.
2. Sweep learning rate and/or introduce decay around the observed validation peak.
3. Test embedding dropout if supported.
4. Reassess early-stopping patience and checkpoint restoration, while ensuring selection is not leaking test information.
5. Only revisit EMA with a defined warm start near the pre-overfit region and tuned decay.

The most important immediate improvement is not a more exotic model; it is a reliable, seed-paired local search around regularization and optimization, with proposal failures repaired rather than counted as exploration.

## Run logs/run_desig_full_04
dataset: pure
stop_reason: converged
best_primary: 0.602756
- node_001 | method: seed-ensemble | hypothesis: Because validation peaks at epoch 8 and then | primary: 0.602756 | verdict: accepted
- node_002 | method: swa-ema | hypothesis: Replacing the five-seed ensemble with a single FM | primary: 0.602685 | verdict: rejected
- node_003 | method: none | hypothesis: (proposal failed) | primary: n/a | verdict: failed
self_critique:
Overall assessment

The run found a small gain, but the search was too narrow and stopped prematurely. The best result, 0.602756, improved on the reported baseline by only about 0.001—not the hypothesized 0.004—and required a five-model ensemble.

What was suboptimal

- The policy treated a tiny metric difference as decisive. Node_002 at 0.6027 is only about 0.00006 below the winner and is likely statistically indistinguishable, while being substantially cheaper as a single model.
- Acceptance ignored the compute/performance Pareto frontier. Node_002 may be the preferable practical result despite its nominal rejection.
- The search remained confined to seed and checkpoint averaging. It did not explore regularization, early stopping, learning-rate schedules, latent dimension, optimizer settings, feature handling, or alternative objectives.
- The overfitting diagnosis was plausible, but only variance-reduction methods were tested. No direct intervention such as stronger weight decay, reduced capacity, or tuned early stopping was attempted.
- The five-seed result was apparently evaluated once on the same validation setup used for selection. Without independent replication, the roughly 0.001 gain may reflect validation-selection noise.
- Node_003 failed at proposal generation, and the run then declared convergence after only two valid follow-up experiments. That is not convincing evidence of convergence.
- The hypothesis calibration was poor: the predicted gain was about four times the observed gain, but the journal did not update its expectations accordingly.

Scaffold changes

- Track uncertainty across repeated training runs and report confidence intervals or paired seed comparisons.
- Treat configurations within the noise floor as ties, then prefer lower training/inference cost.
- Use multi-objective selection: primary metric, variance, training cost, inference cost, and model count.
- Require a minimum number of successful, meaningfully distinct experiments before allowing convergence.
- Maintain multiple search branches rather than repeatedly mutating the current winner.
- Add proposal-validation and automatic retry/fallback logic so one malformed proposal does not terminate exploration.
- Record exact, unrounded metrics and ensemble construction details.
- Separate confirmation from exploration: re-evaluate a selected improvement on fresh seeds or a held-out split before calling it accepted.

What the next run should try first

First, directly compare node_001 and node_002 with paired fresh seeds and include compute cost. Also compare per-user rank averaging against raw-score averaging using the same predictions. If their performance remains statistically tied, select the single EMA model.

After that, test a small targeted regularization/early-stopping sweep around the baseline—weight decay, latent dimension, and stopping epoch—before spending more search budget on larger ensembles.

## Run logs/run_desig_full_05
dataset: pure
stop_reason: converged
best_primary: 0.601838
- node_001 | method: none | hypothesis: (proposal failed) | primary: n/a | verdict: failed
- node_002 | method: none | hypothesis: Adding a 0.3-weight CWM-style censored watch-ratio auxiliary loss | primary: 0.601550 | verdict: rejected
- node_003 | method: embedding-dim-down | hypothesis: Because validation primary peaks at epoch 8 and | primary: 0.600950 | verdict: rejected
self_critique:
Run critique

The harness converged prematurely. It evaluated only two substantive alternatives after the baseline, neither of which was a well-calibrated local improvement. That is insufficient evidence that the search space is exhausted.

Policy issues:
- node_001 failed at proposal generation, yet node_002 was attached to that failed node rather than cleanly branching from the best accepted node. Failed proposals should be retried or discarded, not retained as ancestry.
- The hypotheses demanded gains of +0.0020 to +0.0025 despite a reported baseline sigma of 0.0001. Those targets are unnecessarily large and encourage coarse changes rather than credible incremental tuning.
- The CWM auxiliary loss introduced a new objective with an arbitrary 0.3 weight and uncertain alignment with the primary metric. It needed either stronger evidence or a small weight sweep.
- Reducing embedding dimension from 16 directly to 8 was a blunt response to overfitting. The result suggests useful capacity was removed; regularization strength, dropout, or training duration would have been more targeted.
- The journal is inconsistent about model identity: “baseline FM” versus “regularized DCN-lite hybrid BCE/BPR.” Model and inherited configuration should be recorded unambiguously.
- There is no evidence of repeated seeds or confirmation runs. Differences of 0.0003–0.0009 should be interpreted using empirical run-to-run variance, not only a single reported sigma.

Scaffold changes:
- Enforce valid proposal generation and always branch experiments from the best accepted runnable node.
- Require each proposal to specify the exact inherited configuration, changed parameters, mechanism, and smallest useful ablation.
- Prefer local, one-factor changes before adding auxiliary objectives.
- Add automatic baseline replication and promotion only after multi-seed confirmation.
- Do not declare convergence after two rejected experiments; require a minimum exploration budget across optimization, regularization, architecture, and loss categories.
- Use diagnostic evidence quantitatively: train/validation curves, best epoch, gap at stopping, and per-component loss behavior.

Next run:
First preserve k=16 and target the observed late-epoch overfit with stronger embedding/model regularization—e.g. a small 2–4× L2 increase or modest embedding dropout around 0.05–0.10—while keeping the loss and optimizer fixed. Test a narrow sweep and replicate the best setting. If that fails, try k=12 rather than the coarse k=8 reduction. Revisit the censored watch-ratio loss only afterward, with much smaller weights and an explicit ablation showing that its target is aligned with validation primary.
