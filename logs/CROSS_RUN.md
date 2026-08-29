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
