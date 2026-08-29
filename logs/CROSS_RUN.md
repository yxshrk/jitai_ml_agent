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
