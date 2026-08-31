## Run logs/run_farm_f4r2
dataset: pure
stop_reason: converged
best_primary: 0.604499
- node_001 | method: gauge-fixed-bce | hypothesis: (proposal failed) | primary: n/a | verdict: failed
- node_002 | method: gauge-fixed-bce | hypothesis: Replacing ordinary pointwise logits with complete-slate user-centered BCE | primary: 0.602915 | verdict: accepted
- node_003 | method: diverse-family-farm-close | hypothesis: Because the parent validation primary is still rising | primary: 0.604499 | verdict: accepted
self_critique:
Run critique

What was suboptimal:
- The search was too shallow to justify “converged”: only three metric-bearing nodes were evaluated, with one failed proposal.
- node_001 failed before producing a metric, yet node_002 inherited from it. The harness should have recovered from the last valid parent, node_000, and recorded the failed attempt as a side branch.
- The reported sigma=0.0001 is implausibly precise unless based on multiple independent seeds. If it came from a single run or epoch noise, acceptance thresholds were overconfident.
- Hypotheses were more specific than the evidence supported. node_002 predicted +0.0026 but delivered +0.0011; node_003 predicted +0.0030 but delivered +0.0016.
- node_003 changed several dimensions simultaneously—model families, temporal/causal features, recency weighting, and rank ensembling. This found the best score, but provides little causal information and risks validation-set overfitting through model/weight selection.
- The “still rising at epoch 10” observation should first have triggered a controlled longer-training experiment, not immediately a complex ensemble.
- There is no evidence of repeated-seed confirmation, component ablations, ensemble-weight fitting discipline, or evaluation on a held-out secondary split.

Scaffold changes:
- Add proposal preflight checks so malformed proposals fail before consuming a search node.
- Keep failed nodes out of ancestry; branch from the most recent executable checkpoint.
- Require each acceptance to exceed uncertainty estimated from repeated seeds or paired bootstrap analysis.
- Separate atomic experiments from compound ones. Enforce one primary intervention per node, then build ensembles only from validated components.
- Track predicted versus realized gain and calibrate future proposal confidence.
- Add explicit convergence criteria based on search budget, replicated performance, and lack of improvement across several diverse branches—not merely a short accepted chain.
- Preserve per-component predictions so ensemble gains can be attributed and redundant models removed.

What the next run should try first:
1. Reproduce node_003 across multiple seeds and verify the +0.0027 total gain over baseline is stable.
2. Run a controlled longer-training/early-stopping sweep on node_002, since underfitting was the stated motivation.
3. Ablate node_003 into its DCN, temporal-kernel, causal sequential DeepFM, recency-weighted FM, and ensemble-only contributions.
4. Rebuild the ensemble incrementally using out-of-fold or untouched validation predictions, selecting the smallest subset that retains the gain.
5. If the ensemble remains best, tune blending weights under strict holdout discipline; otherwise continue from the strongest single component, prioritizing the user-centered slate loss plus the most useful temporal/recency feature family.
