## Run logs/run_f19
dataset: pure
stop_reason: converged
best_primary: 0.604686
- node_001 | method: package-dial-sweep | hypothesis: The epoch-8 validation peak followed by decline diagnoses | primary: 0.603896 | verdict: accepted
- node_002 | method: context-stratified-pairs | hypothesis: The validation curve peaks near 0.6039 and then | primary: 0.604217 | verdict: rejected
- node_003 | method: gauge-fixed-bce | hypothesis: Validation peaks near 0.6039 and then declines, diagnosing | primary: 0.603487 | verdict: rejected
- node_004 | method: ensemble-design-sweep | hypothesis: Because validation peaks early and then falls, the | primary: 0.604686 | verdict: accepted
self_critique:
Overall: the run found a modest improvement from 0.6018 to 0.604686, but it converged too quickly and the final gain may be partly validation-selection bias rather than a robust modeling advance.

What was suboptimal

- Exploration was shallow: only four proposals beyond baseline, with three children concentrated around the same accepted package. “Converged” is not well supported by this search breadth.
- Node_001 bundled architecture, loss mixing, dropout, AdamW, scheduling, recency weighting, and a two-stage dial search. This produced a gain but destroyed attribution; the harness cannot tell which components mattered.
- Predictions were overconfident and poorly calibrated. Node_002 and node_003 forecast +0.0012 and +0.0022 but delivered roughly +0.0003 and -0.0004 relative to node_001.
- Node_002 was rejected despite scoring above its parent. Even if it missed a statistical acceptance threshold, that result was useful evidence and warranted replication or a smaller contextual-negative sweep rather than immediate abandonment.
- Node_004 used 48 retrainings and a gated ensemble-design sweep on the same validation signal used for acceptance. That is both compute-heavy and especially vulnerable to validation overfitting. The final +0.0008 over node_001 needs confirmation on untouched data or repeated splits.
- Repeatedly diagnosing “overfit” from an early validation peak was reasonable, but the policy mostly restated that diagnosis rather than testing competing causes such as seed variance, optimization instability, or recency distribution shift.
- Reporting one sigma only for the baseline is insufficient. At this effect size, seed-level means, dispersion, and paired comparisons are essential.

Scaffold changes

- Decompose bundled proposals into controlled ablations, then combine only demonstrated contributors.
- Track uncertainty for every node using matched seeds and paired bootstrap or repeated-split estimates.
- Separate exploratory validation from final acceptance validation, particularly for ensemble member selection and gating.
- Treat near-miss nodes as uncertain evidence, not dead ends; replicate promising results that improve numerically but fail the gate.
- Penalize proposals by compute and selection degrees of freedom. Compare ensembles against equal-budget alternatives.
- Require a convergence criterion based on search coverage and confidence intervals, not merely several failed children.
- Record per-component ablations, seed predictions, ensemble diversity, and out-of-fold ensemble weights.

What the next run should try first

First, reproduce node_001 and node_004 across matched seeds on an untouched validation split. Evaluate a fixed uniform or rank-averaged ensemble chosen without tuning on that split, and compare it with the gated ensemble. This will establish whether the 0.604686 result is real.

If confirmed, run a compact ablation of node_001’s major components—DCN-lite, BPR mixing, recency weighting, and the regularization/schedule package—using matched seeds. Then revisit contextual BPR negatives from node_002 with a small mixture sweep and replication, since it showed a positive numerical signal at relatively low conceptual complexity.
