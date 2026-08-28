# Frozen-stack overnight polish

## Protocol

Validation only (`data/real_ws`), official primary scorer at every half epoch.
Exploration uses seed 42; a claimed win requires seeds 42/43/44 with mean delta
at least +0.001000 over the frozen-stack reference 0.604700. Population standard
deviation is reported. Every individual process has a 350-second alarm.

Dependency note: the repository safety contract forbids modifying any existing
file, which includes `pyproject.toml` and `uv.lock`. Consequently Optuna is run
through `uv run --with optuna` rather than `uv add optuna`; this preserves both
uv-only dependency management and the stricter file-mutation contract.

## Baseline reproduction

| run | seed | lr / decay | dropout / emb-drop | wd | k / batch | half-life / BPR | primary | best epoch | runtime | result |
|---|---:|---|---|---:|---|---|---:|---:|---:|---|
| baseline-repro | 42 | 0.001 / 0.5 every 1 epoch | 0.2 / 0.1 | 0.00001 | 16 / 8192 | 7 / 0.5 | **0.604998355** | 3.5 | 43.2s | reproduced expected 0.6047-0.6050 |

## Optuna TPE exploration

TPE sampler seed 42, median pruner evaluated on each half-epoch primary metric.
The exact sampler/pruner settings and results follow after execution.

| trial | lr | decay factor / every | dropout | wd | k | batch | half-life | BPR wt | primary | notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|

## Top-three confirmation

| rank / trial | seed 42 | seed 43 | seed 44 | mean | pop. std | delta vs 0.604700 | >= +0.001? |
|---|---:|---:|---:|---:|---:|---:|---|

## Five-seed rank-average ensemble and submission recipe

Pending.

## Longer-training mini-sweep

All cells use 0.5 decay every half epoch after epoch 2 and the winning schedule
family's remaining parameters.

| cell | epochs | patience halves | primary | best epoch | runtime | notes |
|---|---:|---:|---:|---:|---:|---|

## Final summary

Pending completion of the campaign.
