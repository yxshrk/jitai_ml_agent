# Schedule and ensemble sweep — 2026-08-28

Protocol: validation only; every metric is computed by importing
`data/official/evaluate.py`. Exploration uses seed 42. A delta of at least +0.002
over the frozen 0.6016 validation baseline is called a win only after confirmation
at seeds 43 and 44, with population mean ± standard deviation reported. Each run
uses the self-contained DCN-lite reimplementation in `zoo/sweep_train.py` and is
limited to 25 epochs with GAUC early-stopping patience 4.

## Phase 0 — reimplementation verification

| run | config | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

## Phase 1 — joint dropout × weight decay × schedule grid

The requested 27-cell grid is sampled with 15–20 informative cells rather than
run exhaustively. Every sampled cell appears below.

| run | dropout | weight decay | schedule | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|

## Phase 2 — embedding-specific regularization and k=8

| run | config | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

## Phase 3 — batch size and linearly scaled learning rate

| run | batch size | learning rate | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Phase 4 — EMA

| run | config | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

## Phase 5 — diverse rank ensembles

| run | members / rank weights | seed(s) | GAUC | nDCG@5 | primary | delta | verdict |
|---|---|---|---:|---:|---:|---:|---|

## Final summary

Pending completion of all five phases.
