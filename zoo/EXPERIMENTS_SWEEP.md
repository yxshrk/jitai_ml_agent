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
| p0_verify_s42 | dropout=0.1, wd=0, constant lr=1e-3 | 42 | 0.671704 | 0.537585 | 0.604645 | +0.003045 | 2 | 35.2s | no win (seed-42 verification only) |

The reimplementation is verified: its 0.604645 primary is within 0.0002 of the
existing stack's 0.6048 seed-42 score and exhibits the same epoch-2 peak.

## Phase 1 — joint dropout × weight decay × schedule grid

The requested 27-cell grid is sampled with 15–20 informative cells rather than
run exhaustively. Every sampled cell appears below.

| run | dropout | weight decay | schedule | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| g01 | 0.2 | 1e-5 | constant | 42 | 0.670486 | 0.537464 | 0.603975 | +0.002375 | 1 | 32.6s | no win (unconfirmed seed-42 cell) |
| g02 | 0.2 | 1e-3 | constant | 42 | 0.670652 | 0.537517 | 0.604084 | +0.002484 | 1 | 47.6s | no win (unconfirmed seed-42 cell) |
| g03 | 0.35 | 1e-5 | constant | 42 | 0.671313 | 0.537222 | 0.604268 | +0.002668 | 2 | 62.2s | no win (unconfirmed seed-42 cell) |
| g04 | 0.35 | 1e-3 | constant | 42 | 0.671106 | 0.537049 | 0.604078 | +0.002478 | 2 | 1785.6s | failed protocol: resource starvation exceeded 6m cap; score excluded |
| g05 | 0.5 | 1e-5 | constant | 42 | 0.670779 | 0.537652 | 0.604215 | +0.002615 | 1 | 30.1s | no win (unconfirmed seed-42 cell) |
| g06 | 0.5 | 1e-3 | constant | 42 | 0.670914 | 0.537711 | 0.604313 | +0.002713 | 1 | 29.9s | no win (unconfirmed seed-42 cell) |
| g07 | 0.35 | 1e-4 | constant | 42 | 0.671355 | 0.537306 | 0.604331 | +0.002731 | 2 | 55.6s | no win (unconfirmed seed-42 cell) |
| g08 | 0.2 | 1e-5 | cosine | 42 | 0.670486 | 0.537464 | 0.603975 | +0.002375 | 1 | 43.5s | no win (unconfirmed seed-42 cell) |
| g09 | 0.2 | 1e-3 | cosine | 42 | 0.670652 | 0.537517 | 0.604084 | +0.002484 | 1 | 46.8s | no win (unconfirmed seed-42 cell) |
| g10 | 0.5 | 1e-5 | cosine | 42 | 0.670779 | 0.537652 | 0.604215 | +0.002615 | 1 | 41.7s | no win (unconfirmed seed-42 cell) |
| g11 | 0.5 | 1e-3 | cosine | 42 | 0.670914 | 0.537711 | 0.604313 | +0.002713 | 1 | 43.2s | no win (unconfirmed seed-42 cell) |
| g12 | 0.35 | 1e-4 | cosine | 42 | 0.671174 | 0.537184 | 0.604179 | +0.002579 | 2 | 44.0s | no win (unconfirmed seed-42 cell) |

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
