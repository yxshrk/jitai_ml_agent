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
| g13 | 0.2 | 1e-5 | step | 42 | 0.672548 | 0.537992 | 0.605270 | +0.003670 | 2 | 46.4s | no win (promising seed-42 cell; confirmation pending) |
| g14 | 0.2 | 1e-3 | step | 42 | 0.672494 | 0.537974 | 0.605234 | +0.003634 | 2 | 49.7s | no win (promising seed-42 cell; confirmation pending) |
| g15 | 0.5 | 1e-5 | step | 42 | 0.671895 | 0.537799 | 0.604847 | +0.003247 | 2 | 42.3s | no win (unconfirmed seed-42 cell) |
| g16 | 0.5 | 1e-3 | step | 42 | 0.671990 | 0.537840 | 0.604915 | +0.003315 | 2 | 40.0s | no win (unconfirmed seed-42 cell) |
| g17 | 0.35 | 1e-4 | step | 42 | 0.672393 | 0.538066 | 0.605229 | +0.003629 | 2 | 1055.7s | failed protocol: resource suspension exceeded 6m cap; score excluded |

Grid conclusion: constant and 25-epoch cosine schedules are flat; cosine has not
decayed enough before early stopping. Step decay is consistently better, with g13
(dropout 0.2, wd 1e-5) the valid seed-42 leader at 0.605270. Weight decay has little
effect within each matched pair. Runs g04 and g17 were resource-suspended beyond
the runtime cap and are retained in the log for completeness but excluded from
model selection. No grid cell is called a win without seeds 43/44.

## Phase 2 — embedding-specific regularization and k=8

| run | config | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| r01 | MLP dropout=0.2/wd=1e-5; embedding dropout=0/wd=1e-3; step | 42 | 0.672518 | 0.538016 | 0.605267 | +0.003667 | 2 | 40.1s | no win (promising seed-42 cell; confirmation pending) |
| r02 | MLP dropout=0.2/wd=1e-5; embedding dropout=0/wd=1e-2; step | 42 | 0.672557 | 0.538032 | 0.605294 | +0.003694 | 2 | 52.0s | no win (promising seed-42 cell; confirmation pending) |
| r03 | MLP dropout=0.2; embedding dropout=0.1; wd=1e-5; step | 42 | 0.672359 | 0.538276 | 0.605318 | +0.003718 | 3 | 68.4s | no win (seed-42 leader; confirmation pending) |
| r04 | MLP dropout=0.2; embedding dropout=0.2; wd=1e-5; step | 42 | 0.672110 | 0.538157 | 0.605133 | +0.003533 | 4 | 100.3s | no win (unconfirmed seed-42 cell) |
| r05 | k=8; MLP dropout=0.2; embedding dropout=0.1; wd=1e-5; step | 42 | 0.672063 | 0.537881 | 0.604972 | +0.003372 | 12 | 153.8s | no win (worse than k=16) |

Embedding-specific conclusion: embedding L2 from 1e-3 to 1e-2 is effectively
flat. Embedding dropout 0.1 delays the peak and gives the phase's best score;
0.2 is too strong. The smaller k=8 model survives longer only because step decay
has made the learning rate negligible and finishes below k=16.

## Phase 3 — batch size and linearly scaled learning rate

| run | batch size | learning rate | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b01 | 2048 | 2.5e-4 | 42 | 0.672123 | 0.538306 | 0.605215 | +0.003615 | 7 | 270.6s | no win (below batch=8192 reference) |
| b02 (r03 reference; no retrain) | 8192 | 1e-3 | 42 | 0.672359 | 0.538276 | 0.605318 | +0.003718 | 3 | 68.4s | no win (seed-42 leader; confirmation pending) |
| b03 | 32768 | 4e-3 | 42 | 0.671828 | 0.537394 | 0.604611 | +0.003011 | 7 | 108.3s | no win (regression vs batch=8192) |

Batch conclusion: the default 8192 batch remains best. The 2048 batch is close
but costs 4× as many optimizer steps and 270.6s; 32768 with linear LR scaling is
a clear regression.

## Phase 4 — EMA

| run | config | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| e01 | epoch EMA start=2, decay=0.5 on r03 | 42 | 0.672359 | 0.538276 | 0.605318 | +0.003718 | 3 raw | 84.2s | no win (EMA never beats best raw checkpoint) |
| e02 | epoch EMA start=2, decay=0.8 on r03 | 42 | 0.672451 | 0.538150 | 0.605301 | +0.003701 | 8 EMA | 118.2s | no win (GAUC up, primary below raw) |
| e03 | epoch EMA start=2, decay=0.9 on r03 | 42 | 0.672551 | 0.538413 | 0.605482 | +0.003882 | 9 EMA | 128.1s | no win (seed-42 leader; confirmation pending) |
| e04 | epoch EMA start=3, decay=0.9 on r03 | 42 | 0.672386 | 0.538261 | 0.605324 | +0.003724 | 5 EMA | 81.5s | no win (below start=2) |
| e03-confirm | epoch EMA start=2, decay=0.9 | 43 | 0.671889 | 0.537698 | 0.604793 | +0.003193 | 10 raw | 127.2s | no win (confirmation incomplete) |
| e03-confirm | epoch EMA start=2, decay=0.9 | 44 | 0.671814 | 0.537872 | 0.604843 | +0.003243 | 9 raw | 122.5s | confirmed win (with seeds 42/43) |
| **e03 3-seed** | same config, population mean ± std | 42–44 | **0.672085 ± 0.000331** | **0.537994 ± 0.000305** | **0.605040 ± 0.000314** | **+0.003440** | — | — | **confirmed win** |

EMA conclusion: decay 0.9 starting at epoch 2 is best and lifts the raw r03
checkpoint by 0.000165 primary at seed 42. Across seeds 42–44 the complete config
scores **0.605040 ± 0.000314**, delta +0.003440: a confirmed win. Seeds 43 and 44
select later raw checkpoints, so EMA is beneficial but not universally selected.

## Phase 5 — diverse rank ensembles

| run | members / rank weights | seed(s) | GAUC | nDCG@5 | primary | delta | verdict |
|---|---|---|---:|---:|---:|---:|---|
| ens01 | e03 k16 EMA + r05 k8 (1:1) | 42 | 0.672567 | 0.538476 | 0.605521 | +0.003921 | no win (seed-42 ensemble; confirmation pending) |
| ens02 | e03 k16 EMA + r05 k8 + g06 constant/dropout-0.5 (1:1:1) | 42 | 0.673170 | 0.538653 | 0.605911 | +0.004311 | no win (seed-42 ensemble leader; confirmation pending) |

## Final summary

Pending completion of all five phases.
