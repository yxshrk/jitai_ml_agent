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
| g04 | 0.35 | 1e-3 | constant | 42 | 0.671106 | 0.537049 | 0.604078 | +0.002478 | 2 | 1785.6s | no win (failed protocol: >6m; score excluded) |
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
| g17 | 0.35 | 1e-4 | step | 42 | 0.672393 | 0.538066 | 0.605229 | +0.003629 | 2 | 1055.7s | no win (failed protocol: >6m; score excluded) |

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
| r05 | k=8; MLP dropout=0.2; embedding dropout=0.1; wd=1e-5; step | 42 | 0.672063 | 0.537881 | 0.604972 | +0.003372 | 12 | 153.8s | regression vs k=16 |

Embedding-specific conclusion: embedding L2 from 1e-3 to 1e-2 is effectively
flat. Embedding dropout 0.1 delays the peak and gives the phase's best score;
0.2 is too strong. The smaller k=8 model survives longer only because step decay
has made the learning rate negligible and finishes below k=16.

## Phase 3 — batch size and linearly scaled learning rate

| run | batch size | learning rate | seed | GAUC | nDCG@5 | primary | delta | epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| b01 | 2048 | 2.5e-4 | 42 | 0.672123 | 0.538306 | 0.605215 | +0.003615 | 7 | 270.6s | no win (below batch=8192 reference) |
| b02 (r03 reference; no retrain) | 8192 | 1e-3 | 42 | 0.672359 | 0.538276 | 0.605318 | +0.003718 | 3 | 68.4s | no win (seed-42 leader; confirmation pending) |
| b03 | 32768 | 4e-3 | 42 | 0.671828 | 0.537394 | 0.604611 | +0.003011 | 7 | 108.3s | regression vs batch=8192 |

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
| e03-confirm | epoch EMA start=2, decay=0.9 | 43 | 0.671889 | 0.537698 | 0.604793 | +0.003193 | 10 raw | 127.2s | raw selected; EMA loses this seed |
| e03-confirm | epoch EMA start=2, decay=0.9 | 44 | 0.671814 | 0.537872 | 0.604843 | +0.003243 | 9 raw | 122.5s | raw selected; EMA loses this seed |
| **e03 raw 3-seed** | r03 raw checkpoints, population mean ± std | 42–44 | **0.672021 ± 0.000241** | **0.537949 ± 0.000242** | **0.604985 ± 0.000236** | **+0.003385** | — | — | **confirmed config win; raw wins 2/3 vs EMA** |
| **e03 selector 3-seed** | best raw-or-EMA checkpoint, population mean ± std | 42–44 | **0.672085 ± 0.000331** | **0.537994 ± 0.000305** | **0.605040 ± 0.000314** | **+0.003440** | — | — | **confirmed config win; no robust EMA win** |

EMA conclusion: decay 0.9 starting at epoch 2 lifts the raw r03 checkpoint by
0.000165 primary at seed 42, but raw checkpoints win seeds 43 and 44. The honest
three-seed verdict is therefore **raw wins the 2–1 majority; EMA is not a robust
improvement**. The underlying r03 configuration remains a confirmed win at
**0.604985 ± 0.000236** raw. Allowing validation to select raw or EMA produces
**0.605040 ± 0.000314**, only +0.000055 above raw and entirely due to seed 42.

## Phase 5 — diverse rank ensembles

All ensemble rows below were re-scored from their saved `predictions.csv` files,
not copied from pre-serialization metrics.

| run | members / rank weights | seed(s) | GAUC | nDCG@5 | primary | delta | verdict |
|---|---|---|---:|---:|---:|---:|---|
| ens01 | e03 k16 EMA + r05 k8 (1:1) | 42 | 0.672549 | 0.538464 | 0.605506 | +0.003906 | no win (seed-42 ensemble; confirmation pending) |
| ens02 | e03 k16 EMA + r05 k8 + g06 constant/dropout-0.5 (1:1:1) | 42 | 0.673209 | 0.538676 | 0.605942 | +0.004342 | no win (seed-42 ensemble leader; confirmation pending) |
| ens03 | ens02 + existing DCN-lite (1:1:1:1) | 42 | 0.672850 | 0.538478 | 0.605664 | +0.004064 | no win (regression vs ens02) |
| ens04 | ens02 + existing MTL (1:1:1:1) | 42 | 0.672777 | 0.538405 | 0.605591 | +0.003991 | no win (regression vs ens02) |
| ens05 | e03 + r05 + g06 (2:1:1) | 42 | 0.672958 | 0.538709 | 0.605833 | +0.004233 | no win (regression vs equal-rank ens02) |
| ens07 | e03 + 7-day recency (1:1) | 42 | 0.673285 | 0.538245 | 0.605765 | +0.004165 | no win (below ens02) |
| ens08 | e03 + r05 + 7-day recency (1:1:1) | 42 | 0.672969 | 0.538347 | 0.605658 | +0.004058 | no win (below ens02) |
| ens09 | e03 + g06 + 7-day recency (1:1:1) | 42 | 0.673543 | 0.538662 | 0.606103 | +0.004503 | no win (seed-42 leader; confirmation pending) |
| ens10 | ens02 + 7-day recency (1:1:1:1) | 42 | 0.673341 | 0.538599 | 0.605970 | +0.004370 | no win (below ens09; not confirmed) |
| ens02-support | g06 constant/dropout-0.5 retrain | 43 | 0.669579 | 0.536227 | 0.602903 | +0.001303 | no win (ensemble confirmation support run) |
| ens02-support | g06 constant/dropout-0.5 retrain | 44 | 0.670173 | 0.537370 | 0.603771 | +0.002171 | no win (ensemble confirmation support run) |
| ens02-support | r05 k8/embedding-dropout-0.1 retrain | 43 | 0.671445 | 0.537551 | 0.604498 | +0.002898 | no win (ensemble confirmation support run) |
| ens02-support | r05 k8/embedding-dropout-0.1 retrain | 44 | 0.671922 | 0.537663 | 0.604793 | +0.003193 | no win (ensemble confirmation support run) |
| ens02-confirm | e03 + r05 + g06 (1:1:1) | 43 | 0.671942 | 0.537835 | 0.604888 | +0.003288 | no win (confirmation incomplete) |
| ens02-confirm | e03 + r05 + g06 (1:1:1) | 44 | 0.671746 | 0.538118 | 0.604932 | +0.003332 | confirmed win (with seeds 42/43) |
| **ens02 3-seed** | same recipe, population mean ± std | 42–44 | **0.672299 ± 0.000648** | **0.538210 ± 0.000350** | **0.605254 ± 0.000487** | **+0.003654** | **confirmed win** |
| ens09-confirm | e03 + g06 + 7-day recency (1:1:1) | 43 | 0.671557 | 0.537535 | 0.604546 | +0.002946 | no win (below ens02 seed 43) |
| ens09-confirm | e03 + g06 + 7-day recency (1:1:1) | 44 | 0.671512 | 0.537731 | 0.604622 | +0.003022 | no win (below ens02 seed 44) |
| **ens09 3-seed** | same recipe, population mean ± std | 42–44 | **0.672204 ± 0.000947** | **0.537976 ± 0.000492** | **0.605090 ± 0.000716** | **+0.003490** | **confirmed vs baseline; loses 2/3 to ens02** |
| ens06 | rank ensemble of e03 seeds 42/43/44 | 42–44 | 0.672660 | 0.538140 | 0.605400 | +0.003800 | confirmed win (single ensemble artifact) |

Architecture-ensemble conclusion: ens02 is the best seed-matched recipe. Adding
the weaker existing DCN-lite or MTL models hurts, as does overweighting the best
single. The confirmed architecture-ensemble mean is 0.605254 ± 0.000487, 0.000215
above the confirmed single-model mean. A three-seed rank ensemble of the best
single config scores 0.605400 as one variance-reduced artifact. The confirmed
7-day recency model was tested in four seed-42 ensemble structures. Replacing k=8
with recency yields the campaign's highest individual score, 0.606103, but falls
below ens02 on seeds 43 and 44 and averages 0.605090 ± 0.000716. Recency therefore
adds useful diversity on seed 42 but does not improve the three-seed recipe and is
not incorporated into the frozen ensemble.

## Final summary

The plateau is broken, modestly but repeatably. The best single configuration is
`zoo/sweep_best.py`: k=16 DCN-lite, hidden=128, two cross layers, MLP dropout 0.2,
embedding dropout 0.1, AdamW wd 1e-5, step LR decay 0.5/epoch, and optional epoch
EMA candidate with decay 0.9 from epoch 2. Validation selects EMA only for seed 42
and raw for seeds 43/44. Official validation primary is 0.605482 / 0.604793 /
0.604843 = **0.605040 ± 0.000314**, delta **+0.003440** over 0.6016: the
configuration is a **confirmed win**, while EMA itself loses the 2–1 majority.
Raw-only primary is **0.604985 ± 0.000236**, delta **+0.003385**.

The best architecture rank ensemble combines equal within-user ranks from:

1. the frozen k=16 best config above;
2. the k=8 step-decay config with embedding dropout 0.1; and
3. the k=16 high-MLP-dropout 0.5, wd 1e-3, constant-LR config.

Its seed-matched primary values are 0.605942 / 0.604888 / 0.604932 =
**0.605254 ± 0.000487**, delta **+0.003654**: **confirmed win**. The highest
individual ensemble validation result is **0.606103** at seed 42 from replacing
k=8 with the 7-day recency model, but that recipe loses to the frozen ensemble on
seeds 43/44 and is not adopted. The existing DCN-lite and MTL artifacts were
explicitly tested as fourth members and regressed.

Main findings: rapid step decay is the only schedule that consistently beats the
constant/cosine plateau; embedding dropout 0.1 delays the useful peak; weight decay
is nearly flat; k=8, batch 2048/32768, and stronger embedding dropout do not win;
EMA helps at seed 42 but raw checkpoints win seeds 43/44; and 7-day recency adds
seed-specific diversity without improving the three-seed architecture ensemble.
All metrics above are validation-only outputs from `data/official/evaluate.py`;
test data was never read.

Reproduction for one seed (replace `<seed>` and output paths as needed):

```bash
uv run python zoo/sweep_best.py --data-dir data/real_ws --out-dir /tmp/best --seed <seed>
uv run python zoo/sweep_train.py --data-dir data/real_ws --out-dir /tmp/k8 --seed <seed> --k 8 --dropout 0.2 --embedding-dropout 0.1 --weight-decay 1e-5 --schedule step
uv run python zoo/sweep_train.py --data-dir data/real_ws --out-dir /tmp/constant --seed <seed> --dropout 0.5 --weight-decay 1e-3 --schedule constant
uv run python zoo/ensemble.py --data-dir data/real_ws --out-dir /tmp/ensemble --seed <seed> --inputs /tmp/best/predictions.csv /tmp/k8/predictions.csv /tmp/constant/predictions.csv
```
