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

TPE sampler seed 42 (`n_startup_trials=8`, multivariate TPE), median pruner
evaluated on each half-epoch primary metric (`n_startup_trials=8`, four-step
warmup, one-step interval). Search jobs used 8 epochs and five-half patience.
For pruned rows, primary is the last reported half-epoch value, not an eligible
final score. There were 22 complete and 18 pruned trials. No run exceeded 62.9s.

| trial | lr | decay factor / every | dropout | wd | k | batch | half-life | BPR wt | primary | notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.001190638 | 0.3624 / 1.5 | 0.3003 | 0.001111499 | 12 | 8192 | 5.455 | 0.4367 | 0.604273322 | complete; epoch 5.0, 50.4s |
| 1 | 0.0005866113 | 0.5447 / 1.5 | 0.2640 | 0.001444775 | 16 | 8192 | 8.860 | 0.4341 | 0.604824944 | complete; epoch 4.0, 37.3s |
| 2 | 0.001929828 | 0.4218 / 1.0 | 0.1805 | 0.0005388109 | 12 | 16384 | 6.494 | 0.5040 | 0.603993616 | complete; epoch 6.0, 54.2s |
| 3 | 0.001787533 | 0.6758 / 1.5 | 0.1721 | 0.0001947558 | 16 | 16384 | 10.630 | 0.4714 | 0.604471433 | complete; epoch 5.0, 62.6s |
| 4 | 0.001902472 | 0.3298 / 0.5 | 0.1514 | 0.001601531 | 16 | 8192 | 6.868 | 0.4232 | 0.602942959 | complete; epoch 2.0, 32.3s |
| 5 | 0.0003472799 | 0.4244 / 1.0 | 0.3718 | 0.0004983319 | 16 | 4096 | 10.168 | 0.4988 | 0.603673893 | complete; epoch 8.0, 62.9s |
| 6 | 0.000384603 | 0.3126 / 0.5 | 0.3769 | 0.000233472 | 12 | 4096 | 6.318 | 0.4322 | 0.598996471 | complete; epoch 3.0, 49.0s |
| 7 | 0.002231423 | 0.6215 / 1.0 | 0.3519 | 0.002106859 | 24 | 4096 | 10.544 | 0.5721 | 0.600718259 | complete; epoch 1.0, 26.7s |
| 8 | 0.0003343278 | 0.5948 / 1.5 | 0.2160 | 0.001404451 | 16 | 8192 | 8.643 | 0.4041 | 0.602586892 | pruned |
| 9 | 0.0006226485 | 0.6737 / 1.5 | 0.3018 | 0.002737079 | 16 | 8192 | 10.277 | 0.4761 | 0.604784593 | complete; epoch 5.0, 44.1s |
| 10 | 0.001570417 | 0.6408 / 1.5 | 0.2429 | 0.0003542791 | 16 | 8192 | 6.005 | 0.4393 | 0.603185013 | pruned |
| 11 | 0.0009061996 | 0.6084 / 1.5 | 0.2501 | 0.00259211 | 16 | 8192 | 8.354 | 0.5098 | 0.604831442 | complete; epoch 5.0, 47.9s |
| 12 | 0.0009595469 | 0.4839 / 1.5 | 0.2121 | 0.002778631 | 16 | 8192 | 8.282 | 0.5382 | 0.604580156 | complete; epoch 5.0, 49.9s |
| 13 | 0.001689865 | 0.6348 / 1.5 | 0.2881 | 0.002828351 | 8 | 8192 | 8.751 | 0.5477 | 0.604264911 | complete; epoch 2.0, 23.8s |
| 14 | 0.0004378031 | 0.6389 / 1.5 | 0.3185 | 0.002403032 | 12 | 8192 | 5.431 | 0.5245 | 0.602816085 | pruned |
| 15 | 0.00106753 | 0.5644 / 1.0 | 0.2363 | 0.002137615 | 16 | 4096 | 6.458 | 0.4441 | 0.604052303 | complete; epoch 2.0, 31.5s |
| 16 | 0.001191109 | 0.5393 / 1.5 | 0.1880 | 0.002058841 | 12 | 8192 | 9.369 | 0.4270 | 0.604500112 | complete; epoch 3.5, 33.6s |
| 17 | 0.0007903421 | 0.6657 / 1.5 | 0.2332 | 0.002321125 | 16 | 16384 | 5.300 | 0.5001 | 0.603629745 | pruned |
| 18 | 0.0006297316 | 0.5428 / 1.5 | 0.2024 | 0.0007455318 | 16 | 16384 | 11.167 | 0.4977 | 0.602347594 | pruned |
| 19 | 0.001558361 | 0.6949 / 0.5 | 0.2023 | 0.002244414 | 16 | 8192 | 9.526 | 0.4369 | **0.604950815** | complete; epoch 2.5, 34.1s; rank 1 |
| 20 | 0.001279882 | 0.6928 / 0.5 | 0.1521 | 0.001350235 | 16 | 4096 | 11.723 | 0.4907 | 0.603507196 | pruned |
| 21 | 0.001709804 | 0.6420 / 0.5 | 0.2354 | 0.002626602 | 24 | 8192 | 8.977 | 0.4472 | 0.604077661 | pruned |
| 22 | 0.001314445 | 0.6707 / 0.5 | 0.2664 | 0.001045314 | 16 | 8192 | 10.654 | 0.4179 | 0.604815203 | complete; epoch 5.0, 49.6s |
| 23 | 0.0004916961 | 0.5647 / 1.5 | 0.3092 | 0.0006086779 | 12 | 8192 | 11.594 | 0.4084 | 0.602793228 | pruned |
| 24 | 0.00104876 | 0.6538 / 1.0 | 0.1569 | 0.001167433 | 16 | 8192 | 8.417 | 0.4370 | 0.604828596 | complete; epoch 5.0, 48.8s |
| 25 | 0.0008922458 | 0.6401 / 1.0 | 0.1588 | 0.000977594 | 16 | 16384 | 7.907 | 0.4316 | 0.603451506 | pruned |
| 26 | 0.0006301979 | 0.6326 / 1.0 | 0.1657 | 0.0004384305 | 24 | 8192 | 7.388 | 0.4323 | 0.603755284 | pruned |
| 27 | 0.001059361 | 0.6622 / 1.0 | 0.1762 | 0.002087429 | 16 | 8192 | 7.894 | 0.4118 | 0.604806068 | complete; epoch 5.0, 48.9s |
| 28 | 0.002757272 | 0.5939 / 0.5 | 0.2588 | 0.002183944 | 16 | 16384 | 11.731 | 0.4710 | 0.603875810 | pruned |
| 29 | 0.002863681 | 0.6409 / 0.5 | 0.1640 | 0.002317758 | 16 | 4096 | 6.956 | 0.4710 | 0.598636734 | pruned |
| 30 | 0.001427977 | 0.6576 / 1.5 | 0.1959 | 0.002562114 | 16 | 8192 | 8.450 | 0.4863 | 0.603609168 | pruned |
| 31 | 0.000485422 | 0.4677 / 1.0 | 0.2924 | 0.00293276 | 16 | 8192 | 8.128 | 0.4645 | 0.602661526 | pruned |
| 32 | 0.0007003875 | 0.6103 / 1.0 | 0.1820 | 0.0003144226 | 16 | 8192 | 11.424 | 0.4453 | **0.604909703** | complete; epoch 8.0, 51.0s; rank 2 |
| 33 | 0.0003797022 | 0.6339 / 1.0 | 0.1651 | 0.0001593018 | 16 | 8192 | 10.940 | 0.4172 | 0.602390673 | pruned |
| 34 | 0.0007563363 | 0.5192 / 1.0 | 0.2032 | 0.0004299367 | 16 | 8192 | 10.497 | 0.4623 | 0.604778517 | complete; epoch 5.5, 55.3s |
| 35 | 0.001182555 | 0.6687 / 0.5 | 0.1679 | 0.0008717549 | 8 | 4096 | 8.007 | 0.4129 | **0.604901426** | complete; epoch 4.0, 41.0s; rank 3 |
| 36 | 0.001214979 | 0.5891 / 1.5 | 0.2394 | 0.0006443571 | 8 | 4096 | 6.525 | 0.4139 | 0.604415857 | complete; epoch 2.0, 27.4s |
| 37 | 0.001993192 | 0.6016 / 0.5 | 0.2236 | 0.0008639219 | 8 | 4096 | 10.143 | 0.4381 | 0.604119826 | pruned |
| 38 | 0.001230183 | 0.6618 / 0.5 | 0.1942 | 0.0007000427 | 8 | 4096 | 7.655 | 0.4703 | 0.604763050 | complete; epoch 4.0, 36.2s |
| 39 | 0.001498139 | 0.6311 / 0.5 | 0.2213 | 0.0006886688 | 8 | 8192 | 7.341 | 0.4099 | 0.603915003 | pruned |

## Top-three confirmation

| rank / trial | seed 42 | seed 43 | seed 44 | mean | pop. std | delta vs 0.604700 | >= +0.001? |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 / 19 | 0.604950815 | 0.603952728 | 0.604629297 | 0.604510947 | 0.000415973 | -0.000189053 | **FAIL** |
| 2 / 32 | 0.604909703 | 0.604329316 | 0.604618083 | **0.604619034** | **0.000236943** | **-0.000080966** | **FAIL** |
| 3 / 35 | 0.604901426 | 0.603919396 | 0.604619888 | 0.604480236 | 0.000412894 | -0.000219764 | **FAIL** |

No claimed improvement survives confirmation. Trial 32 is called “best confirmed”
only because its mean is the highest of these candidates; it is not a win over
the frozen stack and misses the required delta by 0.001080966.

## Five-seed rank-average ensemble and submission recipe

The best confirmed configuration (trial 32) was additionally run at seeds 45
and 46. Individual primaries for seeds 42--46 were 0.604909703, 0.604329316,
0.604618083, 0.604365332, and 0.604891252 (mean 0.604622737, population std
0.000247719). Global within-run ranks were averaged equally and rescored:

| ensemble | GAUC | nDCG@5 | primary | delta vs 0.604700 | verdict |
|---|---:|---:|---:|---:|---|
| trial-32 seeds 42--46 rank average | 0.671604592 | 0.537893023 | **0.604748808** | +0.000048808 | variance-reduction recipe; not a real win |

Final submission recipe: use `zoo/polish_best.py` to train seeds 42--46, then
pass their `scores.npy` paths to the same script's `--rank-average-scores` mode.
The wired parameters are lr 0.0007003874872132884, decay 0.6103216481366316
every epoch, dropout 0.18199037655935982, embedding dropout 0.1, weight decay
0.00031442255073239905, k=16, batch=8192, recency half-life
11.424348428709624 days, and BPR weight 0.4453160212036508.

## Longer-training mini-sweep

All cells use 0.5 decay every half epoch after epoch 2 and the winning schedule
family's remaining parameters.

| cell | epochs | patience halves | primary | best epoch | runtime | notes |
|---|---:|---:|---:|---:|---:|---|
| long-12 | 12 | 40 | 0.604903247 | 4.0 | 95.5s | no gain after epoch 4; LR effectively zero |
| long-14 | 14 | 40 | 0.604903247 | 4.0 | 80.8s | identical selected checkpoint |
| long-16 | 16 | 40 | 0.604903247 | 4.0 | 180.6s | identical selected checkpoint; still under 6m |

The schedule answers “can we train longer yet?” with **no**: its best checkpoint
is already epoch 4. The half-epoch factor of 0.5 drives the learning rate below
1e-8 by epoch 10, so extra epochs merely repeat the converged ranking. The
12/14/16 cells deliberately used patience 40 to observe the full requested
horizons; no run needed shortening and all stayed below six minutes.

## Final summary

The frozen implementation reproduced at 0.604998355. TPE found small seed-42
movements, but the top three all regressed in three-seed mean versus 0.604700;
there is **no real win** under the stated +0.001 rule. Trial 32 had the most
stable/highest confirmed mean and is wired into `polish_best.py`. Its five-seed
rank average scores 0.604748808, effectively matching rather than improving the
frozen stack. Stronger delayed half-epoch decay peaks at epoch 4 regardless of a
12-, 14-, or 16-epoch horizon, so longer training does not help.
