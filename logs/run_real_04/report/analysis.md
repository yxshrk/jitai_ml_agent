# Post-hoc analysis: run_real_04

Metrics are recomputed from validation predictions with `data/official/evaluate.py` conventions.
Rank-change uses mean per-user Kendall tau-b; high change means tau < 0.90 or top-1 changes for at least 10% of users.

## Data availability notes

- Train-history tercile cut points across validation users: 22, 52.

## Segment metrics vs node_000

| node | segment | rows | GAUC | nDCG@5 | primary | primary delta vs node_000 | flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| node_000 | duration<=18000ms | 22292 | 0.710969 | 0.389475 | 0.550222 | +0.000000 |  |
| node_000 | duration>18000ms | 102617 | 0.653025 | 0.527225 | 0.590125 | +0.000000 |  |
| node_000 | tab=0 | 92672 | 0.617813 | 0.540364 | 0.579089 | +0.000000 |  |
| node_000 | tab=1 | 13726 | 0.566336 | 0.062949 | 0.314642 | +0.000000 |  |
| node_000 | tab=2 | 547 | 0.675926 | 0.030182 | 0.353054 | +0.000000 |  |
| node_000 | tab=3 | 5170 | 0.601343 | 0.105350 | 0.353347 | +0.000000 |  |
| node_000 | tab=4 | 3834 | 0.623928 | 0.512085 | 0.568007 | +0.000000 |  |
| node_000 | tab=5 | 7877 | 0.560767 | 0.546067 | 0.553417 | +0.000000 |  |
| node_000 | tab=6 | 226 | 0.611696 | 0.169744 | 0.390720 | +0.000000 |  |
| node_000 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_000 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_000 | tab=9 | 121 | 0.722222 | 0.247010 | 0.484616 | +0.000000 |  |
| node_000 | tab=10 | 92 | 0.464286 | 0.248250 | 0.356268 | +0.000000 |  |
| node_000 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_000 | tab=12 | 30 | 0.166667 | 0.653944 | 0.410305 | +0.000000 |  |
| node_000 | tab=13 | 57 | 0.718137 | 0.481204 | 0.599670 | +0.000000 |  |
| node_000 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_000 | history:low<=22 | 26709 | 0.659964 | 0.535855 | 0.597910 | +0.000000 |  |
| node_000 | history:mid(22,52] | 36727 | 0.667011 | 0.555800 | 0.611406 | +0.000000 |  |
| node_000 | history:high>52 | 61473 | 0.671626 | 0.516914 | 0.594270 | +0.000000 |  |
| node_000 | val-list<=5 | 37566 | 0.660885 | 0.526680 | 0.593782 | +0.000000 |  |
| node_000 | val-list>5 | 87343 | 0.670041 | 0.552636 | 0.611339 | +0.000000 |  |
| node_001 | duration<=18000ms | 22292 | 0.710191 | 0.389718 | 0.549955 | -0.000268 |  |
| node_001 | duration>18000ms | 102617 | 0.653412 | 0.527041 | 0.590226 | +0.000101 |  |
| node_001 | tab=0 | 92672 | 0.618590 | 0.540236 | 0.579413 | +0.000325 |  |
| node_001 | tab=1 | 13726 | 0.557859 | 0.062912 | 0.310385 | -0.004257 |  |
| node_001 | tab=2 | 547 | 0.268519 | 0.025070 | 0.146794 | -0.206260 |  |
| node_001 | tab=3 | 5170 | 0.673382 | 0.106449 | 0.389915 | +0.036569 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=4 | 3834 | 0.628310 | 0.511809 | 0.570059 | +0.002053 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=5 | 7877 | 0.575963 | 0.547508 | 0.561735 | +0.008318 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=6 | 226 | 0.542690 | 0.167752 | 0.355221 | -0.035499 |  |
| node_001 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_001 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=9 | 121 | 0.777778 | 0.252868 | 0.515323 | +0.030707 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=10 | 92 | 0.464286 | 0.248250 | 0.356268 | +0.000000 |  |
| node_001 | tab=11 | 84 | 0.535714 | 0.273264 | 0.404489 | -0.039015 |  |
| node_001 | tab=12 | 30 | 0.500000 | 0.662192 | 0.581096 | +0.170790 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=13 | 57 | 0.563725 | 0.475698 | 0.519712 | -0.079959 |  |
| node_001 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_001 | history:low<=22 | 26709 | 0.664079 | 0.536078 | 0.600079 | +0.002169 | GATING/ENSEMBLE CANDIDATE |
| node_001 | history:mid(22,52] | 36727 | 0.665191 | 0.554397 | 0.609794 | -0.001612 |  |
| node_001 | history:high>52 | 61473 | 0.672429 | 0.518013 | 0.595221 | +0.000951 |  |
| node_001 | val-list<=5 | 37566 | 0.663536 | 0.526946 | 0.595241 | +0.001459 |  |
| node_001 | val-list>5 | 87343 | 0.669998 | 0.552120 | 0.611059 | -0.000280 |  |
| node_002 | duration<=18000ms | 22292 | 0.710851 | 0.389807 | 0.550329 | +0.000107 |  |
| node_002 | duration>18000ms | 102617 | 0.653641 | 0.527518 | 0.590579 | +0.000454 |  |
| node_002 | tab=0 | 92672 | 0.618339 | 0.540895 | 0.579617 | +0.000529 |  |
| node_002 | tab=1 | 13726 | 0.548888 | 0.062560 | 0.305724 | -0.008919 |  |
| node_002 | tab=2 | 547 | 0.453704 | 0.027311 | 0.240507 | -0.112547 |  |
| node_002 | tab=3 | 5170 | 0.666056 | 0.106083 | 0.386070 | +0.032723 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=4 | 3834 | 0.624451 | 0.511581 | 0.568016 | +0.000010 |  |
| node_002 | tab=5 | 7877 | 0.579501 | 0.547527 | 0.563514 | +0.010097 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=6 | 226 | 0.544444 | 0.160710 | 0.352577 | -0.038143 |  |
| node_002 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_002 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_002 | tab=9 | 121 | 0.694444 | 0.241151 | 0.467798 | -0.016818 |  |
| node_002 | tab=10 | 92 | 0.621429 | 0.255125 | 0.438277 | +0.082009 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_002 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=13 | 57 | 0.757353 | 0.527226 | 0.642289 | +0.042619 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_002 | history:low<=22 | 26709 | 0.661299 | 0.535949 | 0.598624 | +0.000714 |  |
| node_002 | history:mid(22,52] | 36727 | 0.666899 | 0.554878 | 0.610888 | -0.000517 |  |
| node_002 | history:high>52 | 61473 | 0.672910 | 0.519704 | 0.596307 | +0.002037 | GATING/ENSEMBLE CANDIDATE |
| node_002 | val-list<=5 | 37566 | 0.661301 | 0.526582 | 0.593942 | +0.000159 |  |
| node_002 | val-list>5 | 87343 | 0.671041 | 0.554607 | 0.612824 | +0.001485 |  |
| node_003 | duration<=18000ms | 22292 | 0.709450 | 0.390143 | 0.549796 | -0.000426 |  |
| node_003 | duration>18000ms | 102617 | 0.649694 | 0.525937 | 0.587815 | -0.002310 |  |
| node_003 | tab=0 | 92672 | 0.611843 | 0.538388 | 0.575115 | -0.003973 |  |
| node_003 | tab=1 | 13726 | 0.531697 | 0.061853 | 0.296775 | -0.017867 |  |
| node_003 | tab=2 | 547 | 0.268519 | 0.025251 | 0.146885 | -0.206169 |  |
| node_003 | tab=3 | 5170 | 0.657509 | 0.106338 | 0.381924 | +0.028577 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=4 | 3834 | 0.621332 | 0.512210 | 0.566771 | -0.001236 |  |
| node_003 | tab=5 | 7877 | 0.568137 | 0.546513 | 0.557325 | +0.003908 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=6 | 226 | 0.647661 | 0.171434 | 0.409547 | +0.018827 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_003 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_003 | tab=9 | 121 | 0.592593 | 0.242252 | 0.417422 | -0.067194 |  |
| node_003 | tab=10 | 92 | 0.535714 | 0.250468 | 0.393091 | +0.036823 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_003 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=13 | 57 | 0.593137 | 0.488743 | 0.540940 | -0.058730 |  |
| node_003 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_003 | history:low<=22 | 26709 | 0.659263 | 0.536073 | 0.597668 | -0.000242 |  |
| node_003 | history:mid(22,52] | 36727 | 0.661154 | 0.553176 | 0.607165 | -0.004241 |  |
| node_003 | history:high>52 | 61473 | 0.667354 | 0.514546 | 0.590950 | -0.003320 |  |
| node_003 | val-list<=5 | 37566 | 0.657619 | 0.525998 | 0.591808 | -0.001974 |  |
| node_003 | val-list>5 | 87343 | 0.665763 | 0.549526 | 0.607645 | -0.003694 |  |

## Ranking change vs parent

| node | parent | mean Kendall tau-b | top-1 changed | primary delta | classification |
| --- | --- | --- | --- | --- | --- |
| node_001 | node_000 | 0.833701 | 17.02% | +0.000328 | flat metrics + high change (ensemble candidate) |
| node_002 | node_000 | 0.845139 | 15.01% | +0.000749 | flat metrics + high change (ensemble candidate) |
| node_003 | node_000 | 0.741250 | 23.24% | -0.002786 | metric change + high ranking change |

## Post-hoc ensembles

Best single: `run_real_04/node_002` — primary 0.602601.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| all accepted nodes | 1 | 0.667603 | 0.536102 | 0.601853 | -0.000749 |  |
| best + high-change rejected nodes | 3 | 0.668284 | 0.536666 | 0.602475 | -0.000126 |  |
| all nodes | 4 | 0.669307 | 0.537097 | 0.603202 | +0.000601 |  |

Members:
- all accepted nodes: `run_real_04/node_000`
- best + high-change rejected nodes: `run_real_04/node_002`, `run_real_04/node_001`, `run_real_04/node_003`
- all nodes: `run_real_04/node_000`, `run_real_04/node_001`, `run_real_04/node_002`, `run_real_04/node_003`

## Curve stats

Train-loss slope is an ordinary least-squares slope over the final three usable epochs.

| node | val peak epoch | val peak | final val | peak-final | train-loss slope at stop | note |
| --- | --- | --- | --- | --- | --- | --- |
| node_000 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_001 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_002 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_003 | N/A | N/A | N/A | N/A | N/A | history unavailable |

## Combined cross-run ensembles

Best single: `run_real_01/node_002` — primary 0.604149.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| cross-run: all accepted nodes | 9 | 0.669630 | 0.536956 | 0.603293 | -0.000856 |  |
| cross-run: best + high-change rejected nodes | 13 | 0.669630 | 0.537562 | 0.603596 | -0.000553 |  |
| cross-run: all nodes | 24 | 0.671180 | 0.537782 | 0.604481 | +0.000332 |  |

Members:
- cross-run: all accepted nodes: `run_real_01/node_000`, `run_real_01/node_002`, `run_real_02/node_000`, `run_real_02/node_002`, `run_real_03/node_000`, `run_real_03/node_002`, `run_real_04/node_000`, `run_real_05/node_000`, `run_real_05/node_003`
- cross-run: best + high-change rejected nodes: `run_real_01/node_002`, `run_real_01/node_001`, `run_real_01/node_003`, `run_real_01/node_004`, `run_real_02/node_001`, `run_real_02/node_003`, `run_real_03/node_001`, `run_real_03/node_003`, `run_real_04/node_001`, `run_real_04/node_002`, `run_real_04/node_003`, `run_real_05/node_001`, `run_real_05/node_002`
- cross-run: all nodes: `run_real_01/node_000`, `run_real_01/node_001`, `run_real_01/node_002`, `run_real_01/node_003`, `run_real_01/node_004`, `run_real_01/node_005`, `run_real_02/node_000`, `run_real_02/node_001`, `run_real_02/node_002`, `run_real_02/node_003`, `run_real_02/node_004`, `run_real_02/node_005`, `run_real_03/node_000`, `run_real_03/node_001`, `run_real_03/node_002`, `run_real_03/node_003`, `run_real_04/node_000`, `run_real_04/node_001`, `run_real_04/node_002`, `run_real_04/node_003`, `run_real_05/node_000`, `run_real_05/node_001`, `run_real_05/node_002`, `run_real_05/node_003`
