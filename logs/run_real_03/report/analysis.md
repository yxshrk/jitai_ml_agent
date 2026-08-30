# Post-hoc analysis: run_real_03

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
| node_001 | duration<=18000ms | 22292 | 0.703787 | 0.389666 | 0.546727 | -0.003496 |  |
| node_001 | duration>18000ms | 102617 | 0.649007 | 0.525385 | 0.587196 | -0.002929 |  |
| node_001 | tab=0 | 92672 | 0.610778 | 0.537810 | 0.574294 | -0.004795 |  |
| node_001 | tab=1 | 13726 | 0.553985 | 0.062787 | 0.308386 | -0.006256 |  |
| node_001 | tab=2 | 547 | 0.212963 | 0.024551 | 0.118757 | -0.234297 |  |
| node_001 | tab=3 | 5170 | 0.645299 | 0.106489 | 0.375894 | +0.022547 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=4 | 3834 | 0.617394 | 0.510381 | 0.563888 | -0.004119 |  |
| node_001 | tab=5 | 7877 | 0.571659 | 0.546456 | 0.559057 | +0.005640 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=6 | 226 | 0.662573 | 0.175898 | 0.419236 | +0.028515 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_001 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=9 | 121 | 0.490741 | 0.229435 | 0.360088 | -0.124528 |  |
| node_001 | tab=10 | 92 | 0.550000 | 0.252080 | 0.401040 | +0.044772 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_001 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=13 | 57 | 0.593137 | 0.498835 | 0.545986 | -0.053684 |  |
| node_001 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_001 | history:low<=22 | 26709 | 0.655215 | 0.534888 | 0.595051 | -0.002858 |  |
| node_001 | history:mid(22,52] | 36727 | 0.662156 | 0.552341 | 0.607249 | -0.004157 |  |
| node_001 | history:high>52 | 61473 | 0.666969 | 0.514700 | 0.590834 | -0.003435 |  |
| node_001 | val-list<=5 | 37566 | 0.656415 | 0.525799 | 0.591107 | -0.002675 |  |
| node_001 | val-list>5 | 87343 | 0.665202 | 0.548142 | 0.606672 | -0.004667 |  |
| node_002 | duration<=18000ms | 22292 | 0.711341 | 0.390431 | 0.550886 | +0.000664 |  |
| node_002 | duration>18000ms | 102617 | 0.655063 | 0.527832 | 0.591448 | +0.001323 |  |
| node_002 | tab=0 | 92672 | 0.619874 | 0.540934 | 0.580404 | +0.001316 |  |
| node_002 | tab=1 | 13726 | 0.564378 | 0.062711 | 0.313544 | -0.001098 |  |
| node_002 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_002 | tab=3 | 5170 | 0.666056 | 0.106460 | 0.386258 | +0.032911 |  |
| node_002 | tab=4 | 3834 | 0.622078 | 0.510866 | 0.566472 | -0.001535 |  |
| node_002 | tab=5 | 7877 | 0.575977 | 0.548144 | 0.562061 | +0.008644 |  |
| node_002 | tab=6 | 226 | 0.612573 | 0.173039 | 0.392806 | +0.002086 |  |
| node_002 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_002 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 |  |
| node_002 | tab=9 | 121 | 0.648148 | 0.242252 | 0.445200 | -0.039416 |  |
| node_002 | tab=10 | 92 | 0.664286 | 0.251316 | 0.457801 | +0.101533 |  |
| node_002 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_002 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 |  |
| node_002 | tab=13 | 57 | 0.495098 | 0.465627 | 0.480362 | -0.119308 |  |
| node_002 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_002 | history:low<=22 | 26709 | 0.665870 | 0.537786 | 0.601828 | +0.003919 |  |
| node_002 | history:mid(22,52] | 36727 | 0.667124 | 0.555392 | 0.611258 | -0.000148 |  |
| node_002 | history:high>52 | 61473 | 0.672826 | 0.518436 | 0.595631 | +0.001361 |  |
| node_002 | val-list<=5 | 37566 | 0.664667 | 0.526810 | 0.595738 | +0.001956 |  |
| node_002 | val-list>5 | 87343 | 0.671218 | 0.555253 | 0.613236 | +0.001897 |  |
| node_003 | duration<=18000ms | 22292 | 0.713254 | 0.390916 | 0.552085 | +0.001863 |  |
| node_003 | duration>18000ms | 102617 | 0.654888 | 0.527671 | 0.591279 | +0.001154 |  |
| node_003 | tab=0 | 92672 | 0.620437 | 0.541336 | 0.580886 | +0.001798 |  |
| node_003 | tab=1 | 13726 | 0.561221 | 0.062552 | 0.311886 | -0.002756 |  |
| node_003 | tab=2 | 547 | 0.333333 | 0.027037 | 0.180185 | -0.172869 |  |
| node_003 | tab=3 | 5170 | 0.692918 | 0.106643 | 0.399781 | +0.046434 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=4 | 3834 | 0.640151 | 0.515475 | 0.577813 | +0.009806 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=5 | 7877 | 0.572338 | 0.547311 | 0.559825 | +0.006408 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=6 | 226 | 0.588304 | 0.166608 | 0.377456 | -0.013264 |  |
| node_003 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_003 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=9 | 121 | 0.592593 | 0.242252 | 0.417422 | -0.067194 |  |
| node_003 | tab=10 | 92 | 0.657143 | 0.251316 | 0.454229 | +0.097961 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_003 | tab=12 | 30 | 0.666667 | 0.665118 | 0.665892 | +0.255587 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=13 | 57 | 0.502451 | 0.475719 | 0.489085 | -0.110585 |  |
| node_003 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_003 | history:low<=22 | 26709 | 0.664369 | 0.537134 | 0.600751 | +0.002842 | GATING/ENSEMBLE CANDIDATE |
| node_003 | history:mid(22,52] | 36727 | 0.666882 | 0.555507 | 0.611194 | -0.000211 |  |
| node_003 | history:high>52 | 61473 | 0.673715 | 0.518982 | 0.596348 | +0.002078 | GATING/ENSEMBLE CANDIDATE |
| node_003 | val-list<=5 | 37566 | 0.663744 | 0.526906 | 0.595325 | +0.001543 |  |
| node_003 | val-list>5 | 87343 | 0.671559 | 0.555068 | 0.613313 | +0.001975 |  |

## Ranking change vs parent

| node | parent | mean Kendall tau-b | top-1 changed | primary delta | classification |
| --- | --- | --- | --- | --- | --- |
| node_001 | node_000 | 0.752684 | 22.37% | -0.003466 | metric change + high ranking change |
| node_002 | node_000 | 0.799863 | 20.07% | +0.001452 | flat metrics + high change (ensemble candidate) |
| node_003 | node_002 | 0.876869 | 12.95% | -0.000001 | flat metrics + high change (ensemble candidate) |

## Post-hoc ensembles

Best single: `run_real_03/node_002` — primary 0.603304.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| all accepted nodes | 2 | 0.669930 | 0.537020 | 0.603475 | +0.000171 |  |
| best + high-change rejected nodes | 3 | 0.669407 | 0.537274 | 0.603341 | +0.000036 |  |
| all nodes | 4 | 0.670244 | 0.537351 | 0.603797 | +0.000493 |  |

Members:
- all accepted nodes: `run_real_03/node_000`, `run_real_03/node_002`
- best + high-change rejected nodes: `run_real_03/node_002`, `run_real_03/node_001`, `run_real_03/node_003`
- all nodes: `run_real_03/node_000`, `run_real_03/node_001`, `run_real_03/node_002`, `run_real_03/node_003`

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
