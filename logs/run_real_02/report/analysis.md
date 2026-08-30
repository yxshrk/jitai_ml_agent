# Post-hoc analysis: run_real_02

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
| node_001 | duration<=18000ms | 22292 | 0.713835 | 0.389962 | 0.551899 | +0.001676 |  |
| node_001 | duration>18000ms | 102617 | 0.649808 | 0.526316 | 0.588062 | -0.002063 |  |
| node_001 | tab=0 | 92672 | 0.614698 | 0.539612 | 0.577155 | -0.001933 |  |
| node_001 | tab=1 | 13726 | 0.544777 | 0.062244 | 0.303511 | -0.011132 |  |
| node_001 | tab=2 | 547 | 0.268519 | 0.025251 | 0.146885 | -0.206169 |  |
| node_001 | tab=3 | 5170 | 0.691697 | 0.106726 | 0.399212 | +0.045865 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=4 | 3834 | 0.628348 | 0.510247 | 0.569297 | +0.001291 |  |
| node_001 | tab=5 | 7877 | 0.571263 | 0.546725 | 0.558994 | +0.005578 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=6 | 226 | 0.538012 | 0.169261 | 0.353636 | -0.037084 |  |
| node_001 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_001 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_001 | tab=9 | 121 | 0.777778 | 0.252868 | 0.515323 | +0.030707 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=10 | 92 | 0.421429 | 0.246542 | 0.333985 | -0.022283 |  |
| node_001 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_001 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=13 | 57 | 0.696078 | 0.498835 | 0.597457 | -0.002214 |  |
| node_001 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_001 | history:low<=22 | 26709 | 0.659752 | 0.535792 | 0.597772 | -0.000138 |  |
| node_001 | history:mid(22,52] | 36727 | 0.663102 | 0.553669 | 0.608385 | -0.003020 |  |
| node_001 | history:high>52 | 61473 | 0.668323 | 0.515489 | 0.591906 | -0.002364 |  |
| node_001 | val-list<=5 | 37566 | 0.659872 | 0.526209 | 0.593040 | -0.000742 |  |
| node_001 | val-list>5 | 87343 | 0.666555 | 0.550189 | 0.608372 | -0.002967 |  |
| node_002 | duration<=18000ms | 22292 | 0.710918 | 0.390233 | 0.550575 | +0.000353 |  |
| node_002 | duration>18000ms | 102617 | 0.656253 | 0.527940 | 0.592097 | +0.001972 |  |
| node_002 | tab=0 | 92672 | 0.620513 | 0.541314 | 0.580913 | +0.001825 |  |
| node_002 | tab=1 | 13726 | 0.562828 | 0.062291 | 0.312560 | -0.002083 |  |
| node_002 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_002 | tab=3 | 5170 | 0.629426 | 0.105941 | 0.367683 | +0.014337 |  |
| node_002 | tab=4 | 3834 | 0.631795 | 0.512759 | 0.572277 | +0.004270 |  |
| node_002 | tab=5 | 7877 | 0.576226 | 0.547556 | 0.561891 | +0.008474 |  |
| node_002 | tab=6 | 226 | 0.665789 | 0.177186 | 0.421488 | +0.030768 |  |
| node_002 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_002 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 |  |
| node_002 | tab=9 | 121 | 0.481481 | 0.236393 | 0.358937 | -0.125678 |  |
| node_002 | tab=10 | 92 | 0.514286 | 0.249449 | 0.381867 | +0.025599 |  |
| node_002 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_002 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 |  |
| node_002 | tab=13 | 57 | 0.585784 | 0.488743 | 0.537264 | -0.062407 |  |
| node_002 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_002 | history:low<=22 | 26709 | 0.666718 | 0.537984 | 0.602351 | +0.004442 |  |
| node_002 | history:mid(22,52] | 36727 | 0.669886 | 0.556216 | 0.613051 | +0.001645 |  |
| node_002 | history:high>52 | 61473 | 0.672829 | 0.518586 | 0.595707 | +0.001438 |  |
| node_002 | val-list<=5 | 37566 | 0.666459 | 0.527389 | 0.596924 | +0.003142 |  |
| node_002 | val-list>5 | 87343 | 0.672037 | 0.555301 | 0.613669 | +0.002330 |  |
| node_003 | duration<=18000ms | 22292 | 0.720507 | 0.390859 | 0.555683 | +0.005461 | GATING/ENSEMBLE CANDIDATE |
| node_003 | duration>18000ms | 102617 | 0.656208 | 0.528098 | 0.592153 | +0.002028 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=0 | 92672 | 0.620722 | 0.541433 | 0.581077 | +0.001989 |  |
| node_003 | tab=1 | 13726 | 0.547785 | 0.062021 | 0.304903 | -0.009739 |  |
| node_003 | tab=2 | 547 | 0.388889 | 0.026540 | 0.207715 | -0.145339 |  |
| node_003 | tab=3 | 5170 | 0.607448 | 0.105817 | 0.356633 | +0.003286 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=4 | 3834 | 0.636458 | 0.513629 | 0.575044 | +0.007037 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=5 | 7877 | 0.571521 | 0.547243 | 0.559382 | +0.005965 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=6 | 226 | 0.561988 | 0.162322 | 0.362155 | -0.028565 |  |
| node_003 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_003 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=9 | 121 | 0.481481 | 0.236393 | 0.358937 | -0.125678 |  |
| node_003 | tab=10 | 92 | 0.657143 | 0.251316 | 0.454229 | +0.097961 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_003 | tab=12 | 30 | 0.666667 | 0.665118 | 0.665892 | +0.255587 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=13 | 57 | 0.539216 | 0.478671 | 0.508943 | -0.090727 |  |
| node_003 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_003 | history:low<=22 | 26709 | 0.664091 | 0.536997 | 0.600544 | +0.002634 | GATING/ENSEMBLE CANDIDATE |
| node_003 | history:mid(22,52] | 36727 | 0.670230 | 0.556862 | 0.613546 | +0.002141 | GATING/ENSEMBLE CANDIDATE |
| node_003 | history:high>52 | 61473 | 0.673759 | 0.518573 | 0.596166 | +0.001896 |  |
| node_003 | val-list<=5 | 37566 | 0.663246 | 0.526814 | 0.595030 | +0.001248 |  |
| node_003 | val-list>5 | 87343 | 0.673160 | 0.555944 | 0.614552 | +0.003213 | GATING/ENSEMBLE CANDIDATE |
| node_004 | duration<=18000ms | 22292 | 0.713245 | 0.390348 | 0.551796 | +0.001574 |  |
| node_004 | duration>18000ms | 102617 | 0.655434 | 0.528090 | 0.591762 | +0.001637 |  |
| node_004 | tab=0 | 92672 | 0.621017 | 0.541872 | 0.581445 | +0.002356 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=1 | 13726 | 0.558989 | 0.062323 | 0.310656 | -0.003986 |  |
| node_004 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_004 | tab=3 | 5170 | 0.656288 | 0.106353 | 0.381320 | +0.027974 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=4 | 3834 | 0.631483 | 0.513921 | 0.572702 | +0.004696 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=5 | 7877 | 0.571480 | 0.546998 | 0.559239 | +0.005822 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=6 | 226 | 0.630702 | 0.173829 | 0.402266 | +0.011546 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_004 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=9 | 121 | 0.592593 | 0.238472 | 0.415532 | -0.069084 |  |
| node_004 | tab=10 | 92 | 0.521429 | 0.249449 | 0.385439 | +0.029171 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_004 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=13 | 57 | 0.607843 | 0.488743 | 0.548293 | -0.051377 |  |
| node_004 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_004 | history:low<=22 | 26709 | 0.666847 | 0.537989 | 0.602418 | +0.004509 | GATING/ENSEMBLE CANDIDATE |
| node_004 | history:mid(22,52] | 36727 | 0.669996 | 0.557337 | 0.613667 | +0.002261 | GATING/ENSEMBLE CANDIDATE |
| node_004 | history:high>52 | 61473 | 0.672581 | 0.518149 | 0.595365 | +0.001095 |  |
| node_004 | val-list<=5 | 37566 | 0.666748 | 0.527656 | 0.597202 | +0.003420 | GATING/ENSEMBLE CANDIDATE |
| node_004 | val-list>5 | 87343 | 0.671863 | 0.555447 | 0.613655 | +0.002316 | GATING/ENSEMBLE CANDIDATE |
| node_005 | duration<=18000ms | 22292 | 0.710801 | 0.390188 | 0.550494 | +0.000272 |  |
| node_005 | duration>18000ms | 102617 | 0.656044 | 0.527883 | 0.591964 | +0.001839 |  |
| node_005 | tab=0 | 92672 | 0.620362 | 0.541298 | 0.580830 | +0.001741 |  |
| node_005 | tab=1 | 13726 | 0.562828 | 0.062291 | 0.312560 | -0.002083 |  |
| node_005 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_005 | tab=3 | 5170 | 0.629426 | 0.105941 | 0.367683 | +0.014337 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=4 | 3834 | 0.631240 | 0.512914 | 0.572077 | +0.004070 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=5 | 7877 | 0.576125 | 0.547606 | 0.561866 | +0.008449 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=6 | 226 | 0.665789 | 0.177186 | 0.421488 | +0.030768 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_005 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=9 | 121 | 0.481481 | 0.236393 | 0.358937 | -0.125678 |  |
| node_005 | tab=10 | 92 | 0.507143 | 0.240452 | 0.373797 | +0.017530 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=11 | 84 | 0.535714 | 0.279866 | 0.407790 | -0.035714 |  |
| node_005 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=13 | 57 | 0.585784 | 0.488743 | 0.537264 | -0.062407 |  |
| node_005 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_005 | history:low<=22 | 26709 | 0.666455 | 0.537920 | 0.602188 | +0.004278 | GATING/ENSEMBLE CANDIDATE |
| node_005 | history:mid(22,52] | 36727 | 0.669771 | 0.556225 | 0.612998 | +0.001592 |  |
| node_005 | history:high>52 | 61473 | 0.672759 | 0.518534 | 0.595647 | +0.001377 |  |
| node_005 | val-list<=5 | 37566 | 0.666260 | 0.527338 | 0.596799 | +0.003016 | GATING/ENSEMBLE CANDIDATE |
| node_005 | val-list>5 | 87343 | 0.671937 | 0.555291 | 0.613614 | +0.002275 | GATING/ENSEMBLE CANDIDATE |

## Ranking change vs parent

| node | parent | mean Kendall tau-b | top-1 changed | primary delta | classification |
| --- | --- | --- | --- | --- | --- |
| node_001 | node_000 | 0.775072 | 20.47% | -0.002008 | metric change + high ranking change |
| node_002 | node_000 | 0.801210 | 19.24% | +0.002184 | metric change + high ranking change |
| node_003 | node_002 | 0.871517 | 13.55% | -0.000082 | flat metrics + high change (ensemble candidate) |
| node_004 | node_002 | 0.919867 | 8.28% | +0.000086 | flat metrics + low change (idea changed nothing) |
| node_005 | node_002 | 0.998689 | 0.15% | -0.000081 | flat metrics + low change (idea changed nothing) |

## Post-hoc ensembles

Best single: `run_real_02/node_004` — primary 0.604123.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| all accepted nodes | 2 | 0.670171 | 0.536927 | 0.603549 | -0.000574 |  |
| best + high-change rejected nodes | 3 | 0.670450 | 0.537628 | 0.604039 | -0.000084 |  |
| all nodes | 6 | 0.670831 | 0.537345 | 0.604088 | -0.000035 |  |

Members:
- all accepted nodes: `run_real_02/node_000`, `run_real_02/node_002`
- best + high-change rejected nodes: `run_real_02/node_004`, `run_real_02/node_001`, `run_real_02/node_003`
- all nodes: `run_real_02/node_000`, `run_real_02/node_001`, `run_real_02/node_002`, `run_real_02/node_003`, `run_real_02/node_004`, `run_real_02/node_005`

## Curve stats

Train-loss slope is an ordinary least-squares slope over the final three usable epochs.

| node | val peak epoch | val peak | final val | peak-final | train-loss slope at stop | note |
| --- | --- | --- | --- | --- | --- | --- |
| node_000 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_001 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_002 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_003 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_004 | N/A | N/A | N/A | N/A | N/A | history unavailable |
| node_005 | N/A | N/A | N/A | N/A | N/A | history unavailable |

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
