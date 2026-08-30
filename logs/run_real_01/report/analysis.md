# Post-hoc analysis: run_real_01

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
| node_001 | duration<=18000ms | 22292 | 0.709715 | 0.389645 | 0.549680 | -0.000542 |  |
| node_001 | duration>18000ms | 102617 | 0.649698 | 0.526227 | 0.587962 | -0.002163 |  |
| node_001 | tab=0 | 92672 | 0.614313 | 0.539329 | 0.576821 | -0.002267 |  |
| node_001 | tab=1 | 13726 | 0.539216 | 0.062207 | 0.300711 | -0.013931 |  |
| node_001 | tab=2 | 547 | 0.435185 | 0.026722 | 0.230953 | -0.122101 |  |
| node_001 | tab=3 | 5170 | 0.657509 | 0.106144 | 0.381826 | +0.028480 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=4 | 3834 | 0.621192 | 0.509740 | 0.565466 | -0.002540 |  |
| node_001 | tab=5 | 7877 | 0.577095 | 0.548094 | 0.562595 | +0.009178 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=6 | 226 | 0.557602 | 0.174343 | 0.365972 | -0.024748 |  |
| node_001 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_001 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_001 | tab=9 | 121 | 0.666667 | 0.247010 | 0.456838 | -0.027778 |  |
| node_001 | tab=10 | 92 | 0.421429 | 0.249862 | 0.335645 | -0.020622 |  |
| node_001 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_001 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=13 | 57 | 0.710784 | 0.498835 | 0.604810 | +0.005139 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_001 | history:low<=22 | 26709 | 0.660097 | 0.535609 | 0.597853 | -0.000056 |  |
| node_001 | history:mid(22,52] | 36727 | 0.662971 | 0.553832 | 0.608401 | -0.003004 |  |
| node_001 | history:high>52 | 61473 | 0.667992 | 0.515944 | 0.591968 | -0.002302 |  |
| node_001 | val-list<=5 | 37566 | 0.658632 | 0.526424 | 0.592528 | -0.001255 |  |
| node_001 | val-list>5 | 87343 | 0.666842 | 0.550201 | 0.608522 | -0.002817 |  |
| node_002 | duration<=18000ms | 22292 | 0.715426 | 0.390501 | 0.552964 | +0.002742 |  |
| node_002 | duration>18000ms | 102617 | 0.655324 | 0.528332 | 0.591828 | +0.001703 |  |
| node_002 | tab=0 | 92672 | 0.620512 | 0.541393 | 0.580952 | +0.001864 |  |
| node_002 | tab=1 | 13726 | 0.566171 | 0.062960 | 0.314566 | -0.000077 |  |
| node_002 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_002 | tab=3 | 5170 | 0.663614 | 0.106353 | 0.384983 | +0.031637 |  |
| node_002 | tab=4 | 3834 | 0.636609 | 0.513780 | 0.575194 | +0.007188 |  |
| node_002 | tab=5 | 7877 | 0.576146 | 0.548082 | 0.562114 | +0.008697 |  |
| node_002 | tab=6 | 226 | 0.632456 | 0.176965 | 0.404711 | +0.013991 |  |
| node_002 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_002 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 |  |
| node_002 | tab=9 | 121 | 0.444444 | 0.230535 | 0.337490 | -0.147126 |  |
| node_002 | tab=10 | 92 | 0.535714 | 0.250468 | 0.393091 | +0.036823 |  |
| node_002 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_002 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 |  |
| node_002 | tab=13 | 57 | 0.622549 | 0.488743 | 0.555646 | -0.044025 |  |
| node_002 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_002 | history:low<=22 | 26709 | 0.665022 | 0.536328 | 0.600675 | +0.002765 |  |
| node_002 | history:mid(22,52] | 36727 | 0.669852 | 0.556643 | 0.613247 | +0.001842 |  |
| node_002 | history:high>52 | 61473 | 0.673885 | 0.520245 | 0.597065 | +0.002795 |  |
| node_002 | val-list<=5 | 37566 | 0.666468 | 0.527312 | 0.596890 | +0.003107 |  |
| node_002 | val-list>5 | 87343 | 0.672178 | 0.555764 | 0.613971 | +0.002632 |  |
| node_003 | duration<=18000ms | 22292 | 0.717993 | 0.390773 | 0.554383 | +0.004161 | GATING/ENSEMBLE CANDIDATE |
| node_003 | duration>18000ms | 102617 | 0.655802 | 0.528054 | 0.591928 | +0.001803 |  |
| node_003 | tab=0 | 92672 | 0.621205 | 0.541352 | 0.581278 | +0.002190 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=1 | 13726 | 0.542828 | 0.062133 | 0.302480 | -0.012162 |  |
| node_003 | tab=2 | 547 | 0.296296 | 0.025951 | 0.161124 | -0.191930 |  |
| node_003 | tab=3 | 5170 | 0.615995 | 0.105746 | 0.360871 | +0.007524 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=4 | 3834 | 0.631372 | 0.514103 | 0.572737 | +0.004731 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=5 | 7877 | 0.572151 | 0.547226 | 0.559688 | +0.006272 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=6 | 226 | 0.595906 | 0.170241 | 0.383074 | -0.007646 |  |
| node_003 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_003 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=9 | 121 | 0.481481 | 0.236393 | 0.358937 | -0.125678 |  |
| node_003 | tab=10 | 92 | 0.471429 | 0.240452 | 0.355940 | -0.000328 |  |
| node_003 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_003 | tab=12 | 30 | 0.666667 | 0.665118 | 0.665892 | +0.255587 | GATING/ENSEMBLE CANDIDATE |
| node_003 | tab=13 | 57 | 0.546569 | 0.478671 | 0.512620 | -0.087051 |  |
| node_003 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_003 | history:low<=22 | 26709 | 0.665570 | 0.537380 | 0.601475 | +0.003566 | GATING/ENSEMBLE CANDIDATE |
| node_003 | history:mid(22,52] | 36727 | 0.670661 | 0.556799 | 0.613730 | +0.002324 | GATING/ENSEMBLE CANDIDATE |
| node_003 | history:high>52 | 61473 | 0.673380 | 0.518543 | 0.595962 | +0.001692 |  |
| node_003 | val-list<=5 | 37566 | 0.664622 | 0.526926 | 0.595774 | +0.001992 |  |
| node_003 | val-list>5 | 87343 | 0.673051 | 0.556026 | 0.614539 | +0.003200 | GATING/ENSEMBLE CANDIDATE |
| node_004 | duration<=18000ms | 22292 | 0.714147 | 0.390299 | 0.552223 | +0.002001 | GATING/ENSEMBLE CANDIDATE |
| node_004 | duration>18000ms | 102617 | 0.655709 | 0.527690 | 0.591700 | +0.001575 |  |
| node_004 | tab=0 | 92672 | 0.620491 | 0.541011 | 0.580751 | +0.001662 |  |
| node_004 | tab=1 | 13726 | 0.545010 | 0.062121 | 0.303565 | -0.011077 |  |
| node_004 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_004 | tab=3 | 5170 | 0.635531 | 0.106135 | 0.370833 | +0.017486 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=4 | 3834 | 0.624405 | 0.512499 | 0.568452 | +0.000446 |  |
| node_004 | tab=5 | 7877 | 0.574775 | 0.547023 | 0.560899 | +0.007482 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=6 | 226 | 0.655263 | 0.176795 | 0.416029 | +0.025309 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_004 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=9 | 121 | 0.444444 | 0.230535 | 0.337490 | -0.147126 |  |
| node_004 | tab=10 | 92 | 0.521429 | 0.249449 | 0.385439 | +0.029171 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_004 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_004 | tab=13 | 57 | 0.549020 | 0.488743 | 0.518881 | -0.080789 |  |
| node_004 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_004 | history:low<=22 | 26709 | 0.666455 | 0.537302 | 0.601878 | +0.003969 | GATING/ENSEMBLE CANDIDATE |
| node_004 | history:mid(22,52] | 36727 | 0.667780 | 0.555261 | 0.611520 | +0.000115 |  |
| node_004 | history:high>52 | 61473 | 0.673439 | 0.518415 | 0.595927 | +0.001657 |  |
| node_004 | val-list<=5 | 37566 | 0.663952 | 0.526551 | 0.595252 | +0.001469 |  |
| node_004 | val-list>5 | 87343 | 0.672323 | 0.555112 | 0.613718 | +0.002379 | GATING/ENSEMBLE CANDIDATE |
| node_005 | duration<=18000ms | 22292 | 0.713557 | 0.390307 | 0.551932 | +0.001710 |  |
| node_005 | duration>18000ms | 102617 | 0.655480 | 0.528155 | 0.591818 | +0.001693 |  |
| node_005 | tab=0 | 92672 | 0.619952 | 0.541039 | 0.580496 | +0.001407 |  |
| node_005 | tab=1 | 13726 | 0.565633 | 0.063040 | 0.314336 | -0.000306 |  |
| node_005 | tab=2 | 547 | 0.324074 | 0.025951 | 0.175013 | -0.178041 |  |
| node_005 | tab=3 | 5170 | 0.644078 | 0.106159 | 0.375118 | +0.021772 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=4 | 3834 | 0.625274 | 0.512252 | 0.568763 | +0.000757 |  |
| node_005 | tab=5 | 7877 | 0.571258 | 0.546962 | 0.559110 | +0.005693 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=6 | 226 | 0.585088 | 0.170241 | 0.377664 | -0.013056 |  |
| node_005 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_005 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=9 | 121 | 0.444444 | 0.230535 | 0.337490 | -0.147126 |  |
| node_005 | tab=10 | 92 | 0.535714 | 0.250468 | 0.393091 | +0.036823 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_005 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_005 | tab=13 | 57 | 0.556373 | 0.488743 | 0.522558 | -0.077113 |  |
| node_005 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_005 | history:low<=22 | 26709 | 0.665098 | 0.536923 | 0.601010 | +0.003101 | GATING/ENSEMBLE CANDIDATE |
| node_005 | history:mid(22,52] | 36727 | 0.667821 | 0.555536 | 0.611678 | +0.000273 |  |
| node_005 | history:high>52 | 61473 | 0.673762 | 0.519034 | 0.596398 | +0.002129 | GATING/ENSEMBLE CANDIDATE |
| node_005 | val-list<=5 | 37566 | 0.665726 | 0.526822 | 0.596274 | +0.002492 | GATING/ENSEMBLE CANDIDATE |
| node_005 | val-list>5 | 87343 | 0.671499 | 0.555089 | 0.613294 | +0.001955 |  |

## Ranking change vs parent

| node | parent | mean Kendall tau-b | top-1 changed | primary delta | classification |
| --- | --- | --- | --- | --- | --- |
| node_001 | node_000 | 0.777130 | 20.14% | -0.001997 | flat metrics + high change (ensemble candidate) |
| node_002 | node_000 | 0.805569 | 19.58% | +0.002296 | metric change + high ranking change |
| node_003 | node_002 | 0.869864 | 13.23% | -0.000001 | flat metrics + high change (ensemble candidate) |
| node_004 | node_002 | 0.896695 | 9.96% | -0.000642 | flat metrics + high change (ensemble candidate) |
| node_005 | node_002 | 0.915680 | 8.01% | -0.000626 | flat metrics + low change (idea changed nothing) |

## Post-hoc ensembles

Best single: `run_real_01/node_002` — primary 0.604149.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| all accepted nodes | 2 | 0.669996 | 0.536796 | 0.603396 | -0.000753 |  |
| best + high-change rejected nodes | 4 | 0.670716 | 0.537324 | 0.604020 | -0.000129 |  |
| all nodes | 6 | 0.670738 | 0.537204 | 0.603971 | -0.000178 |  |

Members:
- all accepted nodes: `run_real_01/node_000`, `run_real_01/node_002`
- best + high-change rejected nodes: `run_real_01/node_002`, `run_real_01/node_001`, `run_real_01/node_003`, `run_real_01/node_004`
- all nodes: `run_real_01/node_000`, `run_real_01/node_001`, `run_real_01/node_002`, `run_real_01/node_003`, `run_real_01/node_004`, `run_real_01/node_005`

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
