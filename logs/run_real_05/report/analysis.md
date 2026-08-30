# Post-hoc analysis: run_real_05

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
| node_001 | duration<=18000ms | 22292 | 0.710404 | 0.390061 | 0.550233 | +0.000011 |  |
| node_001 | duration>18000ms | 102617 | 0.653302 | 0.526457 | 0.589879 | -0.000245 |  |
| node_001 | tab=0 | 92672 | 0.617024 | 0.540164 | 0.578594 | -0.000494 |  |
| node_001 | tab=1 | 13726 | 0.546350 | 0.062122 | 0.304236 | -0.010407 |  |
| node_001 | tab=2 | 547 | 0.324074 | 0.025770 | 0.174922 | -0.178132 |  |
| node_001 | tab=3 | 5170 | 0.639805 | 0.106361 | 0.373083 | +0.019736 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=4 | 3834 | 0.634337 | 0.509924 | 0.572131 | +0.004124 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=5 | 7877 | 0.565165 | 0.546300 | 0.555732 | +0.002316 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=6 | 226 | 0.541228 | 0.166021 | 0.353625 | -0.037095 |  |
| node_001 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_001 | tab=8 | 291 | 0.250000 | 0.106294 | 0.178147 | +0.000000 |  |
| node_001 | tab=9 | 121 | 0.425926 | 0.230535 | 0.328231 | -0.156385 |  |
| node_001 | tab=10 | 92 | 0.635714 | 0.258170 | 0.446942 | +0.090674 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_001 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_001 | tab=13 | 57 | 0.556373 | 0.486934 | 0.521653 | -0.078017 |  |
| node_001 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_001 | history:low<=22 | 26709 | 0.664016 | 0.537039 | 0.600528 | +0.002618 | GATING/ENSEMBLE CANDIDATE |
| node_001 | history:mid(22,52] | 36727 | 0.665532 | 0.553780 | 0.609656 | -0.001750 |  |
| node_001 | history:high>52 | 61473 | 0.669704 | 0.516312 | 0.593008 | -0.001262 |  |
| node_001 | val-list<=5 | 37566 | 0.662251 | 0.526025 | 0.594138 | +0.000356 |  |
| node_001 | val-list>5 | 87343 | 0.668888 | 0.552542 | 0.610715 | -0.000624 |  |
| node_002 | duration<=18000ms | 22292 | 0.713848 | 0.389719 | 0.551784 | +0.001562 |  |
| node_002 | duration>18000ms | 102617 | 0.649873 | 0.526172 | 0.588023 | -0.002102 |  |
| node_002 | tab=0 | 92672 | 0.614885 | 0.538859 | 0.576872 | -0.002217 |  |
| node_002 | tab=1 | 13726 | 0.534077 | 0.062436 | 0.298257 | -0.016386 |  |
| node_002 | tab=2 | 547 | 0.268519 | 0.025251 | 0.146885 | -0.206169 |  |
| node_002 | tab=3 | 5170 | 0.625763 | 0.106100 | 0.365932 | +0.012585 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=4 | 3834 | 0.631668 | 0.512384 | 0.572026 | +0.004019 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=5 | 7877 | 0.583681 | 0.547845 | 0.565763 | +0.012346 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=6 | 226 | 0.538012 | 0.169261 | 0.353636 | -0.037084 |  |
| node_002 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_002 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=9 | 121 | 0.666667 | 0.247010 | 0.456838 | -0.027778 |  |
| node_002 | tab=10 | 92 | 0.521429 | 0.249449 | 0.385439 | +0.029171 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_002 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=13 | 57 | 0.713235 | 0.525417 | 0.619326 | +0.019656 | GATING/ENSEMBLE CANDIDATE |
| node_002 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_002 | history:low<=22 | 26709 | 0.662744 | 0.536334 | 0.599539 | +0.001629 |  |
| node_002 | history:mid(22,52] | 36727 | 0.663798 | 0.552887 | 0.608343 | -0.003063 |  |
| node_002 | history:high>52 | 61473 | 0.668080 | 0.516375 | 0.592227 | -0.002042 |  |
| node_002 | val-list<=5 | 37566 | 0.659682 | 0.526202 | 0.592942 | -0.000840 |  |
| node_002 | val-list>5 | 87343 | 0.667664 | 0.550819 | 0.609241 | -0.002098 |  |
| node_003 | duration<=18000ms | 22292 | 0.707622 | 0.389368 | 0.548495 | -0.001727 |  |
| node_003 | duration>18000ms | 102617 | 0.654865 | 0.528105 | 0.591485 | +0.001360 |  |
| node_003 | tab=0 | 92672 | 0.620053 | 0.541550 | 0.580802 | +0.001713 |  |
| node_003 | tab=1 | 13726 | 0.578216 | 0.063282 | 0.320749 | +0.006106 |  |
| node_003 | tab=2 | 547 | 0.638889 | 0.029096 | 0.333993 | -0.019061 |  |
| node_003 | tab=3 | 5170 | 0.637363 | 0.106135 | 0.371749 | +0.018402 |  |
| node_003 | tab=4 | 3834 | 0.621489 | 0.511138 | 0.566313 | -0.001693 |  |
| node_003 | tab=5 | 7877 | 0.573846 | 0.547515 | 0.560681 | +0.007264 |  |
| node_003 | tab=6 | 226 | 0.528070 | 0.169767 | 0.348919 | -0.041801 |  |
| node_003 | tab=7 | 177 | 0.500000 | 0.006173 | 0.253086 | +0.000000 |  |
| node_003 | tab=8 | 291 | 0.500000 | 0.109622 | 0.304811 | +0.126664 |  |
| node_003 | tab=9 | 121 | 0.703704 | 0.248110 | 0.475907 | -0.008709 |  |
| node_003 | tab=10 | 92 | 0.535714 | 0.243497 | 0.389606 | +0.033338 |  |
| node_003 | tab=11 | 84 | 0.607143 | 0.279866 | 0.443504 | +0.000000 |  |
| node_003 | tab=12 | 30 | 0.833333 | 0.665118 | 0.749225 | +0.338920 |  |
| node_003 | tab=13 | 57 | 0.583333 | 0.465627 | 0.524480 | -0.075190 |  |
| node_003 | tab=14 | 5 | 0.500000 | 0.200000 | 0.350000 | +0.000000 |  |
| node_003 | history:low<=22 | 26709 | 0.660942 | 0.536528 | 0.598735 | +0.000825 |  |
| node_003 | history:mid(22,52] | 36727 | 0.669185 | 0.555613 | 0.612399 | +0.000993 |  |
| node_003 | history:high>52 | 61473 | 0.673449 | 0.520080 | 0.596764 | +0.002495 |  |
| node_003 | val-list<=5 | 37566 | 0.662857 | 0.527161 | 0.595009 | +0.001227 |  |
| node_003 | val-list>5 | 87343 | 0.671715 | 0.555142 | 0.613429 | +0.002090 |  |

## Ranking change vs parent

| node | parent | mean Kendall tau-b | top-1 changed | primary delta | classification |
| --- | --- | --- | --- | --- | --- |
| node_001 | node_000 | 0.755985 | 22.84% | -0.000467 | flat metrics + high change (ensemble candidate) |
| node_002 | node_000 | 0.795858 | 18.80% | -0.001515 | flat metrics + high change (ensemble candidate) |
| node_003 | node_000 | 0.850868 | 14.43% | +0.001485 | flat metrics + high change (ensemble candidate) |

## Post-hoc ensembles

Best single: `run_real_05/node_003` — primary 0.603337.

| ensemble | members | GAUC | nDCG@5 | primary | delta vs best | flag |
| --- | --- | --- | --- | --- | --- | --- |
| all accepted nodes | 2 | 0.668960 | 0.537098 | 0.603029 | -0.000309 |  |
| best + high-change rejected nodes | 3 | 0.669539 | 0.537156 | 0.603347 | +0.000010 |  |
| all nodes | 4 | 0.670047 | 0.537072 | 0.603560 | +0.000223 |  |

Members:
- all accepted nodes: `run_real_05/node_000`, `run_real_05/node_003`
- best + high-change rejected nodes: `run_real_05/node_003`, `run_real_05/node_001`, `run_real_05/node_002`
- all nodes: `run_real_05/node_000`, `run_real_05/node_001`, `run_real_05/node_002`, `run_real_05/node_003`

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
