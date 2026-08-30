# Heterogeneous blend audit (evidence only — NOT a designation candidate)

Protocol: four existing champion prediction sets (different objectives: designated
ensemble; temporal-pair-kernel; decayed-positive-sampling; gauge-fixed-BCE), all
validation-aligned. TWO blend rules predeclared before any evaluation; each
evaluated exactly once on official validation; no weight search. Team-assembled,
therefore disclosed as evidence under our designation ruling.

| component | objective family | valid primary |
|---|---|---|
| champ (bc07) | package + seed rank-ensemble | 0.605575 |
| tkern (novel_l1) | + temporal pair kernel | 0.605146 |
| dsamp (qb_b) | decayed-positive sampling | 0.604657 |
| gbce (novel_r1) | gauge-fixed BCE | 0.604479 |

Pairwise vs champion: midrank corr 0.91-0.95; net-correction (GAUC-weighted
rescue minus harm) NEGATIVE for each alone (-0.0006..-0.0020).

| predeclared blend | valid primary |
|---|---|
| A: equal-weight 4-way midrank | 0.605639 |
| B: champion-anchored 0.6/0.4 | **0.605745** |

Reading: individually-harmful but mutually-decorrelated dissenters net +0.00017
when blended — the theoretically expected sign and an honestly noise-class size.
Consistent with the campaign's sub-floor ceiling cluster; quantifies what
validation-tuned blend selection would inflate. Not designated; the designated
submission remains run_bigclock_07's converged checkpoint.
