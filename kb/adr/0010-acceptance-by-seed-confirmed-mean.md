# ADR-0010 — Accept on the seed-confirmed mean, not on the best single seed
Status: Accepted (2026-08-30), measured in live_01

## Context
Selecting the best of k single-seed branches is biased upward (the winner's curse). Measured on the first live run:
the BPR node scored +0.0022 on seed 0 and was accepted under the single-seed ε rule; over seeds {0, 1, 2} it averaged
0.60316 vs the baseline's 0.60144 — a real but smaller +0.0017. Every top-3 node shrank on re-seeding
(0.60392 → 0.60319, 0.60365 → 0.60316, 0.60314 → 0.60260). The organizers' ε = 0.002 is calibrated for one seed
with std 0.0008; with three seeds per side the standard error of the difference is ~0.0003, so a fixed 0.002 floor on
the mean would also reject real improvements.

## Decision
- Every candidate with a positive single-seed delta is re-run with CONFIRM_SEEDS = 2 extra seeds (the champion's
  seeds are cached); seed re-runs are evaluation replicates, not iterations.
- Accept (and make champion) iff the difference of seed means is ≥ MIN_EFFECT = 0.001 **and** ≥ T_CRIT = 2.5
  standard errors (per-node std floored at 0.0002 because three seeds is a small sample).
- The official convergence rule is unchanged: single-seed best-so-far must improve by > 0.002 within N = 3
  generations. Acceptance governs which node becomes the parent; convergence governs when the run stops.
- The final designation re-ranks the top-3 by seed mean (already in place).

## Consequences
+30–60 s per positive candidate at FM scale; champions are real improvements; the journal records
`single_seed_accept` and `seed_confirmation` {seeds, means, delta, se, t} so the effect can be reported.
