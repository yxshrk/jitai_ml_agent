# ADR-0020 — A run's decisions rest on the run's own seed evidence
Status: Accepted (2026-08-31), replayed on live_08 and live_09

## Context
`pooled_sigma` blended a prior on the seed-to-seed SD of the validation primary — `SEED_SD_PRIOR = 0.0003` measured
in live_01/02, weighted at `SEED_SD_PRIOR_DF = 4` degrees of freedom — into the pooled seed SD that every acceptance
z-test uses. That prior is knowledge of seeds from *other* runs entering a decision *inside* a run.

The project is judged on the agent, not on a lucky run. If what a run accepts depends on statistics carried in from
earlier runs, then the submitted model is partly the product of the operator's run history rather than of one
autonomous search, it is harder to explain, and there is no reason to expect the borrowed number to hold on the
hidden test split. The organizers do not forbid it; it is a self-imposed standard of the same kind as ADR-0005's
test firewall.

Measured: the prior sat *below* every run's true within-run seed SD (0.00035–0.00044 measured across live_04–09), so
blending it shrank sigma and inflated every z. Replaying every seed confirmation of live_08 and live_09 with the
prior removed (exact arithmetic: the recorded `sigma_pooled` and `sigma_df` determine the within-run sum of squares):

| run | node | z with prior | z without | verdict |
|---|---|---|---|---|
| live_08 | 001 BPR | 3.73 | 3.37 | accepted either way |
| live_08 | **002 embedding L2** | 3.28 | **2.97** | **accepted → rejected** (already at MAX_CONFIRM_SEEDS) |
| live_08 | 003 seed average | 4.05 | 3.75 | accepted either way |
| live_08 | 005, 006, 007, 008 | 3.25–5.58 | 3.13–5.43 | accepted either way |
| live_09 | 001 BPR | 6.04 | 5.48 | accepted either way |
| live_09 | **002 embedding L2** | 3.28 | **2.97** | **accepted → rejected** |
| live_09 | 003 seed average | 3.30 | 3.10 | accepted either way |
| live_09 | 014 champion | 4.58 | 4.43 | accepted either way |
| both | every rejected node | — | — | rejected either way |

One marginal acceptance per run, and nothing else: neither run's champion, designated node or submission changes
(live_09's node_014 descends from node_001, live_08's node_006 from node_003 — both comfortably confirmed without
the prior).

## Decision
- `pooled_sigma(samples)` pools **only this run's** nodes with ≥ 2 fresh seeds; the prior and its constants are
  deleted. When no node yet has two fresh seeds it returns `None` and the candidate's own SD stands in, journalled as
  `sigma_from_node_only` exactly as before.
- `SEED_SD` survives for knowledge-base bookkeeping only (the ledger's mapping-violation margin, ADR-0018). No
  decision inside a run may read it.
- `rules_text()` states that the SD is pooled over this run and no other, so the roles read what the code does.
- The standard generalises beyond seeds: **the submission is the designated node of one run**, chosen by that run's
  own rules from evidence it gathered in its own iterations. Where a run must be chosen among several, it is chosen
  on process grounds statable in advance — it ran the fixed configuration to convergence, with no interventions and
  no environment failure — never because its validation score was the highest, which is the same selection bias one
  level up.

## Consequences
The test is strictly more conservative: sigma rises by 3–10 % early in a run and by ~2 % late. Generation 1 pays for
it, because that is where the pooled estimate is thinnest (4 df) — a borderline candidate there will spend the
adaptive seeds (`Z_BORDER ≤ z < Z_CRIT` → `MAX_CONFIRM_SEEDS`) that the prior used to save it. Both replayed runs show
this costs one marginal regularisation node and no champion.

**Left open for a later ADR:** with the prior gone, the earliest confirmations weigh a fixed `Z_CRIT = 3.0` against a
sigma estimated from 4 df, which is a t-statistic read on a normal table (nominal 0.13 % tail is nearer 1.5 %). The
honest fixes are a df-dependent critical value or a minimum df before deciding; both raise the bar in generation 1,
where the strongest and most reproducible gain of the project (BPR) lives, so both need their own replay before being
adopted. This ADR deliberately changes one thing only.

## Amendment (same day, after review): degrees of freedom, not a higher threshold
The open question above is closed the way the reviewer proposed. `Z_CRIT` stays at 3.0 — a t-critical at a run's
opening df would veto real effects (6.62 at 4 df would reject BPR outright). Instead `MIN_SIGMA_DF = 6`: an
**accepting** verdict resting on a pooled SD with fewer degrees of freedom than that spends the adaptive seeds before
it stands, reusing the existing `Z_BORDER <= z < Z_CRIT` machinery rather than adding a rule. The predicate is one
function, `loop.needs_more_seeds(z, accepted, df, n_seeds)`.

Measured rather than assumed — live_08 and live_09's node_001 (BPR, the project's most reproducible gain and the
node most exposed to this, since it is confirmed first at 4 df) re-run at fresh seeds 4 and 5:

| run | 3 fresh seeds (ADR-0020 alone) | 5 fresh seeds (with the guard) |
|---|---|---|
| live_08 | diff +0.00103, sigma 0.000374, 4 df, z 3.39 → accept | diff +0.00113, sigma 0.000349, 6 df, **z 4.42 → accept** |
| live_09 | diff +0.00168, sigma 0.000376, 4 df, z 5.47 → accept | diff +0.00151, sigma 0.000493, 6 df, **z 4.19 → accept** |

So the guard preserves both verdicts and costs two extra seed runs — normally only for the first candidate of a run,
whose extra seeds raise the pooled df for every confirmation after it. `thin_df` is journalled next to `adaptive`.
