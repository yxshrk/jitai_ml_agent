# ADR-0015 — The feature screen: measure a proposed signal on valid before spending a node on it

**Status:** accepted, 2026-08-31 (during live_07; merged before the next run).
**Extends:** ADR-0011 (wildcard slot), ADR-0014 (free slot, information-adding wildcards). Nothing about acceptance
or convergence (ADR-0012) changes.

## Context

Every run since live_02 has converged at 0.6037–0.6044 on the same two ingredients (BPR, seed averaging). ADR-0014
opened libraries and forced untried cards into the plan on the theory that the *information* the model sees, not its
capacity, was the bottleneck. On 2026-08-31 the review session measured that theory directly on valid, against
live_07's champion (`kb/data/screens/`):

| family (out-of-fold where labels are involved) | standalone within-user GAUC | added on top of the FM |
|---|---|---|
| item target statistics: video / author / music / tag long-view rate, click rate, play-through, tab×video, tab×author | 0.634–0.649 | ≤ +0.0004 each |
| session position / density / gap (node_002's own features) | 0.508 | ≈ 0 — the column varies within a user's rows for 37 % of users |
| exposure context in the split: gap to next / previous exposure, next / previous same author, first / last of user | 0.50–0.52 | ≤ +0.0005 |
| lambdarank GBDT stacked on the FM score + 62 features from every family, 5-fold CV over users | — | 0.6014 alone, 0.6038 blended (champion 0.6031) |

The FM's id embeddings already carry what a lookup table knows about a video; a user's valid rows are 5–7 impressions
spread over a median of three days, so session structure barely exists; a user has seen the author before in 13 % of
rows. Everything lands in 0.600–0.605: an information ceiling for these inputs, not a search failure. live_07 then
confirmed it the expensive way — node_002 (session features, −0.0001), node_007 (LightGBM lambdarank, −0.0022),
node_013 (session features retested on the seed blend, +0.0002 at z 1.0) — three nodes, three Critics, nine seeds.

Each of those verdicts was available in about a minute of compute *before* the node was built, because the metric
rewards exactly one thing — reordering a user's own rows — and that is measurable for any column on valid without
training anything. The harness had no step for it: a feature hypothesis went straight to an Implementer at xhigh
effort, a Critic, a 30-minute run and three confirmation seeds.

## Decision

1. **A probe before a node.** Every candidate whose `target_component` is in `SCREEN_COMPONENTS` (features, encoding,
   history), and every wildcard that names a `new_signal`, is probed first. Merges and retests are not (their feature
   is already measured). A new LLM role, **Probe** (medium effort, ~20 s), writes a short script that computes the
   proposed signal for the valid rows — as raw as possible (a rate, a count, a gap), not as bucket ids.
2. **Label-freeness is a property of the sandbox.** The probe runs like a node (workspace cwd, static firewall, thread
   env, `SCREEN_TIMEOUT_S` = 180 s) but on `workspace/data_probe/`: every file of `workspace/data` linked, and a
   `valid.csv` with **every outcome column stripped**. A probe that reads the scored row's label fails with a
   `KeyError`, whatever its code says. Train labels remain available for statistics.
3. **Four numbers, all within-user** (`harness/screen.py`, deterministic, no LLM), measured against the champion's
   own seed-0 valid predictions:
   - `varies` — the share of users whose valid rows differ on the column (17.5 % of users have one row, so 0.825 is
     the ceiling; a column constant within a user cannot reorder anything and scores 0);
   - `gauc` — standalone within-user GAUC, best sign;
   - `additive` — the best Δprimary of z(champion) + w·z(column) over w ∈ {0.1, 0.25, 0.5, 1};
   - `stack` — Δprimary of a lambdarank LightGBM on [champion score + all columns] against [champion score] alone,
     5-fold cross-validation over users, seeded (interactions the additive test cannot see).
   `best_gain` = max(additive over columns, stack).
4. **The gate.** `best_gain < SCREEN_MIN_GAIN` (0.0003, below `MIN_EFFECT` = 0.0005 because the screen is a lower
   bound — a feature can do more inside the model than beside it) drops the slot before anything is built. A failed
   probe (timeout, crash, malformed output) never blocks: the candidate proceeds unscreened and the failure is journaled.
5. **Memory.** Each screen is a journal record (`action: screen`, with `card`, `family`, `target_component`,
   `new_signal`, `best_gain`, `best_column`, `stack_gain`, `columns`, `kept`) rendered in the digest and journal.md;
   `state['screened']` carries `{generation, card, family, best_gain, kept}` and the planners see "Screened this run"
   in their state so a dropped signal is not re-proposed in another wording. Passing candidates carry the numbers
   into their node record (`screen`) for the Implementer and Critic. Distill folds screen records into the card's
   Measured line with a `screened_out` suffix (ADR-0016's side, since nothing was trained).
6. **Interface for campaigns (ADR-0016).** Family ordering may read `state['screened'][*].family` and `best_gain`
   directly; `family` comes from the card's front matter, or the target_component for a wildcard.

Off switch: `--no-screen`; the screen is also inactive for any brain without a Probe role (the FakeBrain), so the
existing tests are unchanged.

## Cost

Per probed candidate: one medium-effort call (~20 s, ~10 K tokens) plus 30–90 s of CPU (probe run + ~7 scorer calls
per column + ten small LightGBM fits), run in parallel across candidates. Generation 1 of live_07 would have probed
two of five candidates; the whole screen is cheaper than one Implementer call at xhigh.

## Consequences

- A feature idea that cannot show within-user discrimination on valid costs a minute, not a node. The Explorer's
  information-adding wildcards (ADR-0014) get an immediate, honest verdict; the Librarian's feature cards likewise.
- The screen is a lower bound on the model's use of a signal, so it can reject a feature that a nonlinear model would
  have exploited in combination. The stack term and the low threshold (0.0003) limit that; a candidate the planners
  believe in can be re-proposed as a `retest` (not screened) with the reason.
- It measures on valid, like everything else in the run; the seed test (ADR-0012) remains the acceptance decision.

## Calibration (the real screen on live_07's own feature nodes, 2026-08-31)

| candidate, measured against the champion it was proposed on | seed-test verdict | screen best_gain | additive (best column) | stack |
|---|---|---|---|---|
| node_001 last-positive tag / music / type match + recency, vs node_000 | ACCEPTED +0.0009, z 3.1 | **+0.0013** | −0.0009 (every column negative alone) | +0.0013 |
| session position / density / gap, vs node_003 (BPR) | ACCEPTED +0.0009, z 3.1 (node_010) | **+0.0012** | +0.0012 (previous_gap) | +0.0001 |
| the same session features, vs node_009 (seed blend) | rejected +0.0002, z 1.0 (node_013) | +0.0008 | +0.0003 | +0.0008 |
| item target statistics (video / author / tag / play-through), vs node_003 | never built | +0.0003 … +0.0004 | ≤ +0.0004 | — |
| random noise / constants (tests) | — | ≈ 0 | ≈ 0 | ≈ 0 |

Every node the seed test accepted clears the 0.0003 gate by 4×; node_001 clears it only through the stack term (its
columns are individually negative beside the FM — the interaction test is not optional). The gate is a *floor that
removes noise-equivalent candidates*, not a predictor of acceptance: a +0.0008 screen still failed the seed test. Raising
the threshold to MIN_EFFECT (0.0005) would keep every accepted node but gains little; it stays at 0.0003. A probe runs
in 14–17 s end to end (one Probe call more).

## Tests

`tests/test_screen.py`: the probe dir strips outcomes; a real signal (video train rate) scores varies > 0.8, GAUC > 0.6,
additive and stack > +0.005 against a tab×duration prior, a user-constant column scores varies 0, random noise is
dropped; a label-reading probe fails at the probe stage and does not block; the firewall and malformed outputs fail
at their stages; the loop gate keeps a strong candidate, drops a null wildcard, passes a failed probe through, skips
loss / retest candidates, journals and renders the records, and populates `state['screened']`; the screen is off
without a Probe role or with `screen=False`.
