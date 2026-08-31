# ADR-0019 — The submitted model is refit on train + valid at its validated epoch count

**Status:** accepted 2026-08-31 (proposed by the review session from its window experiment, approved by Yash,
built here). **Extends:** ADR-0012 (designation), ADR-0005 (the test firewall — unchanged).

## Context

Every run selects and validates on `valid` and trains only on `train`; the submission has so far been the designated
node's script run once more with `--score-extra` on the hidden test features. Two measurements say that leaves
something on the table, and one says what kind of thing it is:

- The champion script trained on train up to 2022-04-14 (78 % of the rows) scores 0.5999 on valid; on the full
  window 0.6031 (facts §11.3). More label volume is worth about +0.003.
- Training on train + half of valid and scoring the other half is flat against the train-only model (0.5821 vs
  0.5826): it is not *recency* that helps, it is *volume* — so the extra week is worth roughly its share of rows.
- The hidden test week follows the validation week. A model that has seen the validation week trains on ~11 % more
  rows, immediately adjacent to the test period.

The gain cannot be measured on valid, because valid becomes part of the training data. That is precisely why the
number of epochs must not be chosen again: with the validation rows inside the training set, early stopping would
choose a later epoch and overfit. The epoch count comes from the run that validated it.

## Decision

`harness/refit.py`, invoked by `harness.cli submit --refit` (default off; the plain path is unchanged):

1. **Data.** `workspace/data_refit` is built once: the side tables and `valid.csv` are symlinks; `train.csv` is
   train ∪ valid, the valid rows carrying their `long_view`. The outcome columns valid does not have
   (`is_click`, `play_time_ms`, …) are left **empty**, so a script that needs them fails loudly rather than training
   on fabricated values. No test row is anywhere near this directory; the firewall and `private/` are untouched.
2. **Epochs.** The node's own `metrics.json` `best_epoch` is passed as `SMOKE_EPOCHS`, the contract's cap on every
   training phase — which fixes the epoch count for every member of an ensemble, at the value the run validated.
   The refit's `history` length is checked against it and a mismatch is reported.
3. **Checks before the file is accepted:** the header and row count of `predictions_extra.csv`, all scores finite,
   the organizers' `submit.py --check`, and — as a sanity bound, never as validation — the Spearman rank correlation
   with the train-only submission of the same node.
4. **Reporting.** The refit's valid metrics are in-sample and are labelled as such; no test metric is ever computed.
   The returned record (epochs, rows, correlation, evidence) is journaled as a designation-time pipeline step and
   counted as an intervention in the report — it is a human decision applied after the search, not an iteration.

## Consequences

- Expected +0.001 to +0.003 on the hidden test, unobservable on valid by construction; if the organizers' test score
  comes back below the train-only expectation, this step is the first thing to suspect, which is why both files are
  kept and rank-correlated.
- The mechanism is generic: any node script that obeys the contract can be refit without being rewritten, because it
  only uses `--data-dir` and `SMOKE_EPOCHS`.
- Limitation: one epoch count for every member of an ensemble (the node's overall `best_epoch`), not one per member.
  Members of these ensembles are the same model at different seeds and their best epochs differ by one or two; a
  per-member refit would need the script rewritten, which is not worth the risk on the submission path.
