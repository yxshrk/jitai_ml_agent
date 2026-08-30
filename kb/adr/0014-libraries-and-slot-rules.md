# ADR-0014 — Any open-source library in scripts; slot rules in code; wildcards must add information

**Status:** accepted, 2026-08-30 (after live_06; before live_07).
**Supersedes:** the "numpy only" sentence of `workspace/CONTRACT.md` and of the Implementer / Explorer / Librarian
prompts; extends ADR-0009 (generational branching), ADR-0011 (wildcard slot), ADR-0013 (evolving menu).

## Context

Three runs on the corrected harness converged on the same number: live_02 0.6039, live_04 0.6043, live_05 0.6037,
live_06 0.6044 — BPR loss (+0.0013–0.0017 on fresh seeds) plus a seed average (+0.0013–0.0016) of the organizers'
numpy FM, and nothing else that survives the seed test. After live_06 the menu held 48 cards: 38 dead, 7 proven, 3
untried, every proven one a loss / ensembling / regularisation tweak of the same model. The champion's per-group
breakdown shows +0.003–0.02 over the baseline on the live tabs and ~0.2 of headroom to the oracle in *every* tab and
duration band — the model family is weak everywhere, not in one group.

The family was a constraint we imposed on ourselves: the starter kit says "numpy, nothing else *needed*", and the
contract copied it as "nothing else *available*". `kb/spec/rules.md` (the organizers) allows any open-source library,
any paper or public code, even pretrained weights; only external training data is forbidden. With numpy only, a
30-minute CPU cap and ≤ 200-line diffs, the agent could not reach gradient-boosted trees, autograd heads or attention
over the user's history — the standard recipes for engagement logs like this one.

live_06's last twelve nodes (generations 3–5, all flat) also showed four search failures that are about rules, not
capacity, and that the code did not prevent:

1. **Dose-shrinking deepens.** The Selector proposed long-duration matched BPR pairs at 10 % (node_011, −0.0001), then
   5 % restricted to tabs 1/4 (node_016, −0.0002), then 2.5 % in tab 4 only (node_019, −0.0000), plus a long-duration
   specialist ensemble twice (node_010 +0.0003 at z 0.6, node_020 lost). Five nodes on one rejected mechanism with a
   smaller knob each time.
2. **The weakest group need not be fixable.** Long videos need only 18 s of play to be positive, so their labels are
   the noisiest; a low GAUC there is probably irreducible, and the Diagnostician kept naming it "next".
3. **Untried cards locked out.** From generation 2 every Selector slot had to deepen, so
   `history-same-author-run-features` — the one label-free signal with a measured effect (facts §10) — stayed untried
   through live_05 and live_06, and field-aware embeddings on BPR were never run as their own node.
4. **Capacity-only wildcards.** Inverse-day weighting (−0.0026), a third-order user×tab×duration term (−0.0007), a
   user×video×tab CP tensor (−0.0005), a user-tag affinity field (−0.0004): more parameters over the same five id
   fields, where the facts say capacity is not the bottleneck.

And one mechanics fault: the Consolidator's schema example said `"type": "deepen", "parent": "champion"` verbatim, so a
deepen of node_010 was built on the champion; the Critic could only reject it, three rounds running (node_020, 215 K
tokens, no script).

## Decision

1. **Libraries.** Agent scripts may use numpy, pandas, scikit-learn, LightGBM and PyTorch (CPU build), installed in the
   project venv; `config.AVAILABLE_LIBS` and `config.libs_text()` generate the sentence every role reads, and the
   contract's new section "Libraries and determinism" states the rules that keep runs reproducible and the parallel
   branches fair: CPU only, thread count from `OMP_NUM_THREADS` (the harness sets it per branch), every library
   seeded from `--seed` (`torch.manual_seed`, LightGBM `seed` + `deterministic=True` + `force_row_wise=True`),
   `SMOKE_EPOCHS` capping boosting rounds as well as epochs, the learning curve still required. The static firewall,
   the 30-minute cap and the 120 s smoke test are unchanged. The Critic checks the determinism rules (item 5 of its
   checklist). `tests/test_rules_adr0014.py` trains LightGBM + torch under the runner twice and asserts identical
   predictions. On this Mac (no Homebrew, so no `libomp`) LightGBM is linked to the `libomp.dylib` that the torch wheel
   ships, through an rpath added with `install_name_tool` — documented in the README setup.
2. **A free slot from generation 2** (`FREE_SLOT_FROM_GENERATION`): the Selector's first candidate takes, in
   priority, an `untried` card whose `applies_when` holds against the facts, else a proven card not yet measured on
   the current champion stack, else a deepen. The state lists both sets; `Loop._free_slot_ok` checks the answer and
   asks once more with the violation named; a second violation is journaled and the run proceeds.
3. **Deepen de-duplication.** Every deepen carries `mechanism` (a slug of the mechanism family) and `target_group` (a
   breakdown group or `all`), journaled with the node. A mechanism rejected in this run is **closed** for deepening
   (`Loop._closed_mechanisms`, listed in the state; `Loop._apply_rules` drops repeats) — the next deepen must change
   the mechanism, not the dose, the fraction or the group. A deepen of a near-miss node names that node as `parent`.
4. **Hard groups.** A breakdown group with ≥ `HARD_GROUP_REJECTS` (2) rejected deepens is marked hard in the state
   ("likely irreducible label noise — a long video needs only 18 s"); the Diagnostician is told to move on and deepens
   targeting it are dropped.
5. **Wildcards must add information.** The Explorer names `new_signal` — an input absent from the champion's input
   set (the state lists the columns and side-table fields the champion's script references); a wildcard without one
   is dropped by code, and the Critic's item 4 checks that the diff implements the named signal rather than capacity
   over the same inputs.
6. **Critic-directed rebase.** The Critic may answer `revise` with `rebase_to: <node>`; the loop hands the Implementer
   that node's script for the next round and journals `rebased_from`. The Selector/Consolidator schemas now say what
   `parent` means.
7. **Deepen variants are measurements, not cards.** `distill._card_for` maps `<card id> — <variant>` to the base card and
   records the measurement with the variant text; the Archivist no longer mints a card per variant. The Librarian is
   also called after two flat generations, not only when the untried list is short.
8. **Three cards from the literature**, written by the review session and validated: `features-exposure-session`
   (label-free session position / density / same-author run / repeat exposure — facts §10, §10.5),
   `model-lightgbm-lambdarank` (trees over time-safe target statistics and the session features, per-user lambdarank,
   the FM score as a feature), `model-din-history-attention` (target attention over the user's train history keyed by
   author / tag / duration on top of the BPR FM). LightGBM and DIN conflict (one model slot); both compose with the
   session features and the rank ensembles.

## Consequences

- The search can leave the FM family for the first time; a branch may now take minutes instead of seconds, and the k
  parallel branches share the ten cores (`threads = cores // k`, passed as `OMP_NUM_THREADS`). Timeouts are handled
  by the existing Fixer path (reduce cost, keep the method).
- Acceptance and convergence are untouched (ADR-0012): a library model still has to beat the champion's fresh-seed
  mean at z ≥ 3. Non-determinism from a library would show up as a failed seed confirmation, not as a silent gain.
- Honest expectation, agreed by both sessions: +0.002 to +0.005 on validation primary is realistic for new information
  used non-linearly; the ceiling analysis (30 % all-negative users, ~5 impressions per user) still holds.
- Rules 2–5 are enforced in code and stated once in the prompts; the prompts cite the live_06 numbers so the roles
  know *why*. Their cost is a possible second Selector call per generation (~$0.5).
- What changes for the write-up: the harness is no longer "numpy-only by design"; it is "any library, with
  determinism rules the Critic enforces and a runner that meters threads".
