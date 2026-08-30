# ADR-0008 — Harness architecture: deterministic orchestrator, fixed per-iteration role graph, tree search over scripts
Status: Proposed (2026-08-30) — awaiting Yash's confirmation

## Context
Yash proposed specialised sub-agents (analysis, literature, feature engineering, training/tuning, evaluation) with an
orchestrator that retries a failed agent, and asked whether an orchestrator or a graph of hand-offs is better.
The literature (see `kb/literature/agent-design-notes.md`) is uniform: every strong system is a tree search over
whole scripts with a deterministic controller; specialisation lives inside an iteration, not across a pipeline.

## Decision
1. **Orchestrator = code** (`harness/loop.py`). It runs the same directed graph every iteration:
   `diagnose → select → implement → check → smoke → full run → referee → journal → converge?`
   Each step is retried on its own with a bounded budget (parse retry ×1, fixer ×1 per stage); a node failing twice
   is abandoned and the search moves on. This is Yash's "retry that agent itself", scoped to steps.
2. **Search = a tree of nodes** (script + score + curve). Policy: improve the champion greedily; after a stagnation
   streak, force a branch to a different method family; keep sibling/ancestor summaries in scope (AIRA).
3. **Specialised LLM roles, each a single call with a narrow contract:**
   Analyst (once, build time: `kb/data/facts.md`), Librarian (build time: cards), Diagnostician (reads the parent's
   learning curve + per-cohort metrics → overfit / underfit / flat), Selector (picks one card, cites it, names one
   rejected alternative, declares one `target_component`), Implementer (whole-script edit, smallest coherent change),
   Critic (leakage / noise / contract review — MLE-STAR's checkers, specialised to our rules), Fixer (traceback → patch).
4. **Never an LLM:** scoring, acceptance (≥ ε), champion/argmax selection, convergence, timeouts, firewall.
5. **Anti-overfitting protocol (AIRA):** accept only ≥ ε; grey-zone deltas trigger a 2-seed confirmation; the final
   designated node is chosen among the top-k valid nodes by multi-seed mean plus a train-internal time holdout;
   diagnostics (ablations, probes) run on that internal holdout, never on valid.
6. **Phases with budgets (AI-Scientist-v2):** reproduce → single-component improvements → heavier methods →
   ensembling/closing; each phase has an iteration cap so the run cannot stall in one.

## Consequences
Fewer tokens and less wall-clock than a swarm (one call per role per iteration, compact context), no agent-to-agent
drift, every failure is retried at the step where it happened, and the journal doubles as the deliverable.
The "pipeline stages" from Figure 1 map onto `target_component` values of the Implementer, not onto agents.
