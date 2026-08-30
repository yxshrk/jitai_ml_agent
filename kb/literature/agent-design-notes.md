# Agent-design notes — what the autonomous-ML-agent literature says (2026-08-30 survey)

Sources: AIDE (`agents/2502.13138_aide.pdf`), MLE-bench (`agents/2410.07095_mle-bench.pdf`), AI-Scientist-v2
(`agents/2504.08066_ai-scientist-v2.pdf`), plus three read online: MLE-STAR (arXiv 2506.15692, Google),
R&D-Agent (arXiv 2505.14738, Microsoft), ML-Master (arXiv 2506.16499), and the Meta/UCL search-policy study
"AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench" (arXiv 2507.02554, "AIRA").

## The common shape of every strong system
All of them are **search over a tree/graph of whole runnable scripts**, not pipelines of stage-specialised agents:
- a *node* = one script + its measured score (+ logs, learning curve);
- an *operator* is applied to one selected parent: **draft** (new solution), **improve** (one change to a working
  node), **debug** (fix a failed node) — AIRA adds memory and crossover operators;
- a **fixed external scorer** produces the fitness (never the LLM's own opinion);
- a **hard-coded search policy** (AIDE: greedy on the best node; ML-Master/AIRA: MCTS with UCT; R&D-Agent: DAG with
  parent selection + pruning) decides which node to expand next;
- the LLM's context is a **compact summary** of prior nodes (metrics, hyperparameters, debugging hints) plus the
  parent's code — never a growing transcript.

Orchestration is deterministic code in every case. Roles exist *inside* an operator (R&D-Agent: Researcher proposes a
hypothesis, Developer implements it; MLE-STAR: extractor → planner → coder → checkers → debugger), but agents do not
decide handoffs among themselves.

## Findings that change our design
| finding | source | consequence for us |
|---|---|---|
| **Operators matter more than the search algorithm.** With AIDE's operators, MCTS/evolutionary search gave no gain; improving the operators lifted MLE-bench-lite medals 39.6 % → 47.7 %. | AIRA | Invest in proposal quality (cards, facts, targeted changes), not a fancy search policy. Greedy + forced branching is enough at 50 iterations. |
| **Validation-guided search overfits.** Selecting the final node by test instead of validation would add 9–13 points absolute; validation keeps climbing while test plateaus/declines; longer search makes it worse. Robust final-node selection (e.g. re-rank the top-k) recovers ~10 %. | AIRA | Our valid set is 125 K rows with a 0.0008 seed σ. Accept only ≥ ε; multi-seed confirm grey-zone wins; designate the final node among the top-k valid nodes using an **independent, legal signal** (multi-seed mean and a train-internal time holdout). Stop when the convergence rule fires — do not spend iterations just because they exist. |
| **Targeted refinement beats whole-script rewriting.** Ablate the pipeline, find the most impactful component, refine only that block; +18 points over AIDE on MLE-bench-lite. | MLE-STAR | Every proposal names one `target_component`; diagnostics (ablations) run on a train-internal holdout, never on valid. |
| **Leakage / data-usage checkers pay for themselves.** Without the leakage checker: valid 0.8188 → 0.8677 while test collapsed 0.8033 → 0.7343. | MLE-STAR | A Critic role specialised to *our* leakage rules (outcome columns, statistic file, time-ordered history) before any script runs. |
| **Sample-based debugging** (subset of train) before full runs; dynamic budget (cheap ideas early, ensembles late). | R&D-Agent | Smoke test (`SMOKE_EPOCHS=1`) before every full run; ensembling reserved for the closing iterations. |
| **Removing structured exploration paths hurt most** (35.1 % → 25.3 %); planning and multi-step reasoning each −24 %; memory −9 %. | R&D-Agent | Keep the tree explicit (parent links, journal), plan the ladder up front, let the proposer reason before coding. |
| **Bounded retries.** Debug depth 3 (AI-Scientist-v2), τ_improve = 3 failed improvements before a node is terminal (ML-Master), debug loops capped (AIRA). | all | Fixer gets one attempt per stage; a node that fails twice is abandoned and the search moves on. |
| **Scoped memory.** Draft/improve see *sibling* summaries (diversity); debug sees *ancestors* (no oscillation). Prompt complexity grows with the node's number of children ("minimal / moderate / advanced"). | AIRA | Journal lines + parent code; add sibling summaries for improve, ancestor trace for debug; start simple. |
| **Ensembling at the end** raised medals 37.9 % → 43.9 % (and gold 25.8 → 30.3). | MLE-STAR | Seed / node ensembles as the final operator — cheap here (16 s per seed). |
| Infrastructure alone was worth +10 points (same agent, better sandbox). | AIRA | Timeouts, isolation, clean logs, resumability are score, not polish. |
| Staged progression with node budgets per stage; a stage ends on a criterion (prototype runs; curves converge; budget spent). | AI-Scientist-v2 | Phases: reproduce → cheap single-component improvements → heavier methods → ensembling/closing. |

## What the literature does NOT support
- A chain of stage agents (analysis → features → training → evaluation) handing work to each other: every measured
  system rejected the chain in favour of a tree of scripts; R&D-Agent's ablation shows chain-shaped exploration
  loses ~10 points.
- LLM self-evaluation of results: all systems use an external scorer.
- Free-form agent-to-agent handoffs: orchestration is code everywhere.
