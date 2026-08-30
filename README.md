# Autonomous ML Research Agent for KuaiRand-Pure — TikTok TechJam 2026, Problem Statement 2

An agent that runs the machine-learning iteration loop on its own — read the problem, engineer features, train and
tune, evaluate, reflect, repeat — on the KuaiRand-Pure within-user ranking benchmark, and beats the organizers'
Factorization-Machine baseline without a human in the loop.

> **Results:** see [Results](#results) (filled from the run journals in `runs/`).

## The problem in one paragraph

For every user in the evaluation window, the app showed ~7 videos. The task is to **reorder each user's shown videos**
so the ones they actually long-viewed (`long_view = 1`: watched ≥ min(video length, 18 s)) come first. Score =
mean of GAUC (per-user pairwise ordering, weighted by positives) and nDCG@5 (top-of-list ordering), computed by the
organizers' `evaluate.py`. Official FM baseline: **0.6016 valid / 0.5946 hidden test**. A perfect ordering scores
0.8484 / 0.8645 (27 % of test users have no positive, so nDCG is capped). Full spec: `kb/spec/`.

## Approach

**A deterministic harness searches a graph of whole training scripts; LLM roles propose and write, code judges.**

1. **Knowledge first, then search.** Before any run, the agent's knowledge base is built: the frozen spec
   (`kb/spec`), *measured* facts about the data from an EDA pass (`kb/data/facts.md` — e.g. no item cold start,
   short videos are the hard case, traffic drifts 10× across the training window), and **method cards**
   (`kb/methods`) — one per technique from the literature (`kb/literature`), each with checkable preconditions
   against the facts, an honest expected gain calibrated to the 0.002 noise floor, and an implementation recipe
   written against the baseline script.
2. **Generations of parallel branches.** Each generation, a *Diagnostician* reads the champion's learning curve,
   a *Selector* picks k = 3 candidates from the cards (each targeting one pipeline component, all different), an
   *Implementer* edits the champion's script for each, a *Critic* checks for leakage and contract violations, and
   the three scripts train in parallel. A *Consolidator* then plans merges of orthogonal winners, retests of parked
   ideas, or exploration after a flat generation.
3. **Code decides.** A referee validates every prediction file, scores it with the untouched official
   `evaluate.py`, and accepts a node only if its improvement holds up over **three seeds** (selecting the best of
   three single-seed branches is biased upward — measured: +0.0022 on one seed was +0.0017 over three). The
   official convergence rule (ε = 0.002 over N = 3) stops the run. The final submission is chosen among the top
   nodes by seed-mean, never by a single lucky seed.
4. **Structural safety.** Agent scripts run in a workspace that physically contains no test rows; test features
   live in a private directory touched once by `harness/submit.py`; a static firewall rejects scripts naming the
   raw data. Every iteration is journaled with its hypothesis, code diff, metrics, learning curve, errors and
   recovery, tokens and wall-clock.

Design decisions and their reasons: `kb/adr/` (ten records). The reference: `kb/ARCHITECTURE.md`.
Why this shape and not a pipeline of stage-agents: `kb/literature/agent-design-notes.md` (AIDE, MLE-STAR,
R&D-Agent, ML-Master, AIRA, AI-Scientist-v2 compared).

## Results

_Filled from `runs/<run_id>/summary.json` and `journal.md`._

| run | generations / nodes | champion (valid primary) | Δ vs baseline (valid) | designated node | LLM tokens in / out | cost | wall-clock |
|---|---|---|---|---|---|---|---|
| live_01 | 2 / 7 | node_001 BPR loss — 0.6036 | +0.0021 (3-seed mean +0.0017) | node_004 | 370 K / 64 K | $2.70 | 16 min |
| live_02 | _running_ | | | | | | |

Hidden-test score: reported by the organizers on the designated node's submission (`submission.csv`).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install numpy openai python-dotenv pytest pypdf
cd kuairand-starter-kit && wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz && tar xzf KuaiRand-Pure.tar.gz && cd ..
echo 'OPENAI_API_KEY=sk-...' > .env          # never committed (.gitignore)
.venv/bin/python -m harness.data_access      # builds workspace/ (train + valid) and private/ (test features, no labels)
```

## Reproduce

```bash
# 1. the official baseline through the harness contract (valid primary 0.6015 at seed 0; published 0.6016 ± 0.0008)
.venv/bin/python -c "from harness import referee as R; print(R.run_script('harness/seeds/node_000_fm.py', 'runs/_check/000', seed=0).metrics)"
# 2. tests (6 unit + 1 end-to-end generation with a scripted brain, ~1 min)
.venv/bin/python -m pytest tests -q
# 3. an autonomous run (GPT-5.6 via the OpenAI API; ~$1.5–2 per generation)
.venv/bin/python -u -m harness.cli run --run-id my_run --k 3 --budget-usd 30
# 4. the submission for the designated node (writes the CSV and runs the organizers' --check)
.venv/bin/python -m harness.cli submit --run-id my_run --node <designated> --out submission.csv
```

Iteration logs: `runs/<run_id>/journal.jsonl` (machine-readable), `journal.md` (human-readable), `diffs/`, `nodes/`.
Manual interventions: recorded per node (`intervention` flag) — none in the reported runs.

## Repository map

| path | what |
|---|---|
| `harness/` | the loop (`loop.py`), roles (`brain.py`, `prompts.py`), referee, journal, submission, CLI |
| `kb/spec/` | frozen task, rules, scoring, and corrections to the source doc |
| `kb/data/` | EDA script, report, and the interpreted facts the agent reads |
| `kb/methods/` | method cards (the agent's menu) + validator |
| `kb/literature/` | reading guide, agent-design survey, paper fetcher (`fetch.sh`) |
| `kb/adr/` | architecture decision records |
| `kuairand-starter-kit/` | the organizers' kit, comments translated to English (logic verified identical) |
| `workspace/CONTRACT.md` | the interface every agent-written script obeys |
| `LOG.md` | chronological log of changes, decisions and results |

## Limitations and what we would do with more time

- Only numpy is available to agent scripts; heavier heads (DCN, sequence models) are written by hand and cost
  runtime. Allowing PyTorch would widen the menu.
- Diagnostics (ablations, probes) still cost validation evaluations; a train-tail holdout for diagnostics
  would protect the validation set further as runs get longer.
- The organizers' definition of "iteration" (loop turn vs training run) is ambiguous; both counts are journaled
  and the cap is switchable (`--iteration-unit`).
- Method cards were written by hand from the literature; a Librarian agent extending them from papers and web
  search is the natural next step for the "draws on published methods" criterion.

## Team

Yash Raj Khandelwal — problem framing, architecture decisions, knowledge base, harness, runs, write-up.
