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
2. **Generations of parallel branches.** Each generation, a *Diagnostician* reads the champion's learning curve;
   a *Selector* picks candidates from the cards (each targeting one pipeline component, all different) while an
   *Explorer* proposes one wildcard that is not on the menu; an *Implementer* edits the champion's script for
   each, a *Critic* reviews the diff against the parent's actual stack for leakage, contract and scope, and the
   k = 5 scripts train in parallel. A *Consolidator* then plans merges of orthogonal winners, retests of parked
   ideas, or exploration after a flat generation. From generation 2 the rules are code (ADR-0014): one slot is free
   for an untried card or a proven card not yet on the stack, a rejected deepen mechanism is closed for the run, a
   group with two rejected deepens is marked hard, the wildcard must name a signal the champion does not read, and
   the Critic can rebase a candidate onto the node it actually varies.
3. **Code decides.** A referee validates every prediction file, scores it with the untouched official
   `evaluate.py`, detects no-op changes (predictions byte-identical to the parent), and accepts a node only if its
   improvement holds up over **three fresh seeds** (fresh-seed mean gain ≥ 0.0005 at z ≥ 3 with the seed SD pooled
   over the run — selecting the best of five single-seed branches is biased upward; measured: +0.0022 on one seed was
   +0.0017 over three). The champion is the accepted node with the best seed-mean gain; the run converges after
   N = 3 generations without a ≥ 0.001 cumulative rise of the champion's fresh-seed mean (the organizers' ε = 0.002
   rescaled to a statistic with a third of the noise; the literal single-seed ε rule is tracked and reported
   alongside, with the node it would have submitted). The final submission is chosen among the top
   nodes by seed-mean, never by a single lucky seed.
3b. **The knowledge base evolves.** Every role reads the *exact* run journal — every node's hypothesis, diff,
   curve, seeds and critic notes — from a cached prompt block, plus a foundations note with the task-specific
   mathematics. When a run ends, measurements are folded into the cards and an *Archivist* turns every wildcard
   into a new card written from its actual diff; a *Librarian* with web search adds untried cards from the
   literature after flat generations. The next run starts from a larger, measured menu.
4. **Structural safety.** Agent scripts run in a workspace that physically contains no test rows; test features
   live in a private directory touched once by `harness/submit.py`; a static firewall rejects scripts naming the
   raw data. Every iteration is journaled with its hypothesis, code diff, metrics, learning curve, errors and
   recovery, tokens and wall-clock.

Design decisions and their reasons: `kb/adr/` (fourteen records). The reference: `kb/ARCHITECTURE.md`.
Why this shape and not a pipeline of stage-agents: `kb/literature/agent-design-notes.md` (AIDE, MLE-STAR,
R&D-Agent, ML-Master, AIRA, AI-Scientist-v2 compared).

## Results

_Filled from `runs/<run_id>/summary.json` and `journal.md`._

| run | generations / nodes | champion (valid primary) | Δ vs baseline (valid) | designated node | LLM tokens in / out | cost | wall-clock |
|---|---|---|---|---|---|---|---|
| live_01 | 2 / 7 | node_001 BPR loss — 0.6036 | +0.0021 (3-seed mean +0.0017) | node_004 | 370 K / 64 K | $2.70 | 16 min |
| live_02 | 5 / 16 — **converged** (official rule) | node_001 BPR loss — 0.6031 (3-seed mean 0.6032) | +0.0016 (3-seed +0.0016, t = 8.2) | node_015: 5-seed ensemble of the BPR champion — valid 0.6037, 3-seed mean 0.6039 (+0.0025 over the baseline's 3-seed mean) | 866 K (682 K cached) / 100 K | $4.25 | 30 min |
| live_03 | 1 / 6 — stopped (Explorer schema bug, Critic over-reach; fixed) | node_000 | — | — | 210 K / 34 K | $1.40 | 12 min |
| **live_04** | 6 / 28 — **converged** (k = 5 + wildcard, xhigh; pre-ADR-0012 rule) | **node_015: 5-seed rank-average ensemble of the field-aware(+BPR) and BPR lineages — valid 0.6045 (3-seed mean 0.6043)** | **+0.0030 (3-seed +0.0029; +0.0017 over its parent at z ≈ 6.8)** | **node_015** — tie-break over node_026 (an unaccepted near-no-op variant, means 0.60425 vs 0.60425) | 1.92 M (1.47 M cached) / 393 K | $14.77 | 97 min |
| live_05 | 4 / 16 — **converged** (corrected harness: z-test on fresh seeds, cumulative rule, k 5→3, deepen slots, breakdown, Librarian) | node_002 BPR — 0.6036 (fresh-seed mean 0.6031) | +0.0022 (fresh-seed +0.0017, z 6.5) | node_012: 3-seed rank blend, fresh-seed mean 0.6037 (unconfirmed as champion: +0.0005 at z 1.8) | 1.80 M (0.95 M cached) / 165 K | $9.69 | 35 min |
| live_06 | 5 / 21 — **converged** (parent-resolution and Librarian fixes in) | node_007: BPR + 5-seed rank average — 0.6040 (fresh-seed mean 0.6044) | +0.0025 (fresh-seed +0.0029; +0.0016 over BPR at z 3.8) | node_007 (tie-break over node_010, an unconfirmed +0.0003) | 2.41 M (1.56 M cached) / 190 K | $10.74 | 38 min |
| live_07 | 5 / 25 — **converged** (ADR-0014: libraries + slot rules; first run with LightGBM / torch / session features on the menu) | node_009: BPR + 5-seed rank average — 0.6041 (fresh-seed mean 0.6044) | +0.0026 (fresh-seed +0.0029; +0.0013 over BPR at z 4.7) | node_019 by fresh-seed mean (0.6049, a rejected +0.0005 at z 1.9 — designation rule under revision, see LOG) | 3.71 M (2.13 M cached) / 296 K | $17.82 | 60 min |

`submission.csv` = live_04's designated node (node_015), validated by the organizers' `submit.py --check` (170,588 rows) — it remains the best model by fresh-seed mean (0.6043; live_06's node_007 ties it at 0.6044 with the same recipe, live_05's best was 0.6037). Hidden-test score: reported by the organizers. live_04's per-node evidence and the code review it prompted (ADR-0012) are in `LOG.md`. live_05, the first run on the corrected harness, converged in 35 minutes for $9.69 with the same BPR gain (z 6.5) and, under the stricter test, no confirmed ensemble on top — its deepen attempt at a 5-member blend was lost to a parent-resolution bug (fixed, `c78c1fe`); the organizers' literal ε rule would have stopped it at the same generation. Token counts are total input (cached share in brackets) / output, from the API's usage fields.

What the runs say together: a ranking-aligned loss (within-user BPR) is a real, reproducible gain over the pointwise FM (+0.0016 / +0.0017 / +0.0011 / +0.0013 / +0.0016 over fresh seeds in five runs); averaging seeds — and two lineages — on top adds a statistically clear gain (+0.0009 in live_02, +0.0017 in live_04, +0.0016 in live_06), and four runs stop at the same 0.604: the numpy FM family is exhausted (ADR-0014 lifted our own numpy-only rule — the organizers allow any library — and live_07 then measured LightGBM lambdarank at −0.0022 and DIN-style history attention at −0.0005 on the BPR stack, label-free session features at +0.0009 on BPR but +0.0002 on the seed blend; the research session's ceiling study, `kb/data/screens/CEILING.md`, bounds every remaining family at ≤ +0.0003 — 69 % of the remaining error is tab-1 pairs on different days, and ≈ 0.605–0.607 is what these inputs contain on valid); nearly every other lead the organizers listed — recency weighting, duration features, history aggregates, LambdaRank weighting, listwise softmax, an is_click head, censored watch-time regression, regularisation, lr decay, checkpoint averaging, DCN cross heads — measured flat on these stacks once seed noise was controlled. The winner's-curse correction rejected twelve single-seed "wins" of +0.0002 to +0.0006 across the runs. The remaining headroom is mostly not reachable by modelling: 30 % of validation users have no positive, nDCG@5 saturates near 0.70, and users have ~5 impressions each.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install numpy openai python-dotenv pytest pypdf
.venv/bin/pip install pandas==2.3.3 scikit-learn==1.6.1 lightgbm==4.6.0 torch==2.8.0   # agent-script libraries (ADR-0014)
# macOS without Homebrew: LightGBM needs an OpenMP runtime — point it at the one the torch wheel ships
SP=$PWD/.venv/lib/python3.9/site-packages; install_name_tool -add_rpath "$SP/torch/lib" "$SP/lightgbm/lib/lib_lightgbm.dylib" && codesign -s - -f "$SP/lightgbm/lib/lib_lightgbm.dylib"
cd kuairand-starter-kit && wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz && tar xzf KuaiRand-Pure.tar.gz && cd ..
echo 'OPENAI_API_KEY=sk-...' > .env          # never committed (.gitignore)
.venv/bin/python -m harness.data_access      # builds workspace/ (train + valid) and private/ (test features, no labels)
```

## Reproduce

```bash
# 1. the official baseline through the harness contract (valid primary 0.6015 at seed 0; published 0.6016 ± 0.0008)
.venv/bin/python -c "from harness import referee as R; print(R.run_script('harness/seeds/node_000_fm.py', 'runs/_check/000', seed=0).metrics)"
# 2. tests (16 unit + 1 end-to-end generation with a scripted brain, ~2 min; includes LightGBM + torch under the runner)
.venv/bin/python -m pytest tests -q
# 3. an autonomous run (GPT-5.6 via the OpenAI API; ~$1.5–2 per generation)
.venv/bin/python -u -m harness.cli run --run-id my_run --k 5 --budget-usd 30   # distills + archives the cards when it ends
# 4. the submission for the designated node (writes the CSV and runs the organizers' --check)
.venv/bin/python -m harness.cli submit --run-id my_run --node <designated> --out submission.csv
```

Iteration logs: `runs/<run_id>/journal.jsonl` (machine-readable), `journal.md` (human-readable), `diffs/`, `nodes/`.
Manual interventions: recorded per node (`intervention` flag) — none in the reported runs.

## Repository map

| path | what |
|---|---|
| `harness/` | the loop (`loop.py`), roles (`brain.py`, `prompts.py`), referee, journal, distill/Archivist, Librarian, submission, CLI |
| `kb/spec/` | frozen task, rules, scoring, foundations (task-specific mathematics), corrections to the source doc |
| `kb/data/` | EDA script, report, and the interpreted facts the agent reads |
| `kb/methods/` | method cards (the agent's menu) + validator |
| `kb/literature/` | reading guide, agent-design survey, paper fetcher (`fetch.sh`) |
| `kb/adr/` | architecture decision records |
| `kuairand-starter-kit/` | the organizers' kit, comments translated to English (logic verified identical) |
| `workspace/CONTRACT.md` | the interface every agent-written script obeys |
| `LOG.md` | chronological log of changes, decisions and results |

## Limitations and what we would do with more time

- Until ADR-0014 (after live_06) agent scripts were numpy-only by our own contract, so heavier heads were written by
  hand; the runs reported above are all from that regime. pandas, scikit-learn, LightGBM and PyTorch (CPU) are now
  available under determinism rules the Critic checks; live_07 is the first run that can use them.
- Diagnostics (ablations, probes) still cost validation evaluations; a train-tail holdout for diagnostics
  would protect the validation set further as runs get longer.
- The organizers' definition of "iteration" (loop turn vs training run) is ambiguous; both counts are journaled
  and the cap is switchable (`--iteration-unit`).
- The Librarian (web search) and the Archivist have run on live_04's journal; their cards enter as `untried` and
  are judged by measurement like every other card. `AnthropicBrain` is kept as an alternative backend but has
  not been exercised in the reported runs.

## Team

Yash Raj Khandelwal — problem framing, architecture decisions, knowledge base, harness, runs, write-up.
