# Devpost form — copy/paste fields (1 Sep 2026)

## Project name (≤60 chars)
jit.ai: an agent that runs its own ML experiments

(alternatives: "jit.ai · autonomous ML research agent" / "jit.ai — self-steering ML research on KuaiRand")

## Elevator pitch (tagline)
An LLM agent that diagnoses, changes, retrains and measures a recommendation model in a loop no person steers: 0.6016 → 0.6056 on KuaiRand-Pure, zero mid-run interventions, and a harness built to check its work.

(short alt, if the field is tight: "Half of this system is an agent. The other half exists to check its work. 0.6016 → 0.6056 on KuaiRand-Pure, zero mid-run interventions.")

## Built with (tags)
python, pytorch, numpy, openai-api, gpt-5, uv, optuna, matplotlib, pandas, kuairand, git, github, html5, css3, javascript, svg, ffmpeg, cuda, macos, linux

## "Try it out" links
- Site: <Vercel URL once deployed> (interactive replay of the designated run, all 42 method cards, provenance table)
- Code, journals, cards: https://github.com/yxshrk/jitai_ml_agent (branch `clean-agent`)
- Presenter view used in the video: <Vercel URL>/present.html

## Image gallery (3:2, PNG, all in evidence/devpost_gallery/)
01_title · 02_run_chart · 03_loop · 04_architecture · 05_log_replay · 06_harness ·
07_method_cards · 08_principles · 09_ceiling · 10_conclusion
Suggested order: 04 (architecture), 02 (run chart), 03 (loop), 05 (log replay), 06 (harness), 09 (ceiling), 10 (conclusion), 07, 08, 01.

## About the project (Markdown — paste below)

## Inspiration

Track 2 asks for an agent that does ML research on its own. The easy version is a loop that calls an LLM and keeps whatever scores higher. We wanted the hard version: an agent whose every claim we could audit, whose gains beat measured noise, and whose run could be replayed decision by decision. The guiding line became: *half of this system is an agent; the other half exists to check its work.*

## What it does

Given the KuaiRand-Pure starter kit, the agent reproduces the published FM baseline (0.6016), then iterates on its own: it **diagnoses** the last training curve, **treats** the bottleneck by picking one method from a 42-card library of published ranking techniques (each with mechanism, applicability and measured history), **retrains** on the same time-based split, and **measures** with the official GAUC/nDCG@5 evaluator. A change is kept only if its gain clears a noise-calibrated gate; the run stops by the competition's epsilon rule (3 iterations without +0.002). The designated run, `bigclock_07`, went **0.6016 → 0.605575** in 6 iterations, 17 minutes on a CPU, 115k tokens, **zero mid-run human interventions**. Inside one iteration it ran its own two-stage hyperparameter search, then designed its own seed ensemble (7 trained, 3 validation-selected, per-user rank average).

## How we built it

- **Agent (proposes):** an LLM selector/proposer (`gpt-5.6-sol`) writes journals, cites a card, and emits either a whole training script or, for improvements, an exact search/replace patch against the accepted parent so accepted code evolves byte-identically instead of being re-typed.
- **Harness (checks, fixed code, no LLM):** screens code for test-set access, smoke-runs it (360 s) behind a sanity gate calibrated on measured broken-vs-sane populations, trains under a timeout, evaluates with the official metric, gates on 3-seed sigma with grey-zone reseeds, and journals hypothesis/diff/metrics/recovery every iteration. The hidden test set is physically absent from the workspace.
- **Knowledge:** `agent/METHODS.md` (42 cards) plus a literature-only variant for clean runs. Cards carry measured evidence from earlier runs, so the system is a research loop *across* runs, not just within one.
- **Judgment tested like code:** every wrong live decision was frozen as a benchmark fixture from the real run state and fixed with a general principle (never a scenario-specific answer). 160 harness tests plus decision benches gate every prompt or knowledge change.
- **Site:** a static "flight recorder" that replays the designated run from its journal file, with the full card library and a provenance table.

## Challenges we ran into

- **Noise vs. signal.** Seed sigma on Pure is ~0.0004 and epsilon is 0.002, so most "wins" are noise. We calibrated sigma per run and made the gate own acceptance, not the LLM.
- **Implementation fidelity, not decision quality, is the bottleneck.** Once the selector was at bench-zero on real states, fresh re-implementations of a known-good recipe still spread 0.592–0.600. Reference snippets and constrained patches closed most of that gap.
- **Honest endgames.** Runs love to "close" with a same-family ensemble whose ceiling cannot clear epsilon. We wrote the epsilon arithmetic into the guidance and verified it on replayed states.
- **Provider outages mid-decision** (a 503 ended a run) → transient errors now retry without counting as a strike.

## Accomplishments we're proud of

A clean, unseeded, converged run at **0.605575** that sits at the measured single-run ceiling (0.6055–0.6060, established by five hardened harness generations on the final night). Every number on the site and in the report has a run directory behind it. We declined to submit anything human-assisted (a 0.6065 cross-run blend, a 0.6061 hand-built model) and disclose them as evidence only.

## What we learned

Autonomous research needs accurate memory and verified execution more than clever prompting. Negative results are first-class: sequence models, watch-time auxiliaries, kitchen-sink features and post-hoc blends were all measured dead on Pure and recorded as such. "Less is more" held: the five-field official model with strong joint regularization beat every richer variant.

## What's next

A harness-owned sweep executor (so the agent plans searches as data and never re-implements them), broader search beyond greedy improve-best, and the same loop on the larger KuaiRand variants where the bonus run already reached 0.66892 (1K).

## Resources

Designated Pure run: 6 iterations, 17.0 min wall-clock, 115,315 tokens, no GPU. Campaign: ~146 runs, ~10.6M tokens, ~143 run-hours, disclosed in `logs/RUNS.md` and `evidence/RESULTS_AND_RESOURCES.md`.
