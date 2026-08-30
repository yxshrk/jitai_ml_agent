# mle-agent — TechJam 2026 Track 2

An autonomous ML research agent for the KuaiRand-Pure benchmark (rank each user's
impressions by `long_view`; primary score = mean of GAUC and nDCG@5).

**Headline result (validation):** official FM baseline **0.6016** → our agent,
starting from that baseline with **no team-provided solution and zero manual
interventions**, reached **0.60558** (+0.0040 primary; +0.0054 GAUC, +0.0026
nDCG@5) in 6 iterations / 17 minutes — designated run `logs/run_bigclock_07`.
Bonus: KuaiRand-1K **0.63874** (agent-designated, `logs/run_desig_1k_01`);
KuaiRand-27K 0.67263 (out-of-protocol GPU scaling demo).

## How the agent works

Each iteration is one research decision, executed as deeply as needed:
1. **Diagnose** — read the champion's learning curves and the run journal.
2. **Decide** — select one move from a cited method-card library
   (`agent/METHODS.md`): an atomic method, a literature package, or a *search*
   (dial sweep / cross-stage matrix / add-on combo / ensemble design).
3. **Fan out** — the emitted script may train dozens of candidate variants
   (random search over wide ranges, successive halving, full-fidelity finals),
   select the winner on validation, and commit it as the node's single artifact;
   every probe is logged (`node_*/progress.log`, metrics history).
4. **Judge** — accept/reject against a seed-calibrated noise floor with
   multi-seed grey-zone confirmation.
5. Repeat until the official convergence rule fires (ε=0.002 / N=3, 50-iteration
   cap, 6h ceiling — implemented verbatim, failures counted conservatively).

The designated run's arc: dial-search over a regularized DCN package (accepted
0.60424, clearing ε) → ensemble-design sweep in which the agent trained 7 seed
members and validation-selected a 3-member per-user rank average → 0.60558,
converged. The dials it found are non-obvious (dropout 0.18, weight decay 9e-5,
LR ×0.57 every 2 epochs) and were later reproduced from scratch (0.60561).

## Layout

- `harness/` — the loop (proposal → smoke test → train with timeout → official
  evaluation → journal → acceptance/convergence), budget ledger, CLI.
- `agent/` — LLM roles (proposer/selector/reflector/fixer), prompts, method-card
  knowledge base with measured annotations and search policies, models config.
- `zoo/` — frozen reference scripts and experiment runners from the development
  campaign; `zoo/EXPERIMENTS*.md` are the measured ledgers (every cell, seeds).
- `evidence/` — DEVPOST text, results & resource tables, figures, submission
  builder/checker, test CSVs (built train-only; test labels never read).
- `logs/` — one directory per run: `journal.jsonl` (hypothesis, diff, metrics,
  errors/recovery per iteration), node code, probe logs, `summary.json`.
  `logs/RUNS.md` indexes all ~45 disclosed runs.
- `data/` — exporters and (gitignored) encoded splits; `data/official/` is the
  organizer evaluation code.
- `tests/` — pytest suite (119 tests) covering the harness, scoring parity, and
  the submission pipeline's leakage guards.

## Setup & reproduction

```bash
uv sync                                # or: pip install numpy torch openai
# 1) data: download KuaiRand-Pure (kuairand.com) next to this repo, then
uv run python data/export_real_ws.py   # builds data/real_ws/{train,val}.npz
# 2) verify scoring parity + tests (no API keys needed):
uv run pytest -q
# 3) re-score the designated champion from its saved artifacts:
uv run python -m evidence.render --run logs/run_bigclock_07
# 4) rebuild the hidden-test submission from scratch (train-only, ~20 min CPU):
uv run python tools/predict_test_bc07.py
# 5) launch a NEW agent run (needs your own OpenAI key in .env; ~$1/run):
uv run python -m harness.cli run --data-dir data/real_ws \
  --baseline-script zoo/baseline_ws.py --accept-floor 0.0009 \
  --max-iters 14 --context-mode compact --run-dir logs/run_yours
```

Keys: verifying every reported number needs **no keys** (plain Python over
committed artifacts). New runs need an OpenAI key in `.env` (`.env.example`);
spend is hard-capped by an in-code ledger (`BUDGET_USD`).

## Compliance notes

- Train on train; validation used for tuning/selection only (organizer-endorsed).
- The hidden-test window's **labels are never read**: the export writes features
  only and the predictors assert the archive contains no label-like arrays.
- ~45 development runs are fully disclosed with journals; the submission is the
  clearly designated best run, per the organizers' webinar guidance.
- Zero manual interventions in all designated runs (machine-counted).

## Limitations & what we'd do with more time

- Ensemble payoff is config-dependent (it subtracted on two strong singles);
  making the close reliably positive (snapshot members, member-diversity
  design) is implemented but under-tested.
- The convergence rule ends runs while genuinely improving (all real gains here
  are sub-ε); our fan-out/bundling doctrine mitigates but a principled
  sequential-testing stop rule would be better science.
- Sonnet-as-proposer found a novel working package (click-aux rider, 0.60477)
  but its reply-length reliability lags; a mixed-provider proposer ensemble is
  the obvious next step.
- 27K was only demonstrated out-of-protocol; a compliant 27K agent run needs
  either subsampled calibration or bigger hardware.

## Team contributions

(fill in per member before submission)
