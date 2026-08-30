# mle-agent — agent working notes

TikTok TechJam 2026 Track 2: autonomous ML research agent on KuaiRand.
Deadline: Mon 1 Sep 12:00 noon SGT (Devpost). Team repo: yxshrk/jitai_ml_agent,
push to branch `mle-agent` (git push origin master:mle-agent). NO Claude
attribution in commits.

## Read in this order
1. `../RULES.md` — competition pickup file: webinar rulings, decisions, runbook.
2. `../PROBLEM_STATEMENTS.md` — full official brief (authoritative rules copy).
3. `../webinar-transcript-28aug.md` — verbatim organizer Q&A (compliance receipts).
4. `SUBMISSION_RECIPE.md` — designated champions + compliance ruling. THE decision doc.
5. `CONTRACTS.md` — node-script contract (§3) + search policy (§6).
6. `agent/METHODS.md` — the agent's knowledge: cards + digest + search/depth/clock
   policies. `agent/METHODS_CLEAN.md` = literature-only A/B variant.
7. `PRACTICES.md`, `LEVERS.md`, `MENU.md` — measured A/B decision tables / ledger.
8. `zoo/EXPERIMENTS*.md` — campaign ledgers (every measured cell, with seeds).

## Layout
- `harness/` loop.py (iteration loop, acceptance, official convergence), cli.py.
- `agent/` brain.py (LLM calls; AGENT_REASONING_EFFORT / AGENT_MAX_CODE_TOKENS env),
  prompts.py, models.toml, budget.py ($ ledger logs/spend.json, cap from .env).
- `zoo/` frozen champion scripts + experiment runners. `data/` exporters + real_ws*
  (gitignored npz). `evidence/` figures, DEVPOST.md, submission.py, test CSVs.
- `logs/run_*/` one dir per agent run: journal.jsonl (per-iteration record),
  nodes/NNN.py, node_*/metrics.json (+progress.log in fan-out nodes), summary.json.
- `tools/fleet_status.sh` — live view of runs on laptop + coral + ruby.

## Machines
Runs execute on the primary laptop plus two optional helper machines reachable over
ssh (aliases and paths are user-local — see the untracked `MACHINES.local.md` if
present, or ask). Sync helper copies with rsync (NOT git), excluding
.git logs data/real_ws* evidence .venv*.

## Conventions
- Python: uv on laptop. Every run costs $ — ledger cap in .env (BUDGET_USD).
- Effort: medium ONLY (high/xhigh measured: proposals truncate/fail — see runs 16-22,27,28).
- All numbers quoted anywhere must have a run dir or EXPERIMENTS ledger line behind them.
- After meaningful changes: commit + push to team branch; keep SUBMISSION_RECIPE.md and
  this file current.

## State (Sun 30 Aug pm)
Pure DESIGNATED: 0.60558 run_bigclock_07 (coral) — FULLY UNSEEDED, beats seeded
0.60513; recipe in SUBMISSION_RECIPE.md; test CSV rebuild pending. 1K: 0.63874 agent-designated
(run_desig_1k_01, 10-member). 27K bonus demo: 0.67263. Test CSVs built (labels never
read). Remaining: finish fleet, final designation + clean rerun, resource table,
Devpost + video (user).
