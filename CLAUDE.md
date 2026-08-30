# mle-agent — agent working notes

TikTok TechJam 2026 Track 2: autonomous ML research agent on KuaiRand.
Deadline: TUESDAY 1 Sep 12:00 noon SGT (1 Sep 2026 is a Tuesday — verified) (Devpost). Team repo: yxshrk/jitai_ml_agent,
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

## State (Sun 30 Aug, late night — post-context-clear pickup)
DESIGNATIONS (final unless a live run beats them):
- Pure: run_bigclock_07, 0.605575 valid (+0.00398 published baseline). Test CSV built
  + validated (predict_test_bc07; reproduced 0.60561). Say "best observed validation
  checkpoint at rule convergence", "no executable seed" — see SUBMISSION_RECIPE.md.
- 1K: run_max_1k_c, 0.6524 (triple-verified: independent evaluator exact match,
  in-run replication 0.65221, fresh seeds 7/99 = 0.648/0.647). Test CSV built +
  validated (predict_test_1k_winner, 2 members seeds 42+1051).
- Predicted hidden test ~0.5977 ± 0.0020 (evidence/PREDICTED_TEST.md); robustness CI
  [+0.0020,+0.0055] (evidence/bc07_robustness.md); post-mortem over 108 runs in
  evidence/CAMPAIGN_POSTMORTEM.md (incl. recovered #2 run bigclock_r4 0.60524).

STILL RUNNING (nohup'd on machines — SURVIVE context clear; watchdog does NOT):
- laptop: run_novel_l1 (champ 0.60524 — temporal-pair-kernel discovery; its CLOSE
  is the last realistic shot above the champion — if it beats 0.60558, re-designate:
  update SUBMISSION_RECIPE/README/RESULTS_AND_RESOURCES, rebuild test CSV from its
  recipe, regenerate site via tools/build_site.py + tools/instrument_weights.py).
- coral (pallav@coral.local ~/techjam/mle-agent ./.venv312/bin/python): run_qb_b
  (champ 0.60466, decayed-positive sampling; close pending).
- ruby (gpubox ~/mle-agent ~/techjam27k/.venv/bin/python): run_final_f1 (dial-jitter
  member-bank close, GPU, long), run_novel_r1 (0.60447, final close pending),
  run_omega_1k (0.64975 climbing — if >0.6524, 1K re-designation same procedure).
FIRST ACTIONS after clear: bash /tmp/fleet_watchdog.sh (script survives; run via
run_in_background) to resume monitoring; collect any summary.json that appeared;
append results to CAMPAIGN_POSTMORTEM addendum + RUNS inventory (tools/audit_runs.py).
A stalled Codex task (blend audit) was superseded — blend audit DONE in-house
(evidence/blend_audit.md: predeclared blends +0.00017, evidence only).

SITE: site/index.html = "Agent's Lab Notebook" v3 (artifact cd989436-...-92d747db8f80;
rebuild: tools/build_site.py <run> + optionally tools/instrument_weights.py; assembly
= inline scripts, see git history). User may request further design iterations.

REMAINING before Tue 1 Sep 12:00 noon SGT deadline (verified Tuesday):
1) Harvest last runs; final designations (ask user); rebuild CSVs if changed.
2) Fresh sanitized zip + ChatGPT endgame-review prompt (user asked for this).
3) qb_d optional (untried cards: listwise-regime, curriculum, relative-watch,
   small-batch) — only if machines free and user wants.
4) README team-contributions (user fills); figures/briefing-artifact refresh;
   RESULTS_AND_RESOURCES regenerate if designations changed.
5) User-side Monday: video (VIDEO_SCRIPT.md), Devpost form (evidence/DEVPOST.md),
   teammate Devpost+form registration, GitHub Pages enable for site/, repo public.
6) Post-competition: rotate BOTH API keys (were pasted in chat), delete KuaiRand
   data copies (rules), spend ledgers ~$95-120 total (caps: laptop+machines $125).
BUDGET: BUDGET_USD=125 in .env (all machines). Effort medium ONLY. Dosing: 1 run
laptop / 2 coral / 3 ruby. Machines map: MACHINES.local.md (untracked).