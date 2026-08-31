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

## State (Mon 31 Aug ~02:00, live session)
DESIGNATION POLICY (settled with Rohan 31 Aug): prefer CLEAN runs (no --seed-scripts).
VERIFIED 31 Aug ~02:15: bigclock_07 IS CLEAN — journal roots baseline->node_000, no
seed nodes, bigclock_queue.sh has no --seed-scripts (Claude briefly mis-stated it as
seeded from a runbook template; run record + memory both say unseeded. Check the RUN
RECORD, not command templates). So current designation already satisfies the policy.
Method CARDS are the sanctioned knowledge channel (brief: "drawing on established
methods"); seed scripts remain legal disclosed config but are for experiments only
(combo_r1), never the designation exhibit. temporal-pair-kernel was INVENTED by
novel_l1 (not in its card library) — card written after the fact from its journal.

TERMINOLOGY (do not re-confuse):
- "search 40-80" = IN-NODE probe budget: agent/METHODS.md digest — opener quality
  flattens past ~50-80 well-ranked full-fidelity probes; stage-1 40-80 probes with
  stop-early after 15 non-improving; then ~10-15 refinement probes. Already policy.
- "k≈48" = EMBEDDING WIDTH capacity peak on KuaiRand-1K only (logs/k1_bigk). On Pure
  k=16 is measured optimal. Neither number is an iteration cap (official rule governs).

DESIGNATIONS (unchanged so far):
- Pure: run_bigclock_07 0.605575 (CLEAN, verified). Test CSV built (predict_test_bc07).
- 1K: run_max_1k_c 0.6524. Test CSV built (predict_test_1k_winner).

FLEET (watchdog /tmp/fleet_watchdog.sh, relaunch via run_in_background after any clear):
- laptop: run_novel_l1 — CLEAN, 0.60524 (temporal-pair-kernel discovery), in ensemble
  close (node_005 jitter members ~0.6037-0.6046). If close > 0.60558 => designate (clean!).
- coral: run_clean_c1 — CLEAN, no seeds, enriched 44-card METHODS.md (new cards below).
  mcsweep DONE: member count FLAT on Pure (k=3..12 all 0.6048-0.6050, ±0.0003 = noise);
  verdict = keep predeclared 5-seed recipe (evidence/blend_audit.md).
- ruby: run_combo_r1 — SEEDED experiment (seed_pairkernel + frozen_stack): tests whether
  mechanisms compose; likely NOT the designation per policy above; keep for the report.
  Also run_omega_1k 0.64975 (needs >0.6524; unlikely).
- Concluded tonight (harvested, logs/RUNS.md): qb_b 0.60466, novel_r1 0.60447 (both
  clean), final_f1 0.60403. Lessons: winning shape = strong package -> ensemble close
  (+0.0013); drafts overpromise; ensemble closes underpromise/overdeliver; eps kills
  +0.0004 steps from high base; novelty = 1-2 lottery slots.

NEW METHOD CARDS (agent/METHODS.md, appended 31 Aug, synced to coral): temporal-pair-
kernel (0.60524 evidence), gauge-fixed-bce (0.60447), decayed-positive-sampling
(0.60466), heterogeneous-ensemble-design (untried; blend-audit caveat), snapshot-
ensemble (untried). combo_r1/novel_l1 launched BEFORE these cards existed.

SITE: site/ = "Flight Recorder" v4.1 (2D scrollytelling: hero self-drawing chart, loop
diagram, pinned mission-log replay from rundata.js, memorization evidence panels from
weights.js via tools/instrument_weights.py, receipts). 3D starscape REMOVED (built,
then cut as unintuitive — do not resurrect without asking; build_space.py deleted).
Rebuild data: tools/build_site.py <run> (also emits rundata.js). Serve: python3 -m
http.server (fetch-free, file:// safe). Local preview server may be on :8642.

ENDGAME DONE: sanitized zip ~/Desktop/mle-agent-endgame.zip (34MB, code+docs+key
journals, no secrets) + ChatGPT Pro prompt (delivered via clipboard; re-copy: see
session log) — Rohan consulting ChatGPT for new mechanism ideas -> add as cards.
SPEND: Rohan says real total ≈ $61; per-machine ledgers double-count (sum ~$200, wrong).

NEXT BUILD (user-approved, post-context-clear priority): the PARALLEL FARM-CLOSE node
as a first-class capability — one node: (P1) probe-fidelity members of 5-6 diverse
families trained CONCURRENTLY (~6-8 min on ruby), evaluate all blend combos of probe
scores (nearly free) to map complementarity; (P2) full-train the 2-3 complementary
families in parallel; (P3) full-fidelity re-verified blend, emit. Feasibility: fits one
2h node / 6h run; validation-tuning is rules-legal (organizer: tune on validation);
honest transferable ceiling ~0.6059 (last ~0.0005 of any searched blend = selection
shine, per measured winner's-curse). ChatGPT Pro build-plan was requested (prompt via
clipboard); implement per that plan + external review's "typed ensemble plan +
harness-owned assembler" suggestion. Card already carries the 3-phase description
(diverse-family-farm-close).

CONTEXT-CLEAR PICKUP: remote nohup runs SURVIVE clears; local watchers DIE — re-arm
with `bash /tmp/fleet_watchdog.sh` (ruby: z1 z2 showcase pure_clean2) and
`bash /tmp/cpupod_watch.sh` (pod: c3 pure_clean) via run_in_background; /tmp/jsweep.py
is the status helper (scp'd on both machines). Harvest pattern: on DONE -> pull
summary+journal (rsync --exclude workspace --exclude '*.pt'), append logs/RUNS.md,
commit+push. cpu_c1/c2 already harvested.

CURRENT STATE (Mon 31 Aug ~18:00 — read this first; RUNS.md = scoreboard, postmortem = lessons):
- DESIGNATIONS unchanged: Pure bigclock_07 0.605575 (clean; CSV built predict_test_bc07);
  1K omega_1k 0.66892 (triple-audited; submit FAITHFUL A-form CSV test_submission_1k_faithful.csv
  local/gitignored; B-form 0.6802 = evidence only, NO exceptions). ~150 challenger runs
  have not beaten Pure; measured single-run ceiling 0.6055-0.6060 (cross-family close).
- LIVE RUNS: ruby: z1 (0.60356 base), z2 (0.60514 base — best live position), showcase
  (final full stack, 0.60282 base), pure_clean2 (literature-only + research DOCTRINE,
  calibrating — the purity exhibit). cpupod (157.157.221.30:45155, $0.56/hr, TERMINATE
  post-submission): cpu_c3 (0.60437 base), pure_clean (old-doctrine clean, 0.60284).
- TODAY'S SYSTEM WORK (all merged to master + pushed unless noted): floor-v2 grey accept
  (0.0005 + z repeatability heuristic); two-tier menu (treatments vs evidence-ranked
  opportunities, kind field); strategy layer + phase guidance; insufficient-telemetry
  diagnosis; ensemble contract + no-op-collapse fixer (transactional); smoke 360s; env
  thread-cap passthrough; stale-knowledge purge (MENU directive, 17 statuses); pruned
  library (42 cards); seq-deepfm cards from Adi's verified research (0.606116 exact-
  verified from his saved scores); cross-family blend evidence (best combos 0.6060-65,
  selection-free 0.60586) -> hetero + farm-close STRATEGY BUNDLE cards; recovery-
  journaling fix (cherry-picked). UNMERGED branch harness-v3-audit = structured verdict
  fields + all-mode dead-card enforcement + 8 tests (parked; report exhibit).
- EVAL TOOLING: tools/decision_bench.py (full-knowledge selector bench, 8/8 after
  evidence fixes; LIMITATION: cannot distinguish from always-pick-close constant policy),
  tools/decision_bench_clean.py (clean-mode strategy bench 4/10->10/10 via methodology
  only), tools/decision_replay.py (choices over cached artifacts -> 0.60599 = OPTIMISTIC
  CACHED-ARTIFACT COUNTERFACTUAL, not an autonomous result). Bench constitution: fixes
  may only be evidence corrections or general principles, never state-specific answers.
- EXTERNAL REVIEWS adopted (3x ChatGPT Pro in Downloads/, 2x codex audits): artifact
  policy, tie audit (stable), probe manifest, claim tempering, framing = "autonomous
  research needs accurate memory and verified execution"; autonomy spectrum presented
  as PROVENANCE TIERS (literature-only ~0.604 / measured-memory 0.6056 / human-assisted
  0.6059-0.6065 dev evidence), never a causal ablation.
- MACHINE OPS: ruby=gpubox (~/mle-agent, ~/techjam27k/.venv python); cpupod ssh -p 45155
  root@157.157.157... (see above IP); coral RETIRED (disk); laptop idle. Watchdog
  /tmp/fleet_watchdog.sh (relaunch via run_in_background after each event); jsweep
  helper /tmp/jsweep.py on machines. pkill -f MUST use [b]racket patterns or pid loops
  in a SEPARATE ssh from any text containing the pattern (self-match killed 4 sessions).
  Brain caches knowledge AT LAUNCH (mid-run file edits do NOT affect live runs; earlier
  "re-read per iteration" taint doctrine was wrong — discards stay disclosed).
- RunPod: API key in session history; ~$5 spent of $30; GPU pods = oversubscribed CPU
  (unusable, measured ~1 effective core) — CPU pods fine. TERMINATE pod + rotate BOTH
  API keys + delete KuaiRand data post-submission.

REMAINING before Tue 1 Sep 12:00 noon SGT:
1) Harvest novel_l1/clean_c1/combo_r1/omega_1k + mcsweep verdict; final designations
   (policy above; ask Rohan); rebuild test CSV via designated run recipe if changed.
2) README team-contributions (Rohan fills); RESULTS_AND_RESOURCES + figures refresh if
   designations change; Devpost (evidence/DEVPOST.md) + video (VIDEO_SCRIPT.md,
   optional per webinar) Monday; GitHub Pages enable for site/; repo public.
3) Post-comp: rotate BOTH API keys; delete KuaiRand data copies (rules).
