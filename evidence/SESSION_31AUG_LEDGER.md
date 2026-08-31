# Session ledger: Mon 31 Aug 2026 (evening) — farm-close night

Everything built, measured, broken and fixed in the final-night session, so any
future session can pick it up. Deadline Tue 1 Sep 12:00 SGT. Designations did NOT
change: Pure = run_bigclock_07 0.605575 (clean), 1K = run_omega_1k 0.66892.

## What was built (all on master, pushed to team branch `mle-agent`)

1. **Farm-close capability** (`harness/farm_close.py`, loop + cli + prompts): typed
   JSON ensemble plan from the agent; deterministic harness executor: concurrent
   probe fits of every family -> exhaustive blend enumeration over probe score
   vectors -> anchor-constrained portfolio selection -> parallel full trains ->
   full-stage candidate table (singletons + blends + incumbent) -> frozen recipe.json
   -> re-verify -> emit. Honest fallbacks recorded (`fallback_to_singleton`,
   `fallback_to_incumbent`; incumbent fallback is NOT sent to the no-op fixer).
   Envelope discriminator `execution_kind: script|farm_close` (alias
   `ensemble_plan`), legacy code-only proposals normalize to script. Plan repair
   path replaces the code fixer for plans. Admission floor enforced for blends.
   Alignment keys on row_id+user (video_id encodings differ per family).
   **Anchor rule (last fix of the night):** members[0] MUST be the run's champion
   node script via `script_source` (the loop passes the path as ANCHOR SCRIPT);
   other members are `code` derived from it. Rationale: f6 iter 4 rewrote all four
   members as 46-85 line toys vs the 458-line champion and lost the base.
2. **Resume-from-run** (`--resume-from <run_dir> --at N`): continue a prior run from
   just before iteration N with reconstructed state (nodes, champion, official
   streak, sigma from calib metrics, journal one-liners); artifacts copied; lineage
   in `resume.json` + summary `designation_eligible: false`; clean-mode refused.
   Codex reviewed it (12 findings); the ones affecting fidelity were fixed.
3. **`--no-cross-run`**: memory-free runs (no CROSS_RUN.md read/write).
4. **`AGENT_PROMPT_LOG_DIR=<dir>`**: every LLM call logged verbatim
   (system/user/reply) — this is how root causes were found; keep using it.
5. **History normalizer** (`harness/loop.normalize_history`): sweep-shaped or
   alias-keyed histories are flattened to the contract curve, so the selector is
   never blind (root cause of every "insufficient-telemetry" endgame in f2-f4r).
6. **Transport retry** (one retry, 900 s) — a slow OpenAI call must not cost an
   iteration. Unparseable proposer replies are persisted (`<node>_raw_reply.txt`).
7. **Selector principles added** (agent/prompts.py, general, no canned answers):
   epsilon arithmetic under streak pressure; margin over bare reach; headroom read
   against the CURRENT best; measured gain-class playbook per phase (gated to
   full-knowledge mode only); state discipline (journal = current-run facts).
   Stale steer "a gated ensemble close is usually the finisher" REMOVED.
8. **Evidence corrections** (agent/METHODS.md): composite card's headline number
   is the in-run level (0.6043-0.6046); teammate 0.606116 relabeled human-assisted;
   farm-close card precondition: closing move, never an opener; members derive
   from the champion. Clean library gained a literature-only cross-family card
   (`heterogeneous-ensemble-design` in METHODS_CLEAN.md, no campaign numbers).
9. **Benches**: `tools/bench_farm_close.py` (13/13 on the four real champion
   vectors; 4-way blend reproduces the audit 0.605639 exactly);
   `tools/decision_bench.py` now 9 scenarios incl. every live failure state
   (endgame_eps_math=f1, endgame_margin_not_reach=f2, endgame_unspent_package_trap
   =f4, f4r_exact_state = verbatim live prompt + real curve, opening_expected_value
   = f5 opener with farm-as-opener BAD); plan-emission check reports member source
   kinds and whether the anchor is the champion script; `tools/bench_scenario.py
   <name> <reps>` runs one scenario; `tools/bench_model_sweep.py` sweeps model x
   effort (temperature is rejected by gpt-5.6 models). Clean bench 5/5 (+farm 2/2).
   Weak-base endgames: a strong package is neutral, not bad (recorded in file).
10. **Site**: light graph-paper redesign (Switzer + PP Mori), presenter mode
    `site/present.html` (11 keyed scenes in Rohan's script order),
    `site/RECORDING_CHEATSHEET.md`, clarity pass (headline "An agent that runs its
    own ML experiments."), `site/CLAUDE.md` = site briefing for any agent.
11. **Docs**: `evidence/SCRIPT_ROHAN.md` (V1 verbatim, V2, V3 filled+verified,
    corrections log, shot list); `evidence/DEVPOST.md` reconciled (no placeholders);
    RESULTS_AND_RESOURCES final designations; SUBMISSION_RECIPE 1K = omega_1k;
    rejected CSVs quarantined in `evidence/rejected/` (untracked, >100 MB);
    CLAUDE.md untracked from the repo (internals); README contributions skeleton.

## Model/effort findings (logs/bench_sweep_*.out, bench_model_sweep.json)
sol@medium 6/6 (incl. the exact live state 3/3); sol@low fails the endgame
arithmetic; terra 6/6 on synthetic fixtures but 0/3 on the real f4r state -> NOT
the selector; luna degrades; temperature -> HTTP 400. Keep sol@medium.

## Live runs tonight (all in logs/RUNS.md)
f1 0.60464 (endgame: same-family close, ceiling<eps) -> f2 0.60364 (did the math,
ignored margin) -> f3 0.60314 (CHOSE farm-close; plan died on a guessed script
path) -> f4 0.60362 (unspent package over close: headroom misread) -> f4r killed
(composite via mislabeled 0.606116 + blind curve) -> f4r2 0.60450 (FIRST live
autonomous farm-close; singleton fallback) -> f5 killed (farm as opener; cold
members crashed) -> f6 0.60424 (right decisions throughout; opener code defective;
close members were toy rewrites) -> f6r resumed@4 with anchor: blend 0.60497 beat
best member 0.60443 but failed seed-repeatability at the gate; ledger cap ended it
-> f7 launched 00:30 with the full stack and $200 cap (in flight at time of writing).

## Open engineering items (post-deadline unless f7 forces them)
- Implementation variance is the last bottleneck: reference implementations on
  cards; smoke-stage sanity gates (probe must beat baseline probe, GAUC>0.5);
  diff-against-parent discipline for "improve" moves; agent-written self-tests.
- Grey-zone confirm of a farm node re-runs the whole farm with reseeded members
  (3x cost, and reseeded members can be weaker than the accepted anchor). The
  external plan asked for "replay the frozen recipe"; consider comparing the
  reseeded blend against the reseeded anchor, not the original incumbent.
- Farm executor: blend scope both per_user+global doubles enumeration; fine.

## How to run the final-night configuration
```
# ruby (gpubox), from ~/mle-agent (rsync from laptop first, excluding logs/data/evidence/site)
nohup env PATH=~/techjam27k/.venv/bin:$PATH OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  AGENT_REASONING_EFFORT=medium AGENT_MAX_CODE_TOKENS=28000 \
  AGENT_PROMPT_LOG_DIR=logs/run_X/prompts \
  ~/techjam27k/.venv/bin/python -u -m harness.cli run --data-dir data/real_ws \
  --baseline-script zoo/baseline_ws.py --no-cross-run --accept-floor 0.0009 \
  --confirm-runs 2 --timeout-s 7200 --max-iters 16 --context-mode compact \
  --run-dir logs/run_X > logs/run_X.out 2>&1 < /dev/null &
# resume a failed run at iteration N (experiment lineage, not designation-eligible):
#   add: --resume-from logs/run_X --at N
# benches: uv run python tools/decision_bench.py --n 2 ; tools/bench_scenario.py <name> 3 ;
#          tools/bench_farm_close.py ; tools/decision_bench_clean.py
# kill safely: ssh gpubox 'pkill -f "run_farm_[f]7"'  (bracket trick; NEVER put the
#   literal run name elsewhere in the same command line: pkill self-matches)
```
Budget caps: laptop .env and ruby .env BUDGET_USD=200 (raised twice tonight).
Post-deadline: rotate both API keys, delete KuaiRand data copies, terminate cpupod.

## Added after 00:40 (same night)
- Retry-close principle in selector guidance + fixture `close_rejected_strengthen_first`
  (f6r post-rejection state); verification result in logs/bench_retryclose.out on ruby.
- tools/watch_run.sh and tools/harvest_run.sh (repo-resident; /tmp scripts vanish).

## NEXT HARNESS CHANGE (spec for whoever picks this up): fair confirm for farm nodes
Problem: the grey-zone confirm reruns a farm node script with a new --seed; the
wrapper re-executes the WHOLE farm (probes + selection + full trains) with all
members reseeded, and the acceptance z-test compares the reseeded blend to the
ORIGINAL incumbent. Reseeded members drift ~0.0005, the blend loses to the incumbent,
executor falls back, confirm primary == parent, gain "not repeatable" (f6r iter 4:
blend 0.60497 vs best member 0.60443 rejected this way). Cost is also 3x a farm node.
Fix: in harness/farm_close.run_plan, when execution_seed != base_seed (a confirm
rerun) AND a frozen recipe.json exists for the base seed in a sibling node dir,
REPLAY the frozen recipe: keep the anchor member (script_source) at its original
seed, reseed only the code members, skip probes/selection, retrain the recipe's
members, blend with the recipe's weights, emit. Then in Loop.acceptance, for farm
nodes compare the confirm blend against the confirm-run's own best singleton as
well as the incumbent, and require the blend > both (repeatability of the BLEND
effect, not of an absolute number). Tests: a FakeBrain farm run where confirm
replays (no probe phase in confirm dirs) and a blend accepted when it repeats.
Estimated 45-60 min. Not started.

## 1 Sep ~02:30 — f8-reach exhibit (human-assisted, EVIDENCE ONLY)
Manual per-user rank-average of saved validation predictions (no training):
f8 node_002 (context-stratified pairs, 0.604672 by provisional eval) +
bigclock_07 node_006 (0.605575) + novel_l1 node_004 (0.605146) = 0.605765;
without the f8 member the best pair reaches only 0.605488, so f8's champion is
blend-complementary (+0.0003). Best of 11 combos on val (selection optimism
~0.0001-3). Human-assisted tier, NOT designation-eligible; shows the state f8
reached extends past the designation with the standard close its 503-killed
iteration 4 would have attempted. Script: inline (this session); inputs are the
four predictions.csv files named above.

## 1 Sep ~03:10 — f9 endgame exhibits (human-assisted, EVIDENCE ONLY)
f9 converged at 0.605102 (4 iters; single-model champion from a 48-probe sweep).
Manually executed its declined endgame moves: 5-seed ensemble of the champion
config (seeds 42-46, singles 0.60498-0.60530) = 0.605303 z-avg / 0.605225
rank-avg — only +0.0002, NOT the digest's +0.0004..+0.001 (diminishing returns:
the tuned single had already banked most ensemble profit; bigclock's +0.0014
came off a weaker 0.6042 single). All-own-artifacts blend (5 seeds + rejected
ctx-pairs node) = 0.60546. CONCLUSION: even perfect endgame banking leaves f9
~0.0001-0.0005 short of the 0.605575 designation; the single-run ceiling
0.6055-0.6060 stands. Cross-run blend WITH other runs' artifacts (earlier
exhibit) = 0.605765 remains the only number above the designation tonight.

## 1 Sep ~03:40 — f8 iteration-4 counterfactual EXECUTED (evidence only)
The selector's 3/3-consistent pick for f8's 503-killed iteration
(social-mtl-heads aux bundle, emitted as 20 edit hunks on f8 node_002, 94%
preserved) was trained fully on the laptop: primary 0.605087 (+0.00037 over
f8's champion; sub-floor, would have been grey-zone). Artifacts:
logs/counterfactual_f8_iter4_spec0.json, logs/counterfactual_f8_iter4_node.py,
logs/counterfactual_f8_iter4_out/. Confirms: sane decision, modest gain, the
503 did not cost the designation.
