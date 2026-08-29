# Contracts — frozen after integration

## 1. Experiment spec (agent -> harness)
JSON emitted by the proposer, executed by the harness:
```json
{
  "hypothesis": "BPR pairwise loss should align training with GAUC (expect +0.005-0.01)",
  "expected_delta": 0.0075,             // honest predicted validation-primary delta
  "expected_delta_basis": "The bpr-hybrid card measured primary 0.6048.",
  "parent": "node_007",               // solution-tree node this builds on, or "baseline"
  "action": "improve",                // draft | debug | improve
  "code": "<WHOLE runnable script>",  // whole-file rewrite, never a diff (research/agent-design.md #3)
  "timeout_s": 600
}
```

## 2. Iteration record (harness -> log, one JSON line per iteration in logs/run_<id>/journal.jsonl)
```json
{
  "n": 7,
  "hypothesis": "...",
  "node_id": "node_007", "parent": "node_006", "action": "improve",
  "code_path": "logs/run_<id>/nodes/007.py",
  "change_summary": "one-line what-changed (the journal line)",
  "context_mode": "compact",           // compact | full proposer context used for this run
  "knowledge_mode": "full",            // full | clean literature-only knowledge used for this run
  "method_selection": {                 // null for baseline; debug preserves its parent's selection
    "diagnosis": "overfit",
    "chosen_method_id": "regularization-schedule",
    "citation": "MENU CURRENT DIRECTIVE",
    "why": "early validation peak calls for the highest-gain untried anti-overfit card",
    "rejected": [{"method_id": "listwise-softmax", "reason": "measured dead at 0.5991"}]
  },
  "metrics": {"gauc": 0.0, "ndcg5": 0.0, "primary": 0.0},
  "val_best_so_far": 0.0,
  "accepted": true,
  "expected_delta": 0.0015,            // proposer's prediction vs champion-before
  "expected_delta_basis": "The regularization card reports 0.604-0.605.",
  "realized_delta": 0.0021,            // node primary - champion-before primary; null on error
  "verdict_note": null,                 // populated when card-reference comparison suspects code
  "failure_stage": null,                // null | "smoke" | "full"
  "fixer_eligible": false,
  "duration_s": 0.0,
  "tokens_in": 0, "tokens_out": 0,
  "error": null,                      // or traceback summary
  "recovery": null,                   // "patched" | "reverted" | "skipped"
  "intervention": false               // true iff a human touched the run
}
```

## 3. Experiment script interface (every zoo/ script and every agent-generated script)
- Reads fixed splits from data/ (never touches the test window; harness enforces).
- CLI: `uv run python <script> --data-dir <d> --out-dir <o> [--seed 42]`
- Writes: `<o>/predictions.csv` (row_id,user_id,video_id,score for the VALIDATION split)
  and `<o>/metrics.json` ({"gauc":..., "ndcg5":..., "primary":..., "history": [{"epoch":1,
  "train_loss":..., "val_gauc":..., "val_primary":...}, ...]} via the official evaluate.py).
  "history" is the per-epoch learning curve — REQUIRED so the agent and reviewers can
  diagnose overfit (val peaks early) vs underfit (still rising) vs dead idea (flat).
- Deterministic given --seed.

## 4. Leakage rule (structural, not prompt-based)
The agent workspace mounts data/train/ and data/val/ only. data/test/ exists solely in
the harness's private dir and is used exactly once, by the final submission step.

## 5. Convergence & caps (official)
epsilon=0.002, N=3 consecutive non-improving iterations; hard cap 50 iterations; 6h wall-clock.
Scored artifact = validation-best checkpoint at convergence.

## 6. Search policy (from research/agent-design.md)
Solution TREE, not chat history: each node = whole runnable script + metrics + one-line
journal entry. `--context-mode compact` (the default) gives each proposer call the task
brief + MENU + journal (one line per node) + the parent node's full code. Never a growing
transcript. `--context-mode full` replaces the proposer journal section with structured
evidence for every retained prior node: hypothesis, action, GAUC/nDCG@5/primary, outcome
(accepted/rejected or the last 5 error lines), change summary, and the last 10 learning-
curve entries. Full context is bounded at approximately 20k tokens (80k characters) by
dropping the oldest optional nodes first; node_000 and the current champion are always
retained. Selector and periodic reflector inputs remain the compact journal. Every
journal record
stores the run's `context_mode`, enabling compact/full A/B comparisons.
Every journal record and `summary.json` also stores `knowledge_mode` (`full` or
`clean`). Clean mode uses `agent/METHODS_CLEAN.md`, replaces MENU with a neutral
dataset/metrics/splits task description, ignores cross-run memory, rejects assisted
seed/custom-draft inputs, and disables only card-reference suspicion routing.
Policy (harness-owned, not LLM-chosen): 3 initial drafts; on failure debug same node
(max depth 2); otherwise improve the current best node (greedy); forced branch to a
different menu tier after 5 stagnant iterations.
With opt-in `--plan-budget`, immediately after calibration the harness makes one extra
`reflector` call with the official convergence and acceptance rules, `max_iters`, and the
calibration result. Its requested initial draft count is clamped to 2..6 and replaces the
fixed three-slot opening. Its ordered card-family preferences are advisory to the selector,
which must explain any deviation in `why`. The raw plan and clamped count are recorded as
the iteration-0.5 `action: "plan"` journal record and persisted under `exploration_plan` in
`summary.json`. Without the flag, the call, record, summary field, and selector note are absent.
Before every draft/improve proposal, a separately metered selector diagnoses the parent
learning curve and journal, chooses exactly one card from `agent/METHODS.md`, cites it,
and records one rejected alternative. The selected card text and rationale are passed to
the proposer under `## Selected method (implement THIS)`. Debug iterations skip selection
and preserve the failed node's method. Selector and proposer both receive
`streak_state = {no_improve_streak, n_converge, iters_left}` so an N-1 streak favors the
highest-expected-gain untried method over a dosage tweak.

Runs declare `--dataset {pure,1k}` (default `pure`). Every method card has
`status_pure` and `status_1k`; the selector sees only the active dataset's status,
and harness eligibility parses that same line. Thus a Pure `measured-dead` card
remains eligible on 1K when its 1K status is `untried`. The frozen-stack validation
references are 0.6047 on Pure and 0.6134 on 1K (literal default, seed 42). The 1K
recency half-life 3/14 regressions are measured-dead for that dataset; Pure-only
popularity findings do not close their corresponding 1K direction.

At run start, the harness reads the final approximately 40 lines of
`logs/CROSS_RUN.md` and supplies them to selector and proposer under
`## Prior runs (do not repeat failed openings)`. Selection prefers cards and
directions not already attempted on the same dataset. At run end, the harness
appends a compact block containing run directory, dataset, stop reason, best
primary, and one row per iteration with method id, eight-word hypothesis summary,
primary, and verdict.

Each method card has a parsed `treats` family list and a `reference_primary` line whose
value is a float or `none`. When a completed, selected-card node scores more than 0.002
below that card's numeric reference, the harness does not reject it immediately: it sets
status `suspect_implementation`, records `verdict_note` exactly as `below card reference
(X vs Y) — implementation suspected` (X and Y formatted to four decimals), and routes
the next iteration to debug that node while respecting max debug depth 2. The debug child
preserves the selected method. If that debug result is also more than 0.002 below the
same reference, it is rejected; it is not routed into another reference-suspicion cycle.

Initial and forced draft selection is portfolio-diverse. The harness unions the `treats`
families of cards selected for all prior draft nodes in the run and passes the sorted list
as `excluded_families` to the selector. A card is eligible only when none of its families
is excluded, unless no non-`measured-dead` eligible card remains. If a draft selector
violates the constraint while an eligible card exists, the harness retries it once with
an explicit strict instruction. If the retry also violates it, the harness overrides the
choice with the eligible non-measured-dead card having the largest explicit numeric upper
bound in its `expected_gain / cost` line and records `harness_override: true` in the
method selection. Improve selections receive an empty exclusion list and are not subject
to the retry/override rule.

Every proposer spec must include a finite numeric `expected_delta` and a non-empty
`expected_delta_basis`. The basis is one sentence citing the concrete measured evidence
the estimate extrapolates from: a specific method-card value or a specific journal line.
The harness records both fields alongside `realized_delta = node primary - champion-
before primary`; `realized_delta` is null when the iteration errors. Evidence reports show
expected versus realized delta per iteration and mean absolute calibration error over only
the iterations whose realized delta is non-null.

For `action == "improve"`, the proposer emits the whole parent file but makes the smallest
coherent change needed to test its one hypothesis. Unnecessary rewrites are defects; this
constraint does not change the whole-script JSON output format.

Every draft or improve node has two-stage execution. Before its full run, the harness runs
the same script with `SMOKE_EPOCHS=1` and a fixed 120-second timeout, requiring a zero exit
and valid `predictions.csv` plus `metrics.json`. Nonzero exit, timeout, missing outputs, or
invalid metrics immediately marks `failure_stage="smoke"` and `fixer_eligible=true`; no
full-run budget is spent. The existing fixer gets one attempt, and patched draft/improve
code must pass a fresh smoke before its full run. Scripts must parse `SMOKE_EPOCHS` as an
integer and cap every training phase's epoch count accordingly. A script that ignores the
variable is allowed: if it completes with valid outputs within 120 seconds, smoke passes.
Debug nodes keep the existing single-stage execution because they are themselves the
fix attempt.

The reflector still runs every 5 iterations and additionally whenever stagnation reaches
3 or more; it receives all METHODS.md card ids so its focus note can re-rank them.
After the run stops, one additional call using the `reflector` role receives the full
compact journal summary and critiques harness/policy choices, its own scaffold, and the
best opening for the next run. This `self_critique` is stored in `summary.json` and
appended to `logs/CROSS_RUN.md`; it is archival only and is never auto-applied.
Acceptance: calibrate sigma from 3 baseline seeds; accept if delta >= 2*sigma
(floor 0.002); 0-2sigma -> one reseed confirm run; else revert.
Convergence: OFFICIAL rule — converged when validation primary has not improved by
more than epsilon=0.002 over the last N=3 consecutive COMPLETED iterations (accepted,
rejected, errored all count), vs best-so-far. Journal line 0 = reproduce_baseline
(3-seed calibration doubles as the brief's required baseline reproduction). Every
iteration record carries a unified "diff" vs its parent node (brief requirement).
Harness owns timeouts, validity checks, best-node argmax, stopping.
