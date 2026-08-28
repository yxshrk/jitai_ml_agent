# Contracts — frozen after integration

## 1. Experiment spec (agent -> harness)
JSON emitted by the proposer, executed by the harness:
```json
{
  "hypothesis": "BPR pairwise loss should align training with GAUC (expect +0.005-0.01)",
  "expected_delta": 0.0075,             // honest predicted validation-primary delta
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
  "method_selection": {                 // null only for baseline/debug iterations
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
  "realized_delta": 0.0021,            // node primary - champion-before primary; null on error
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
retained. Selector and reflector inputs remain the compact journal. Every journal record
stores the run's `context_mode`, enabling compact/full A/B comparisons.
Policy (harness-owned, not LLM-chosen): 3 initial drafts; on failure debug same node
(max depth 2); otherwise improve the current best node (greedy); forced branch to a
different menu tier after 5 stagnant iterations.
Before every draft/improve proposal, a separately metered selector diagnoses the parent
learning curve and journal, chooses exactly one card from `agent/METHODS.md`, cites it,
and records one rejected alternative. The selected card text and rationale are passed to
the proposer under `## Selected method (implement THIS)`. Debug iterations skip selection
and preserve the failed node's method. Selector and proposer both receive
`streak_state = {no_improve_streak, n_converge, iters_left}` so an N-1 streak favors the
highest-expected-gain untried method over a dosage tweak.
The reflector still runs every 5 iterations and additionally whenever stagnation reaches
3 or more; it receives all METHODS.md card ids so its focus note can re-rank them.
Acceptance: calibrate sigma from 3 baseline seeds; accept if delta >= 2*sigma
(floor 0.002); 0-2sigma -> one reseed confirm run; else revert.
Convergence: OFFICIAL rule — converged when validation primary has not improved by
more than epsilon=0.002 over the last N=3 consecutive COMPLETED iterations (accepted,
rejected, errored all count), vs best-so-far. Journal line 0 = reproduce_baseline
(3-seed calibration doubles as the brief's required baseline reproduction). Every
iteration record carries a unified "diff" vs its parent node (brief requirement).
Harness owns timeouts, validity checks, best-node argmax, stopping.
Every proposer spec must include a finite numeric `expected_delta`: its honest prediction
of the candidate's validation-primary change relative to the champion before that
iteration. The harness records it alongside `realized_delta = node primary - champion-
before primary`; `realized_delta` is null when the iteration errors. Evidence reports show
expected versus realized delta per iteration and mean absolute calibration error over only
the iterations whose realized delta is non-null.
