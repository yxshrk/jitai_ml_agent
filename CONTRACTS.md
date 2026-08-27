# Contracts — frozen after integration

## 1. Experiment spec (agent -> harness)
JSON emitted by the proposer, executed by the harness:
```json
{
  "hypothesis": "BPR pairwise loss should align training with GAUC (expect +0.005-0.01)",
  "base": "zoo/fm_torch.py",          // script to start from, or "previous"
  "diff": "<unified diff or full file>",
  "timeout_s": 600
}
```

## 2. Iteration record (harness -> log, one JSON line per iteration in logs/run_<id>.jsonl)
```json
{
  "n": 7,
  "hypothesis": "...",
  "diff_summary": "...",
  "diff_path": "logs/run_<id>/diffs/007.patch",
  "metrics": {"gauc": 0.0, "ndcg5": 0.0, "primary": 0.0},
  "val_best_so_far": 0.0,
  "accepted": true,
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
  and `<o>/metrics.json` ({"gauc":..., "ndcg5":..., "primary":...} via the official evaluate.py).
- Deterministic given --seed.

## 4. Leakage rule (structural, not prompt-based)
The agent workspace mounts data/train/ and data/val/ only. data/test/ exists solely in
the harness's private dir and is used exactly once, by the final submission step.

## 5. Convergence & caps (official)
epsilon=0.002, N=3 consecutive non-improving iterations; hard cap 50 iterations; 6h wall-clock.
Scored artifact = validation-best checkpoint at convergence.
