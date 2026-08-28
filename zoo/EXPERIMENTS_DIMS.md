# Under-swept dimensions campaign — validation only

## Protocol

- Control: exactly five offset-encoded NPZ fields; DCN-lite with one cross layer,
  MLP width 128 and dropout 0.1; `0.5 * within-user BPR + 0.5 * logloss`; click and
  effective-view auxiliary heads at weight 0.2 each; Adam at `1e-3`; and seven-day
  recency weights normalized to mean one.
- Validation runs use only `data/real_ws/train.npz` and `val.npz` (plus matching
  CSV columns solely for raw video ids and auxiliary outcomes absent from NPZ).
  Checkpoints are scored with `data/official/evaluate.py` every half epoch and
  selected on official PRIMARY. `metrics.json` records the full run history.
- Exploration uses seed 42. Any cell at least +0.002 over the fixed 0.6016 PRIMARY
  baseline is confirmed at seeds 42, 43, and 44 before being called a real win.
- The runner enforces a 330-second training alarm, leaving margin below the strict
  six-minute per-run cap. Every final CSV is independently rescored by the official
  evaluator before a sweep is closed.
- Requested outcomes `follow`, `comment`, and `forward` are not columns in either
  workspace NPZ or CSV. Their cells are retained below as explicit failures rather
  than being omitted or populated with invented targets.

## Sweep 1 — BPR pair sampling

Pending.

## Sweep 2 — auxiliary task set

Pending.

## Sweep 3 — sparse-embedding optimizer

Pending.

## Final summary and per-sweep conclusions

Pending completion of all cells and confirmations.
