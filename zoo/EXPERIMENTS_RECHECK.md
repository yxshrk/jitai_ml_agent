# Final frozen-stack rechecks

Protocol: validation only, official scorer, full `data/real_ws`, frozen defaults
from `zoo/polish_stack.py`. Seed 42 is explored first. A 42/43/44 confirmation
and the best-pair test are run only when an isolated cell gains at least +0.001000
against the reproduced strong seed-42 control. Every process has a 350-second
training alarm, below the six-minute limit.

## Runs

Results are appended here as each run completes.

### R0 — strong control

Command: `uv run python zoo/polish_stack.py --data-dir data/real_ws
--out-dir /tmp/recheck_control_s42 --seed 42 --max-runtime 350`

| config | seed | GAUC | nDCG@5 | primary | delta vs strong control | best epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| frozen five-field control | 42 | 0.671859 | 0.538138 | **0.604998** | +0.000000 | 3.5 | 30.0s | reproduced strong control |

This measured score, rather than the historical rounded 0.6047 reference, is
the comparison point for every seed-42 trigger decision below.

Preflight note: the first `session` command exited immediately with
`ModuleNotFoundError: zoo` before data loading or training. The new runner's
repository-root import shim was added and the scored invocation below is the
complete rerun; there is no metric to omit from the failed preflight.

### R1 — causal session fields

Config: frozen control plus previous-exposure-gap bucket (7 categories),
within-session index clipped at 31, and session-start flag; causal occurrence-
ordered history exactly follows `audit_campaign.py`. Auxiliary weight is zero.

| config | seed | GAUC | nDCG@5 | primary | delta vs strong control | best epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| session | 42 | 0.671320 | 0.538127 | **0.604723** | **-0.000275** | 5.0 | 45.5s | dead; below trigger |

Seed 42 is below the strong control, so the mandated +0.001 trigger is not met
and no 42/43/44 confirmation is run for this cell.

### R2 — auxiliary click head

Config: unchanged five-field frozen input and shared bottom, plus one click BCE
head at weight 0.1. The main pointwise/BPR objective and all recency weights,
regularization, optimizer settings, and scheduling remain frozen.

| config | seed | GAUC | nDCG@5 | primary | delta vs strong control | best epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| auxiliary click head, weight 0.1 | 42 | 0.671159 | 0.537644 | **0.604401** | **-0.000597** | 5.5 | 36.9s | dead; below trigger |

Seed 42 is below the strong control, so no 42/43/44 confirmation is run.

### R3 — temporal context fields

Config: frozen control plus train-vocabulary-encoded hour bucket (`hourmin // 100`)
and Python weekday (`0..6`). No auxiliary loss is active.

| config | seed | GAUC | nDCG@5 | primary | delta vs strong control | best epoch | runtime | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hour bucket + day of week | 42 | 0.670851 | 0.537213 | **0.604032** | **-0.000966** | 8.0 | 43.2s | dead; below trigger |

Seed 42 is below the strong control, so no 42/43/44 confirmation is run.

## Trigger decision and final verdict

| cell | seed-42 primary | delta vs measured strong control | 42/43/44 confirm | verdict |
|---|---:|---:|---|---|
| causal session fields | 0.604723 | -0.000275 | not run: trigger missed | dead |
| auxiliary click head, 0.1 | 0.604401 | -0.000597 | not run: trigger missed | dead |
| hour bucket + day of week | 0.604032 | -0.000966 | not run: trigger missed | dead |

No isolated cell achieved the required seed-42 delta of +0.001000. Therefore no
cell qualified for three-seed confirmation, and the conditional best-pair test
was not triggered. All scored runs used the full validation export and completed
well below six minutes without timeout or scope reduction.

## 1K flip-check (ruby GPU, Sun ~morning) — neighborhood around frozen tuned config
Single-dial flips off frozen_stack_1k (lr .00168, d .21, wd 3.7e-5, k24, hl7, seed42):
k48 0.61340 | k32+lr1e-3 0.61319 | batch16384 0.61256 | dropout.1 0.62026 |
wd1e-3 0.62104 | hl3.5 0.61721 | hl12 0.61835. (k40d28 cell lost: k=40 not in
choices.) Verdict: NO flip beats champion single 0.6214 — 1K config confirmed
at a local optimum; frozen recipe stands.

## 1K designation run (run_desig_1k_01, Sun morning) — AGENT-DESIGNATED CHAMPION
Seed node failed; agent rebuilt 5-seed ensemble itself (0.63448), then chose to
scale to 10 consecutive seeds (42-51) of frozen_stack_1k -> **0.63874** (node_004,
official oracle). 15-member attempt rejected (0.436, partial members). Converged
at 7 iters. This beats the team-frozen 5-seed 0.6323 and is the 1K submission
champion. Test CSV rebuilt with 10 members.
