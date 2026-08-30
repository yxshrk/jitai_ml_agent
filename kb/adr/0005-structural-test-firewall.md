# ADR-0005 — The test split is walled off structurally, not by prompting
Status: Accepted in principle (2026-08-30); implementation pending

## Context
Test rows and labels are in the public download; `baseline.py` prints test scores. The rules forbid using them.
## Decision
The agent's workspace contains train and valid only. Test features live in a harness-private directory, are used
exactly once to produce the final CSV, and no test metric is ever computed for our models. Access attempts are logged.
## Consequences
Leakage becomes impossible by construction — a robustness/autonomy point, and the only defence against an
invalidated submission.
