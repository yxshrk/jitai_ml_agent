# ADR-0007 — Accept/reject, best-checkpoint selection, and stopping are deterministic code, never an LLM
Status: Proposed (2026-08-30)

## Context
"Is +0.0018 an improvement?" is a comparison against ε. An LLM asked that question will sometimes talk itself into
accepting noise; over 50 iterations that drifts the run.
## Decision
A code-only referee runs `evaluate.py` unmodified on the validation split (multi-seed when a delta is in the grey
zone), applies the ε rule, keeps the argmax checkpoint, and stops the run. The LLM roles propose, implement,
critique, and write — they never judge scores.
## Consequences
Free robustness points; the journal's metrics are trustworthy by construction.
