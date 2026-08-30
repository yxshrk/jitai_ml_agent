# ADR-0002 — Build the knowledge base before the harness
Status: Accepted (2026-08-30)

## Context
The convergence rule (ε = 0.002 over 3 iterations) makes metered iterations the scarce resource.
## Decision
Move as much search as possible out of the metered loop and into build time: a spec layer (`kb/spec`), measured
data facts (`kb/data`), method cards (`kb/methods`), a run ledger (`runs/`), and cross-run lessons (`kb/lessons`).
## Consequences
Every question answerable from the spec, EDA, literature, or a proof is answered before iteration 1.
