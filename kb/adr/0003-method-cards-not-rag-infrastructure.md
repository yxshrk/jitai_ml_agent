# ADR-0003 — Structured method cards with a build-time literature crawl, not vector/graph RAG infrastructure
Status: Accepted (2026-08-30)

## Context
The relevant literature is small and mostly known; what the agent needs per iteration is "given this diagnosis,
which method applies, what does it cost, and has it been tried under these conditions?" — a filter, not a
similarity search. Retrieval that happens inside a timed run costs scored tokens and wall-clock.
## Decision
One markdown card per method with a fixed schema (`applies_when`, `mechanism`, `expected_delta`, `cost`,
`composes_with`, `conflicts_with`, `status`, `evidence`). Cards link to each other, forming a small knowledge
graph the selector walks. A Librarian agent populates and expands the cards at build time using web search and the
PDFs in `kb/literature/`. A narrow on-demand search path remains for a diagnosis no card matches.
## Consequences
Cheap retrieval (~200 tokens per card), checkable preconditions, and provenance for the Innovation criterion.
