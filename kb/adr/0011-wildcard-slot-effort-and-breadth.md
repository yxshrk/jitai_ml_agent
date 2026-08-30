# ADR-0011 — A wildcard slot, more branches, higher reasoning effort, lower minimum effect
Status: Accepted (2026-08-30), requested by Yash after live_02

## Context
live_02 converged with every card measured and one real win (BPR) plus one near-miss (a 5-seed ensemble, +0.0009,
t = 5.4, rejected by the 0.001 minimum effect). Yash asked for more candidates per generation, an element of
randomness for finding new solutions, and a stronger model. GPT-5.6 rejects `temperature` (verified), so
randomness cannot come from sampling; the accepted reasoning levels are none / low / medium / high / xhigh / max.

## Decision
- k = 5 branches per generation (node cap 50 → at most 9 generations).
- One slot per generation is a **wildcard** proposed by an Explorer role: not a card as it stands — a combination
  of mechanisms, an un-carded technique from the literature, or an idea grounded in a numbered data fact; same
  contract (one component, < 120 changed lines, honest expected delta). Creativity is obtained by prompt and role,
  not by sampling temperature. Wildcards are flagged in the journal.
- Selector, Implementer and Explorer run at reasoning effort `xhigh` on gpt-5.6-sol.
- Minimum effect for acceptance lowered from 0.001 to 0.0005; the 2.5-standard-error test still guards noise.

## Consequences
~5/3 the tokens and training per generation; a real +0.0009 improvement is now accepted; one branch per generation
is deliberately off-menu, so some will be strange — that is its job.
