# ADR-0021 — A frontier of progressing nodes and a persistent proposal queue

**Status:** accepted 2026-08-31 (Yash's design). **Extends:** ADR-0009 (generational branching), ADR-0004 (parked
ideas), ADR-0014/0016 (slot rules, campaigns). **Unchanged:** ADR-0012 — promotion to champion still requires a
seed-confirmed gain, and only accepted nodes can be designated for submission.

## Context

Until now the search was one node wide. `pick_champion` chose the best **accepted** node and every generation branched
from it; a node that improved but missed the acceptance bar was *parked* — available for a retest, never built upon.
Three runs in a row, the highest-mean node of the run was exactly such a node, and the search walked away from it:

| run | node | fresh-seed mean | champion's mean | fate |
|---|---|---|---|---|
| live_07 | node_019 | 0.60486 | 0.60437 | rejected at z 1.9, never expanded |
| live_08 | node_012 | 0.60465 | 0.60429 | rejected at z 1.5, never expanded |
| live_09 | node_017 | 0.60469 | 0.60458 | rejected at z 0.4, never expanded |

The second waste was the generation boundary: the Selector proposed k candidates, the diversity, campaign and screen
rules dropped some, and everything not run was discarded with the generation — a good idea competing with a better
one in the same minute was simply lost.

## Decision

1. **Frontier.** `state['frontier']` holds every node worth building on: the champion, every accepted node, and every
   node whose fresh-seed mean is within `FRONTIER_MARGIN_SE` (1.0) standard errors of the champion's — accepted or
   not. It is capped at `FRONTIER_MAX` (6) by mean, and the champion is always a member.
2. **Retirement.** A frontier node that goes `FRONTIER_RETIRE_GENERATIONS` (2) generations without an **accepted**
   child retires, and its pending proposals go with it (`_frontier_book` counts children per parent each generation,
   `frontier_update` retires). The champion never retires. This is the same shape as closed mechanisms (ADR-0014) and
   closed families (ADR-0016): explore freely, close on evidence.
3. **Queue.** Planner proposals are not executed directly. `queue_add` resolves each proposal's parent, drops a
   duplicate of a pending (parent, mechanism), scores it — how far its parent has come (`mean − baseline`) plus what
   the idea is worth on the record (`prompts.card_value`, ADR-0018) plus a small wildcard bonus — and keeps it.
   `queue_pop` takes the best eligible proposals for this generation: parent still on the frontier, mechanism not
   closed, inside the current campaign family, not stale (`QUEUE_STALE_GENERATIONS` = 3). **Ineligible-but-live
   proposals wait** (an out-of-campaign idea runs when its family's campaign comes round); stale ones, orphaned ones
   and closed mechanisms are dropped with a journal line.
4. **The planners see both.** The state block every planning role reads lists the frontier (node, mean, accepted,
   children, barren generations, champion marker) and the queue's best pending items, and the Selector is told to set
   `parent` to whichever frontier node its candidate should edit and not to repeat what is already queued.
5. **`--no-frontier`** restores the champion-only behaviour, so the earlier runs stay reproducible.

## Consequences

- The search can now pursue a lineage the acceptance test could not confirm — which is where the last three runs'
  best numbers were — without any weakening of what may be *submitted*: designation is still accepted-only (ADR-0012
  amendment), so a lucky node can be explored but never handed in on its own.
- Slots are allocated by evidence rather than by seniority: no reserved majority for the champion. If the frontier's
  best ideas all sit on one node, they all run; if they spread, the generation spreads.
- The queue makes the Selector's runner-up ideas real work items instead of transcript. Expect fewer LLM calls to be
  wasted re-deriving an idea that was already proposed.
- The risk is spending slots on noise: most unconfirmed +0.0003s are noise. The retire rule, the frontier margin of
  one standard error, and the mean-based queue score are the guards; if a run shows the frontier feeding on flukes,
  `FRONTIER_MARGIN_SE` and `FRONTIER_RETIRE_GENERATIONS` are the two dials, and `--no-frontier` is the exit.
