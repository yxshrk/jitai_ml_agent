# ADR-0016 — Family campaigns: one card family per generation, chosen in code

**Status:** accepted, 2026-08-31 (built during live_07 on an isolated branch; merged after the run with the review
session's ADR-0015 screen gate).
**Extends:** ADR-0009 (generational branching), ADR-0014 (slot rules in code: free slot, closed mechanisms, hard groups).

## Context

From generation 2 the harness has run *breadth* generations: k candidates, each on a different pipeline component,
each usually a different family of idea. live_06 and live_07 show the cost: a family gets one shot per generation, a
near-miss is deepened by dose rather than by mechanism, and a family whose second mechanism might have worked never
gets it because the next generation's slots go elsewhere. Yash's standing preference is depth over breadth; the
Selector prompt already says so, but prose is not a rule.

The cards carry `family:` (ranking-loss, watch-time, features, data-weighting, aux-targets, history, model,
regularization, ensembling, …). That is the natural unit of a campaign: a family shares inputs and mechanism space,
so k variants of one family in one generation are the cheapest way to learn whether the family has anything left.

## Decision

1. **From `CAMPAIGNS_FROM_GENERATION` (2) every generation has a campaign family**, chosen by code
   (`Loop._campaign_family`): the current family while it is open, else the best-scoring open family. Score = a
   measured screen gain of one of its cards if any (`state['screened']`, ADR-0015) — measured beats promised — else
   the highest card `expected_delta` among the family's cards still measurable on the current stack; families in
   `CAMPAIGN_LAST_FAMILIES` (ensembling) come last, because composition needs members worth blending. A family with
   nothing left to measure on this stack is marked `exhausted`. Generation 1 stays the breadth generation.
2. **Every Selector candidate belongs to the campaign family**, except the Consolidator's merge/retest slots and the
   Explorer's wildcard; `_apply_rules` drops a known card from another family. The free slot (ADR-0014) is kept inside
   the family: its untried or not-yet-stacked cards first.
3. **Diversity inside a campaign is by mechanism, not by component** (`_diversify`): the family's candidates share a
   component by design, so each must carry a distinct `mechanism` slug; a second candidate with the same slug is
   dropped. Outside a campaign the component rule stands, and the free-slot-vs-wildcard rule of ADR-0014 is unchanged.
4. **A family closes after `CAMPAIGN_FLAT_GENERATIONS` (2) campaign generations without an accepted node from it**
   (`_campaign_update`, run at the generation close); the record keeps its generations, nodes, best seed-mean gain
   and the reason. Merges are booked to no family. When every family is closed the run behaves as before (breadth),
   and convergence (ADR-0012) is untouched throughout.
5. The state block every planning role reads carries the campaign and every family's status; the Selector's prompt
   states the rule once; the Diagnostician adds one clause on whether the campaign should stay open. The generation
   journal record and the run summary carry `campaign` and `families`. `--no-campaigns` restores the old behaviour.

## Consequences

- Depth is now enforced: a family gets k different mechanisms in one generation and a second generation if any of
  them was accepted, then the search moves on with a recorded verdict on the whole family — the write-up can say
  "history: closed after 2 generations, best +0.0009" instead of listing scattered nodes.
- The order is decided by evidence the run already has (screen gains, card ranges, what is dead on this stack), so
  the same KB gives the same campaign order — reproducible planning.
- A risk: a family that needs a *different* champion to work (e.g. session features helped BPR but not the seed
  blend) is closed on this stack; the Consolidator's retest slot remains the way back, with a stated reason.
- Cost: none in LLM calls; a few extra lines of state per generation.
