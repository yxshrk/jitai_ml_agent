# ADR-0013 — The knowledge base evolves: full journal in context, Archivist, Librarian, foundations
Status: Accepted (2026-08-30), requested by Yash

## Context
After four runs the cards carried measurements (ADR-0004 / `distill.py`) but the menu itself never grew: the
Explorer's wildcards — including the one that won live_04's first generation — were journaled and forgotten, so a
later run could neither select, retest nor compose them. The Selector saw one 200-character line per node and the
Implementer eight such lines; the diffs, curves, seed numbers and critic notes lived only on disk. Yash's guidance:
GPT-5.6's context is large — do not summarise away the details; make the KB evolve from the Selector/Explorer's
work; consider web search for the literature; give the roles the mathematics they must reason from.

## Decision
1. **The exact journal is in every call.** `Journal.digest()` renders every node — hypothesis, expected vs realised
   Δ, change summary, critic rounds, metrics, learning curve, seed confirmation, failure and recovery, **and the
   diff** — plus each generation's diagnosis and plan. It is frozen at the start of a generation and placed right
   after the stable prefix as a generation-stable block, so the provider's prompt cache serves it to all ~20 calls
   of the generation after the first (≈ 23 K tokens for 14 nodes; ≈ 80 K at 50). Nothing is summarised; the
   one-line index stays as a table of contents.
2. **Archivist role.** When a run ends (`auto_distill`, or `cli distill`), every measured wildcard — and any
   Selector candidate that named a card that does not exist — is turned into a card written from the node's actual
   diff and numbers, with the run journal in context for honest attribution. Code forces `status`, `evidence` and
   the `## Measured` line; the validator rejects malformed cards; a wildcard recognised as an existing card's
   mechanism is filed as a measurement of that card. The next run's Selector sees the new card with its status.
3. **Librarian role with web search.** Through the OpenAI Responses `web_search` tool the Librarian proposes n new
   cards from published sources (papers, competition write-ups, recommender libraries), checked against the
   foundations (a user-constant term cannot move the metric) and the constraints (numpy, one CPU, 30 min, no test
   access, no pretrained weights). It runs after a flat generation when fewer than k untried cards remain (at most
   twice per run) and on demand (`cli librarian`). Cards enter as `untried` with their source URL; measurement,
   not citation, decides their fate.
4. **Foundations in the prefix.** `kb/spec/foundations.md`: what the metric measures, its invariances (what cannot
   help), loss-versus-metric, noise and the winner's curse, the learning dynamics of this data — derivations, not
   textbook material, because the model already has the textbook.
5. Card status wording: `alive` → `proven — accepted on [stack]`, with the legend stating that "proven in some run"
   does not mean "in the current champion"; the champion's actual stack is stated in every call's context
   (ADR-0012).

## Consequences
The menu grows from three sources (hand-written, archived wildcards, web-searched) and every card carries its
measurements across runs; the roles reason from the complete record rather than from summaries; a run costs
roughly one extra cached prefix per generation (a few cents) and a Librarian call costs ~$0.3–0.5 with 2–6 web
searches. Risk: web content is untrusted — the Librarian only writes cards; every idea still passes the firewall,
the Critic and the referee before it can matter.
