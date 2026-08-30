# Method cards — the agent's menu

One file per technique. The Selector filters cards on `applies_when` against `kb/data/facts.md`, ranks by
`expected_delta` per `cost`, and must name one rejected card; the Implementer follows `how_to`; the Consolidator
reads `composes_with` / `conflicts_with` when planning merges. `status` and `evidence` are updated from run journals
(ADR-0004: a card is dead only under the conditions it was measured).

Every card has this front matter (all fields required):

| field | meaning |
|---|---|
| `id` | file name without `.md`; `<component>-<method>` |
| `family` | coarse group: ranking-loss, watch-time, features, data-weighting, aux-targets, history, model, regularization, ensembling |
| `target_component` | exactly one of features, encoding, model, loss, training-schedule, regularization, aux-targets, history, data-weighting, ensembling |
| `source` | paper(s) in `kb/literature/` with section, or the organizer README line |
| `applies_when` | preconditions, each checkable against a numbered fact in `facts.md` |
| `expected_delta` | `[lo, hi]` on validation primary, honest against the 0.002 floor and the +0.023 that all of personalisation is worth |
| `cost` | lines changed / runtime multiplier / dependencies (any library the contract lists: numpy, pandas, scikit-learn, LightGBM, torch on CPU — ADR-0014) |
| `composes_with` / `conflicts_with` | card ids |
| `status` | `untried` · `proven — accepted on [stack]` · `dead_under [stack ×N (best Δ)]` — aggregated over every stack the card was measured on, recomputed by `distill.py` |
| `evidence` | journal references once measured |

Body sections: **Claim**, **Mechanism (why it moves within-user ranking)**, **How to implement on node_000**,
**Risks / failure modes**, **Measured** (appended by runs).

Cards are loaded verbatim into the stable prompt prefix, so keep each under ~60 lines (the validator caps at 70).

Three sources write cards (ADR-0013): hand-written from the literature; the **Archivist**, which turns every measured
wildcard of a run into a card written from its actual diff (`cli distill`); and the **Librarian**, which drafts
`untried` cards from web-searched sources (`cli librarian`, or in-loop after a flat generation). All three pass
`validate.py`; only measurement changes a status.
