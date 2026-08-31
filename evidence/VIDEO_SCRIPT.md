# Video shooting script (~3:00)

Team jit.ai · TechJam 2026 Track 2. Video is optional per the 28 Aug webinar; we
make one anyway because the flight-recorder replay shows autonomy better than any
paragraph. Screen capture the site at http://localhost:8642/index.html plus a few
terminal and file shots. Speak only the narration lines. Roughly 460 spoken words
at a measured pace fills 3:00.

Every number below traces to evidence/RESULTS_AND_RESOURCES.md, logs/RUNS.md, or a
named run directory. Do not ad-lib numbers. Label historical footage "recorded run
replay" on screen. The autonomy claim is "zero mid-run interventions" (the official
definition); never say "zero humans".

---

## Scene 1 · 0:00–0:18 · Cold open: the result

**On screen:** Site hero: the self-drawing trajectory chart, headline settling on
0.605575. Overlay chips: "KuaiRand-Pure · +0.0040 over baseline" and
"6 iterations · 17 min · 115,315 tokens · 0 mid-run interventions".

**Narration:**

> An autonomous agent improved TikTok's KuaiRand-Pure baseline by four thousandths
> of primary score, in seventeen minutes, on a laptop CPU, with zero mid-run human
> interventions. That sentence is easy to say. This video is about how you can
> check it.

## Scene 2 · 0:18–0:55 · The flight recorder

**On screen:** Scroll into the pinned mission-log replay. Let the chart grow
iteration by iteration in sync with the journal cards: baseline reproduction,
the accepted regularized-DCN sweep node, the ensemble-design node, the three
convergence strikes. Hover one card to show the hypothesis, diff, and metrics.
Label: "recorded run replay · run_bigclock_07".

**Narration:**

> Every run writes a flight recorder. This is the designated run replayed from its
> journal: six research decisions. It reproduced the official baseline, ran its own
> two-stage hyperparameter search inside one node, then designed an ensemble,
> trained seven seed members and validation-selected three, and stopped itself
> under the official convergence rule. Each card is a hypothesis, a code diff,
> official metrics, and a verdict. The agent decides; a deterministic harness
> executes, scores, and journals. Nothing in this replay was edited by a human
> while it ran, and the journal is how you would catch us if it had been.

## Scene 3 · 0:55–1:25 · The harness is the product

**On screen:** The loop diagram section, then the harness deep-dive cards. Cut to
a brief code shot of the typed farm-close plan (agent-emitted plan JSON or the
strategy card), then a terminal shot of the phases running: probes, blend map,
full trains, re-verify.

**Narration:**

> The engineering is a strict division of labor. The agent proposes; the harness
> owns seeding, timeouts, acceptance against a seed-calibrated noise floor, and
> stopping. Our newest capability makes even the endgame typed: the agent plans a
> cross-family ensemble as data, and the harness executes it deterministically.
> Cheap probes of diverse model families, a blend map over their predictions,
> full training for only the complementary ones, and a re-verified final blend.
> The agent chooses the research move. It never gets to improvise the accounting.

## Scene 4 · 1:25–2:15 · Testing judgment like code

**On screen:** Terminal: farm_f1's journal tail showing the endgame choice and
early convergence. Then tools/decision_bench files, the frozen fixture, and the
bench run printing 3/3. Then the model-tier bench table. End on a live status
line for farm_f2. Label the failure footage "recorded run replay · run_farm_f1".

**Narration:**

> Here is the part we are proudest of, and it starts with a mistake. Last night a
> live run made a measurably wrong endgame call: it closed with a same-family
> ensemble whose ceiling could never clear the acceptance threshold, and converged
> early. We did not shrug. We froze that exact decision state as a benchmark
> fixture, root-caused the choice to a stale piece of guidance, and fixed it with
> a general principle about epsilon arithmetic, not a hard-coded answer for that
> scenario. The bench now passes three out of three, and the corrected run is back
> in flight. The same benches told us which model tiers have sound judgment: our
> top tier passes all six scenarios at low reasoning effort; the workhorse needs
> medium effort for the hardest one. We test the agent's judgment the way you test
> code: regression suites, frozen fixtures, no fix ships without a passing bench.

## Scene 5 · 2:15–2:45 · Receipts

**On screen:** Site receipts section, then logs/RUNS.md scrolling, then the 1K
audit panel. Overlays: "139 completed disclosed runs · ~250 measured cells ·
~9.9M tokens · ~140 run-hours" and "KuaiRand-1K: 0.66892, triple-audited".

**Narration:**

> The full campaign is disclosed: one hundred thirty-nine completed runs, about
> two hundred fifty measured cells, and every negative result kept on the ledger.
> On the bonus 1K dataset, the agent discovered causal session features worth
> point zero one nine, reaching zero point six six nine. We audited that number
> three ways before believing it: exact independent re-evaluation, fresh-seed
> replication, and a shuffle test. Hidden test is still the decisive judge, and
> we say so.

## Scene 6 · 2:45–3:00 · Close

**On screen:** Final results table with the 27K row visibly labeled
"out-of-protocol scaling demo". End card: repo URL and "Inspect the journal."

**Narration:**

> This is autonomous machine learning research with the evidence needed to
> question it. Read the journals, rerun the benches, rebuild the submission.
> Team jit dot A I.

---

## Shot checklist

- [ ] Site at 1440x790, hero draw captured from a fresh load
- [ ] Mission-log scrolly, slow scroll, one card hover
- [ ] Loop diagram + harness deep-dive cards
- [ ] Farm-close plan (typed plan) + phase terminal output
- [ ] farm_f1 journal tail, decision-bench 3/3 output, model-tier table
- [ ] farm_f2 live status line (or note it completed, with its outcome)
- [ ] RUNS.md scroll + 1K audit panel + receipts section
- [ ] "Recorded run replay" label present on all historical footage
