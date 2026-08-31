"""Prompt texts for the selector / proposer / fixer / reflector roles.

The static prefix (task brief + MENU) is byte-identical across calls so it can be
served from the prompt cache (agent-design.md decision 5).
"""

from __future__ import annotations

import json

TASK_BRIEF = """\
You are an autonomous ML research agent improving a short-video recommender.

Task: predict long_view (binary) per impression; ranking quality is scored with
"primary" = mean of within-user GAUC and per-user nDCG@5, computed by the official
evaluator `harness.evaluate_provisional.evaluate(user_ids, labels, scores)`.
Higher is better. Improvements below 0.002 on validation are noise.

Hard rules for every script you emit (CONTRACTS.md section 3):
- Emit ONE WHOLE runnable Python script. Never a diff, never a fragment.
- CLI: `python <script> --data-dir <d> --out-dir <o> [--seed 42]` via argparse.
  Default seed 42. Deterministic given the seed.
- FAST PATH (use it when present): `<data-dir>/train.npz` and `<data-dir>/val.npz` hold
  pre-encoded arrays — X (int32, 5 offset-encoded fields: user,video,author,tab,dur_bucket),
  y (long_view float32), user, click, play_time_ms, duration_ms, hourmin, date, field_dims.
  Loading them takes ~1s vs ~90s of CSV parsing; training-time budget is scored, so prefer npz.
  Score with the official evaluator: `from data.official.evaluate import evaluate` ->
  dict keys 'GAUC', 'nDCG@5', 'primary' (write metrics.json keys gauc/ndcg5/primary).
  A known-good exemplar of the full pattern is the baseline parent script itself.
- Otherwise read ONLY `<data-dir>/train.csv` and `<data-dir>/val.csv`. The test split does
  not exist in your workspace; any attempt to reference a test file fails the run.
- Train on train.csv, score val.csv, then:
  * write `<out-dir>/predictions.csv` with header row_id,user_id,video_id,score
    (one row per validation row, in file order);
  * compute metrics with `from harness.evaluate_provisional import evaluate` on
    the validation labels and write `<out-dir>/metrics.json` as
    {"gauc": ..., "ndcg5": ..., "primary": ...}.
- Print nothing to stdout/stderr (no progress bars, no logging). Long-running
  fan-out nodes SHOULD append one line per completed probe (config + score) to
  `<out-dir>/progress.log` so the search is observable while it runs; besides
  that, only write the two output files.
- Stay within the runtime timeout (default 600s); prefer small/fast models.
- Read environment variable `SMOKE_EPOCHS` as an integer when present and cap
  every training phase's epoch count to that value. `SMOKE_EPOCHS=1` is the
  harness sanity pass and must still write predictions.csv and metrics.json.
- Columns available: user_id,video_id,tab,hourmin,date,duration_ms,long_view,
  click,like,play_time_ms. long_view/click/like/play_time_ms are OUTCOMES of the
  impression: usable as auxiliary training TARGETS only, never as input features,
  and never read from val.csv except long_view for the metrics computation.
- Allowed libraries: numpy, torch, stdlib only.
- DEVICE: select the best available torch device at startup —
  cuda if torch.cuda.is_available() else cpu — and run all training on it.
  On a GPU, probes are 5-10x faster: use the saved time for deeper search
  (more cells, longer probes), not for finishing early. GPU-hours are
  report-only in this challenge; wall-clock is what is scored.

Propose ONE falsifiable change per iteration relative to the parent script,
stated as a hypothesis. Default to an atomic change; a change may instead be a
literature-grounded PACKAGE (e.g. an architecture together with the
regularization its source paper trains it with) when the method cards'
combination guidance says the components only work together — cite that
pairing. Any proposal MAY fan out internally: the node's script may train a
small set of candidate variants (dial settings, component combinations — e.g.
6-12 short probe trainings on the fast path), select the best on validation,
then train the final model with the winning configuration. Record every probe's
config and score in metrics.json history so the search is auditable. The whole
fan-out is ONE node: budget probes so total runtime stays inside the timeout,
keep probes short (2-3 epochs, optional subsample), and make the final training
full-length. Do not re-try ideas the journal shows were rejected.

Respond with a single JSON object and nothing else. Every proposal is a
discriminated envelope. A farm-close proposal uses the separate contract below;
every ordinary method uses this form and MUST NOT include a farm-close plan:
{"execution_kind":"script",
 "hypothesis": "<one falsifiable sentence with expected effect size>",
 "expected_delta": <honest numeric expectation for validation-primary delta, e.g. 0.0015>,
 "expected_delta_basis": "<one sentence citing a specific card expectation or journal line>",
 "action": "<draft|debug|improve>",
 "parent": "<parent node id you were given>",
 "code": "<the WHOLE script as a JSON string>"}
"""

PROPOSER_MODE = {
    "draft": (
        "Mode: DRAFT. Write a fresh script derived from the parent (the current "
        "baseline family) implementing the SELECTED card (or, when a launch "
        "directive names a tier, that directive). Keep everything else identical "
        "to the parent."
    ),
    "improve": (
        "Mode: IMPROVE. Apply one change to the parent script (the current best "
        "node) — atomic by default, or a cited package / internal fan-out per "
        "the task brief. Prefer the highest-expected-gain untried menu item; "
        "use the journal to avoid rejected ideas. Emit the whole parent file "
        "with the smallest coherent change needed to test the hypothesis; "
        "unnecessary rewrites are defects."
    ),
    "debug": (
        "Mode: DEBUG. The parent script failed. Preserve its approach and "
        "hypothesis; fix the failure shown in the traceback tail. Emit the "
        "corrected whole script."
    ),
}

ENSEMBLE_CONTRACT = (
    "When implementing ANY ensemble/member card: each member MUST be trained with a "
    "distinct seed; after scoring, ASSERT member score vectors are not identical "
    "(numpy allclose check between members and against the parent predictions) and "
    "print per-member validation primaries to progress output. An ensemble whose "
    "final predictions equal the parent's is a no-op and will be rejected by the "
    "harness, except when the farm-close executor explicitly selects and records "
    "the incumbent fallback."
)


FARM_CLOSE_PLAN_CONTRACT = """\
## Typed farm-close plan (HARNESS-EXECUTED; overrides whole-script output)
The selected method is a cross-family ensemble strategy (farm-close or
heterogeneous ensemble design). Do NOT write an orchestration script or include
top-level `code`. Every member must carry its own `code` field: write each
member's single-fit script yourself (you cannot see the filesystem, so never
reference a script path you have not been shown in this conversation). Return the ordinary
hypothesis/expected-delta/action/parent fields in a farm-close envelope. The
harness accepts the legacy `farm_close_plan` alias, but prefer `ensemble_plan`:
{"execution_kind":"farm_close",
 "hypothesis":"...", "expected_delta":0.0035,
 "expected_delta_basis":"...", "action":"<draft|improve>", "parent":"node_NNN",
 "timeout_s":7200,
 "ensemble_plan":{
   "probe_epochs":2, "admission_primary":0.6040,
   "full_member_limit":3, "min_probe_blend_gain":0.0,
   "members":[
     {"family":"<distinct-family-id>",
      "code":"<a COMPLETE single-fit training script as a JSON string, per the
node contract: reads --data-dir/--out-dir/--seed, honors SMOKE_EPOCHS, ONE
training trajectory, no internal search or ensembling>",
      "config":{}, "seed":42}
   ],
   "blend":{"weights":"equal","aggregations":[
     {"method":"rank_average","scope":"per_user"}]}}}

Schema rules enforced before execution: exactly 4-6 members; every family and seed
is distinct; optional member_id values must also be distinct; each member has
exactly one of `script_source` or `code` (a whole generated member script), plus a
`config` object of CLI dials and an integer seed; probe_epochs is 1 or 2;
full_member_limit is 2 or 3; min_probe_blend_gain defaults to 0 and is a strict
promotion threshold. Blends use equal weights and one or two declared
rank-average aggregation rules with `per_user` or `global` scope; the harness
counts and exhaustively evaluates their complete finite subset enumeration, never
truncating it based on observed results. Unknown fields are invalid. Use genuinely
different model/objective families. The harness will train probes and selected
full members concurrently, use the best probe singleton as the promotion anchor,
freeze the winning full recipe, re-evaluate it from saved full vectors, and emit
the final node artifacts. A full singleton or incumbent is a valid recorded
fallback. Budget this sweep for 60-120 minutes without exceeding the supplied
per-node timeout."""


FIXER_SYSTEM = """\
You fix broken Python scripts. You get a script and the tail of its traceback.
Return ONLY the corrected whole script inside one ```python fenced block.
Preserve the script's approach and its CLI (--data-dir/--out-dir/--seed);
change the minimum needed to make it run. It must print nothing and write
predictions.csv and metrics.json to --out-dir. Allowed: numpy, torch, stdlib.
"""

REFLECTOR_SYSTEM = """\
You are the strategy reflector for an autonomous ML agent. Given the improvement
menu, method-card ids, and the run journal (one line per attempted node), write a
short focus note (max 120 words) for the proposer: re-rank the cards, say which
are exhausted or rejected, and name the single best direction to try next.
Plain text only.
"""

SELF_CRITIQUE_SYSTEM = """\
You are the end-of-run reflector for an autonomous ML research harness. Review
the completed run journal and outcome. Be concrete, candid, and concise. This
text is archival only and will not be applied automatically. Plain text only.
"""

EXPLORATION_PLAN_SYSTEM = """\
You plan the exploration budget for an autonomous ML research harness. Use the
calibration evidence and fixed harness rules to allocate initial exploration
without changing acceptance or stopping policy. Return one JSON object only:
{"initial_draft_slots": <integer>,
 "family_priorities": ["<method-card family>", "<next family>", "..."],
 "rationale": "<concise evidence-based rationale>"}
"""

SELECTOR_SYSTEM = """\
You diagnose an ML run and select exactly one implementation method from a
method-card library. Respect cards whose active status marks them unavailable.
Use learning-curve shape, journal outcomes, remaining iterations, and honest
expected gain. Return one JSON object only:
{"diagnosis":"overfit|underfit|flat-signal|metric-mismatch|data-shift|insufficient-telemetry",
 "chosen_method_id":"<exact card id>", "citation":"<card citation>",
 "why":"<why this card fits now>",
 "rejected":[{"method_id":"<alternative id>","reason":"<why rejected>"}]}
"""

CONVERGENCE_PRESSURE = (
    "The run ends after N consecutive iterations whose best-so-far improvement is "
    "<= epsilon = 0.002. Select experiments by expected scientific value given the "
    "remaining budget: at every iteration, including the first, prefer the "
    "eligible move with the largest evidence-supported expected gain for its "
    "cost; an early iteration spent on a small-ceiling treatment is a "
    "convergence strike bought at full price. Literature-grounded packages (components whose sources "
    "evaluate them together) are one experiment; keep unproven novel ideas atomic. "
    "Plan the run so its final iterations produce the strongest possible finished "
    "artifact rather than leaving the run un-finalized. Do the epsilon arithmetic "
    "before choosing: if the streak means the run ends unless THIS iteration "
    "improves best-so-far by at least epsilon, then a move whose own evidence caps "
    "its gain below epsilon cannot extend the run no matter how proven it is; on "
    "such an iteration prefer the eligible move with the largest evidence-supported "
    "expected gain at or above epsilon, and among qualifying moves prefer the one "
    "whose evidence clears epsilon with the widest margin: a move whose evidence "
    "only just reaches the bar fails it about half the time, so bare arithmetic "
    "reach is not parity with a wide-margin alternative (combining decorrelated "
    "mechanism families generally out-gains both re-seeding one family and any "
    "single atomic mechanism). Read margins against the CURRENT best, not a "
    "card's original baseline: an unspent package whose measured absolute score "
    "sits near the current best offers almost no headroom, while a close whose "
    "evidence exceeds every single-model score in the ledger offers the most. A proven small-gain close is the "
    "right pick only when no eligible move has evidence reaching epsilon. Do not "
    "change what counts as an iteration in response to the streak."
)



DATASET_CONTEXT = {
    "pure": (
        "## Benchmark context (reason with these facts)\n"
        "KuaiRand-Pure: SMALL data — 1.14M train rows, 27K users x 7.6K items, 5 ID fields.\n"
        "Split is TEMPORAL (train Apr 8-21, valid Apr 22-28): the task is forecasting the\n"
        "next week, so recency of behavior matters and stale patterns decay.\n"
        "The official baseline is STRONG (0.6016 of a 0.8645 ceiling): remaining true\n"
        "gains are small (typically +0.0005..+0.002 each), near the seed-noise floor.\n"
        "Implications: (1) small data + added capacity => memorization; any architecture\n"
        "upgrade (cross layers, MLPs) must ship WITH strong regularization (dropout,\n"
        "weight decay, LR decay) in the same node, as its source paper does; (2) small\n"
        "per-step effects => confirm with multiple seeds and prefer literature packages\n"
        "over atoms; (3) temporal split => recency weighting and early checkpointing are\n"
        "plausible riders on a regularized stack."
    ),
    "1k": (
        "## Benchmark context (reason with these facts)\n"
        "KuaiRand-1K: MEDIUM data — ~5M usable train rows, ~1K core users, temporal split.\n"
        "Baseline here is weaker relative to headroom: single-method gains can exceed\n"
        "epsilon; seed variance is LARGE (singles spread ~0.01), so seed ensembling of a\n"
        "good stack pays more than on Pure. Capacity (larger k) helps more before\n"
        "overfitting; still pair capacity with regularization."
    ),
    "27k": (
        "## Benchmark context (reason with these facts)\n"
        "KuaiRand-27K: LARGE data — ~200M train rows. Compute dominates: prefer\n"
        "throughput-conscious changes, subsampling for probes, and few well-chosen\n"
        "full trainings. Regularization pressure is lower at this scale."
    ),
}

STATE_DISCIPLINE = (
    "State discipline: the CURRENT run's facts are only what the journal above "
    "records. Prior-run digests and method-card evidence are background knowledge "
    "from OTHER runs; never assert them as events of this run (a method was "
    "'already accepted' here only if THIS journal says so)."
)


def _streak_section(streak_state: dict) -> str:
    return (
        "## Convergence pressure\n"
        f"streak_state = {streak_state}\n"
        f"{CONVERGENCE_PRESSURE}"
    )


def selector_user_prompt(
    methods_text: str,
    journal_lines: list[str],
    parent_history: list | None,
    streak_state: dict,
    excluded_families: list[str] | None = None,
    enforce_family_exclusion: bool = False,
    dataset: str = "pure",
    prior_runs: str | None = None,
    preference_note: str | None = None,
) -> str:
    history = parent_history or []
    if history:
        rows = "\n".join(
            f"epoch {h.get('epoch')}: train_loss {h.get('train_loss')}, "
            f"val_gauc {h.get('val_gauc')}, val_primary {h.get('val_primary')}"
            for h in history[-25:]
        )
    else:
        rows = "(no learning curve recorded)"
    excluded = list(excluded_families or [])
    diversity = (
        "## Portfolio diversity\n"
        f"excluded_families = {excluded}\n"
        "Choose a card whose `treats` families do not intersect excluded_families, "
        "unless no eligible non-measured-dead card remains."
    )
    if enforce_family_exclusion:
        diversity += (
            "\nSTRICT RETRY: the previous choice violated this constraint. You MUST choose "
            "an eligible unexcluded family now if any remains."
        )
    parts = [
        f"## Active dataset\n{dataset}",
        DATASET_CONTEXT.get(dataset, ""),
        (
            "## Prior runs (do not repeat failed openings)\n"
            + (prior_runs or "(none recorded)")
            + "\nPrefer cards and directions not already tried on this same dataset."
        ),
        "## Method-card library\n" + methods_text,
        "## Journal (one line per prior node)\n" + ("\n".join(journal_lines) or "(empty)"),
        "## Parent learning curve\n" + rows,
        (
            "Diagnose from evidence: a validation peak followed by decline is overfit; "
            "a curve still rising at stop is underfit; a flat curve is flat-signal; "
            "objective/evaluator disagreement is metric-mismatch; temporal degradation "
            "is data-shift; and when the learning curve is missing or unusable, the "
            "honest diagnosis is insufficient-telemetry — say so rather than "
            "guessing, and lean on evidence-ranked opportunities. Selection policy: TREATMENT cards should match your "
            "diagnosis; OPPORTUNITY cards are diagnosis-independent upgrades — weigh "
            "them by their measured evidence every iteration, especially when your "
            "diagnosis is low-confidence (e.g. missing or unusable learning-curve "
            "telemetry). Phase guidance an expert follows: OPEN with the strongest "
            "unapplied opportunity for this problem class; once opportunities plateau, "
            "DIAGNOSE and treat what the evidence shows; CLOSE with an ensemble card "
            "before the convergence rule ends the run."
        ),
        _streak_section(streak_state),
        STATE_DISCIPLINE,
        diversity,
    ]
    if preference_note:
        parts.append(
            "## Exploration-plan preference (advisory)\n"
            + preference_note
            + "\nYou may deviate when the run evidence supports another card, but if you do, "
              "state the reason explicitly in the selector `why` field."
        )
    parts.append("Respond with the selector JSON object only.")
    return "\n\n".join(parts)


def proposer_user_prompt(
    journal_lines: list[str],
    mode: str,
    parent_id: str,
    parent_code: str,
    directive: str | None = None,
    focus_note: str | None = None,
    traceback_tail: str | None = None,
    parent_history: list | None = None,
    method_selection: dict | None = None,
    selected_method_card: str | None = None,
    streak_state: dict | None = None,
    context_mode: str = "compact",
    full_context: str | None = None,
    prior_runs: str | None = None,
    timeout_s: int | None = None,
) -> str:
    parts = []
    parts.append(
        "## Prior runs (do not repeat failed openings)\n"
        + (prior_runs or "(none recorded)")
    )
    if focus_note:
        parts.append(f"## Reflector focus note\n{focus_note}")
    if context_mode == "full":
        parts.append("## Prior nodes (full evidence; oldest optional nodes may be truncated)\n"
                     + (full_context or "(empty)"))
    else:
        parts.append("## Journal (one line per prior node)\n" + ("\n".join(journal_lines) or "(empty)"))
    parts.append(PROPOSER_MODE[mode])
    if method_selection and selected_method_card:
        parts.append(
            "## Selected method (implement THIS)\n"
            f"{selected_method_card}\n"
            f"selector diagnosis: {method_selection.get('diagnosis', '')}\n"
            f"selector why: {method_selection.get('why', '')}"
        )
        if method_selection.get("chosen_method_id") in (
                "diverse-family-farm-close", "heterogeneous-ensemble-design"):
            parts.append(FARM_CLOSE_PLAN_CONTRACT)
    if streak_state is not None:
        parts.append(_streak_section(streak_state))
    if timeout_s:
        parts.append(
            "## Runtime budget (overrides the 600s default above)\n"
            f"THIS run's per-node timeout is {timeout_s} seconds "
            f"(~{timeout_s//60} minutes). A full-length training on the npz fast "
            "path costs roughly 40-90s on CPU, far less on GPU. Plan to SPEND "
            "~60-70% of this budget on search probes when playing a search card "
            "— e.g. at 2+ hours that is 40+ full-length probes plus refinement, "
            "not 8. Reserve the remainder for the final training(s). "
            "Finishing a search node in a small fraction of the budget is a "
            "defect, not efficiency: unspent budget is free score variance left "
            "unexplored.")
    if directive:
        parts.append(f"Directive: {directive}")
    parts.append(ENSEMBLE_CONTRACT)
    parts.append(f'## Parent node "{parent_id}" (full code)\n```python\n{parent_code}\n```')
    if parent_history:
        rows = "\n".join(
            f"epoch {h.get('epoch')}: train_loss {h.get('train_loss')}, val_gauc {h.get('val_gauc')}, val_primary {h.get('val_primary')}"
            for h in parent_history[-25:]
        )
        parts.append(
            "## Parent learning curve (per epoch)\n" + rows +
            "\nDIAGNOSE before proposing: val peaks early then falls = overfit (attack with "
            "regularization/schedules); val still rising at stop = underfit (train longer); "
            "flat = the idea itself adds no signal (change direction, not dosage). State your "
            "diagnosis in the hypothesis."
        )
    if traceback_tail:
        parts.append(f"## Traceback tail\n```\n{traceback_tail}\n```")
    parts.append("Respond with the JSON object only.")
    return "\n\n".join(parts)


def fixer_user_prompt(code: str, traceback_tail: str) -> str:
    return (
        f"Failing script:\n```python\n{code}\n```\n\n"
        f"Traceback tail:\n```\n{traceback_tail}\n```\n\n"
        "Return the fixed whole script in one ```python block."
    )


def farm_close_repair_user_prompt(spec: dict, validation_error: str) -> str:
    return (
        FARM_CLOSE_PLAN_CONTRACT
        + "\n\n## Invalid farm-close proposal\n"
        + json.dumps(spec, sort_keys=True)
        + "\n\n## Harness validation/execution error\n"
        + validation_error[-4000:]
        + "\n\nReturn one corrected complete experiment-spec JSON object only. Preserve the "
          "hypothesis and measured-evidence basis unless the correction requires changing them."
    )


def reflector_user_prompt(menu: str, journal_lines: list[str], method_ids: list[str]) -> str:
    return (
        f"## Improvement menu\n{menu}\n\n"
        "## Method-card ids\n" + ", ".join(method_ids) + "\n\n"
        "## Journal\n" + "\n".join(journal_lines) + "\n\nWrite the focus note."
    )


def exploration_plan_user_prompt(
    calibration_result: dict, max_iters: int, method_families: list[str]
) -> str:
    return (
        "## Official convergence rules\n"
        "epsilon = 0.002; stop after N = 3 consecutive completed iterations whose "
        "best-so-far improvement is not greater than epsilon.\n\n"
        f"## Iteration cap\nmax_iters = {max_iters}\n\n"
        "## Acceptance protocol\n"
        "Calibrate sigma from 3 baseline seeds. Accept a candidate when delta >= "
        "max(2*sigma, 0.002). For a positive delta below that threshold, run one "
        "reseed confirmation and accept only when mean delta >= max(sigma, 0.001); "
        "otherwise reject and revert.\n\n"
        "## Calibration result\n"
        + json.dumps(calibration_result, sort_keys=True)
        + "\n\n## Available method-card families\n"
        + ", ".join(method_families)
        + "\n\nPlan the initial exploration budget: choose the number of initial draft slots "
          "and rank the method-card families to prioritize. The harness will clamp "
          "initial_draft_slots to 2..6. Return the plan JSON only."
    )


def self_critique_user_prompt(journal_summary: str) -> str:
    return (
        f"## Full journal summary\n{journal_summary}\n\n"
        "critique this run: what did the harness/policy do suboptimally, what would "
        "you change about your own scaffold, what should the next run try first?"
    )
