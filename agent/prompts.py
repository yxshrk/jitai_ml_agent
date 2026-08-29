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
- Print nothing to stdout/stderr (no progress bars, no logging). Only write the
  two output files.
- Stay within the runtime timeout (default 600s); prefer small/fast models.
- Read environment variable `SMOKE_EPOCHS` as an integer when present and cap
  every training phase's epoch count to that value. `SMOKE_EPOCHS=1` is the
  harness sanity pass and must still write predictions.csv and metrics.json.
- Columns available: user_id,video_id,tab,hourmin,date,duration_ms,long_view,
  click,like,play_time_ms. long_view/click/like/play_time_ms are OUTCOMES of the
  impression: usable as auxiliary training TARGETS only, never as input features,
  and never read from val.csv except long_view for the metrics computation.
- Allowed libraries: numpy, torch, stdlib only.

Propose exactly ONE atomic, falsifiable change per iteration relative to the
parent script, and state it as a hypothesis. Do not re-try ideas the journal
shows were rejected.

Respond with a single JSON object and nothing else:
{"hypothesis": "<one falsifiable sentence with expected effect size>",
 "expected_delta": <honest numeric expectation for validation-primary delta, e.g. 0.0015>,
 "expected_delta_basis": "<one sentence citing a specific card expectation or journal line>",
 "action": "<draft|debug|improve>",
 "parent": "<parent node id you were given>",
 "code": "<the WHOLE script as a JSON string>"}
"""

PROPOSER_MODE = {
    "draft": (
        "Mode: DRAFT. Write a fresh script derived from the parent (the current "
        "baseline family) implementing one idea from the menu tier named in the "
        "directive. Keep everything else identical to the parent."
    ),
    "improve": (
        "Mode: IMPROVE. Apply exactly one atomic change to the parent script "
        "(the current best node). Prefer the highest-expected-gain untried menu "
        "item; use the journal to avoid rejected ideas. Emit the whole parent "
        "file with the smallest coherent change needed to test the hypothesis; "
        "unnecessary rewrites are defects."
    ),
    "debug": (
        "Mode: DEBUG. The parent script failed. Preserve its approach and "
        "hypothesis; fix the failure shown in the traceback tail. Emit the "
        "corrected whole script."
    ),
}

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
{"diagnosis":"overfit|underfit|flat-signal|metric-mismatch|data-shift",
 "chosen_method_id":"<exact card id>", "citation":"<card citation>",
 "why":"<why this card fits now>",
 "rejected":[{"method_id":"<alternative id>","reason":"<why rejected>"}]}
"""

CONVERGENCE_PRESSURE = (
    "If streak is N-1 of N, this may be the final iteration — prefer the "
    "highest-expected-gain untried card, not a small dose adjustment. When the "
    "current streak is N-1 of N and no untried single-model card remains promising, "
    "the seed-ensemble card is the canonical closing move."
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
            "is data-shift."
        ),
        _streak_section(streak_state),
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
    if streak_state is not None:
        parts.append(_streak_section(streak_state))
    if directive:
        parts.append(f"Directive: {directive}")
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
