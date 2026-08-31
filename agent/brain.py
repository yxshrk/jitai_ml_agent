"""Provider-agnostic LLM client: selector, proposer, fixer, reflector.

Primary backend: OpenAI Responses API (hand-rolled urllib, no SDK needed).
Secondary backend: Anthropic Messages (SDK), kept for A/B via --provider anthropic.
Role -> model mapping lives in agent/models.toml, never in code.

API keys come from the environment or .env (hand-rolled parser). The static
prefix (task brief + MENU) is byte-identical across calls: on OpenAI that hits
automatic prefix caching; on Anthropic it carries an ephemeral cache_control
block. Token usage is metered per provider+model+role and cumulatively; the
judged number (and the --max-tokens stop) is total in+out across providers.
"""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from agent import prompts

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = Path(__file__).resolve().parent / "models.toml"
METHODS_PATH = Path(__file__).resolve().parent / "METHODS.md"
CLEAN_METHODS_PATH = Path(__file__).resolve().parent / "METHODS_CLEAN.md"


def load_model_config(path: Path = MODELS_PATH) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_api_key(name: str, env_path: Path | None = None) -> str:
    """Key from the environment, else from .env."""
    key = os.environ.get(name)
    if key:
        return key
    path = env_path or ROOT / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in environment or .env")


@dataclass
class TokenMeter:
    per_role: dict[str, dict[str, int]] = field(default_factory=dict)
    last_in: int = 0
    last_out: int = 0

    def add(self, key: str, tokens_in: int, tokens_out: int) -> None:
        bucket = self.per_role.setdefault(key, {"in": 0, "out": 0, "calls": 0})
        bucket["in"] += tokens_in
        bucket["out"] += tokens_out
        bucket["calls"] += 1
        self.last_in, self.last_out = tokens_in, tokens_out

    @property
    def total(self) -> int:
        return sum(b["in"] + b["out"] for b in self.per_role.values())


PLAN_ENVELOPE_KEYS = ("farm_close_plan", "ensemble_plan")


def normalize_proposal_envelope(spec: dict) -> dict:
    """Validate the code-or-plan proposal union and return its canonical form."""
    normalized = dict(spec)
    execution_kind = normalized.get("execution_kind")
    plan_keys = [key for key in PLAN_ENVELOPE_KEYS if key in normalized]
    has_code = "code" in normalized

    if execution_kind is None:
        if has_code and not plan_keys:
            normalized["execution_kind"] = "script"
            execution_kind = "script"
        else:
            raise ValueError(
                "proposal missing execution_kind; only legacy code-only proposals "
                "normalize to script"
            )
    if execution_kind not in ("script", "farm_close"):
        raise ValueError("execution_kind must be 'script' or 'farm_close'")

    if execution_kind == "script":
        if not has_code or not isinstance(normalized["code"], str):
            raise ValueError("script proposal must carry code")
        if plan_keys:
            raise ValueError("script proposal must not carry a farm-close plan")
        return normalized

    if has_code:
        raise ValueError("farm_close proposal must not carry code")
    if len(plan_keys) != 1:
        raise ValueError(
            "farm_close proposal must carry exactly one of farm_close_plan or ensemble_plan"
        )
    plan = normalized.pop(plan_keys[0])
    normalized["farm_close_plan"] = plan
    return normalized


def extract_json_spec(text: str) -> dict:
    """Parse the proposer reply into an experiment spec, with fallbacks."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))
    start = text.find("{")
    if start != -1:
        candidates.append(text[start : text.rfind("}") + 1])
    for cand in candidates:
        try:
            spec = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(spec, dict):
            return normalize_proposal_envelope(spec)
    # Last resort: pull hypothesis + a python fence out of free-form text.
    code = extract_code_block(text)
    hyp = re.search(r'"hypothesis"\s*:\s*"([^"]+)"', text)
    if code and hyp:
        return normalize_proposal_envelope({"hypothesis": hyp.group(1), "code": code})
    raise ValueError(f"could not parse proposer reply ({text[:200]!r}...)")


def extract_json_object(text: str) -> dict:
    """Parse a JSON object from a role that does not emit experiment code."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError(f"could not parse JSON object ({text[:200]!r}...)")


def parse_method_cards(methods_text: str) -> dict[str, str]:
    """Return exact card markdown keyed by its stable id."""
    matches = list(re.finditer(r"^### ([a-z0-9-]+): .+$", methods_text, re.MULTILINE))
    cards = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(methods_text)
        cards[match.group(1)] = methods_text[match.start():end].strip()
    return cards


def parse_method_card_metadata(card_text: str, dataset: str = "pure") -> dict:
    """Parse harness-owned routing metadata from one method card."""
    if dataset not in ("pure", "1k"):
        raise ValueError("dataset must be 'pure' or '1k'")
    treats_match = re.search(r"^- treats:\s*(.+)$", card_text, re.MULTILINE)
    kind_match = re.search(r"^- kind:\s*(treatment|opportunity)\s*$", card_text, re.MULTILINE)
    reference_match = re.search(
        r"^- reference_primary:\s*(none|[-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
        card_text,
        re.MULTILINE | re.IGNORECASE,
    )
    gain_match = re.search(r"^- expected_gain / cost:\s*(.+)$", card_text, re.MULTILINE)
    status_match = re.search(
        rf"^- status_{re.escape(dataset)}:\s*(.+)$", card_text, re.MULTILINE
    )
    if status_match is None:
        status_match = re.search(r"^- status:\s*(.+)$", card_text, re.MULTILINE)
    gains = []
    if gain_match:
        gains = [
            float(value)
            for value in re.findall(r"(?<!\d)(?:\+)?(0\.\d+)(?!\d)", gain_match.group(1))
            if float(value) <= 0.1
        ]
    reference = None
    if reference_match and reference_match.group(1).lower() != "none":
        reference = float(reference_match.group(1))
    return {
        "treats": [part.strip() for part in treats_match.group(1).split("|")] if treats_match else [],
        "kind": kind_match.group(1) if kind_match else "treatment",
        "reference_primary": reference,
        "expected_gain": max(gains, default=0.0),
        "measured_dead": bool(status_match and status_match.group(1).startswith("measured-dead")),
        "status": status_match.group(1) if status_match else None,
    }


def method_cards_for_dataset(methods_text: str, dataset: str) -> str:
    """Expose only the active dataset status to the selector."""
    if dataset not in ("pure", "1k"):
        raise ValueError("dataset must be 'pure' or '1k'")
    active = f"status_{dataset}"
    lines = []
    for line in methods_text.splitlines():
        match = re.match(r"^- (status_(?:pure|1k)):\s*(.*)$", line)
        if not match:
            lines.append(line)
        elif match.group(1) == active:
            lines.append(f"- status: {match.group(2)}")
    return "\n".join(lines)


def two_tier_methods_text(methods_text: str) -> str:
    """Render the card library as two sections: diagnosis-matched treatments and
    evidence-ranked, diagnosis-independent opportunities (v2 selection design)."""
    matches = list(re.finditer(r"^### [a-z0-9-]+: .+$", methods_text, re.MULTILINE))
    if not matches:
        return methods_text
    preamble = methods_text[: matches[0].start()].rstrip()
    cards = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(methods_text)
        cards.append(methods_text[match.start():end].strip())
    treatments, opportunities = [], []
    for card in cards:
        meta = parse_method_card_metadata(card)
        (opportunities if meta["kind"] == "opportunity" else treatments).append((card, meta))
    def rank(item):
        card, meta = item
        import re as _re
        status = (meta.get("status") or "").strip()
        win = status.startswith("measured-win")
        ref = meta["reference_primary"]
        if ref is None:  # annotated reference lines (e.g. "0.6047 single / ...") still rank
            m = _re.search(r"^- reference_primary:[^\n]*?([0-9]+\.[0-9]+)", card, _re.MULTILINE)
            ref = float(m.group(1)) if m else 0.0
        return (0 if win else 1, -(ref or 0.0))
    opportunities.sort(key=rank)
    parts = [preamble,
             "\n## TREATMENT CARDS — match these to your diagnosis\n",
             "\n\n".join(card for card, _ in treatments),
             "\n## OPPORTUNITY CARDS — diagnosis-independent; ranked by measured evidence; "
             "always in your consideration set\n",
             "\n\n".join(card for card, _ in opportunities)]
    return "\n".join(parts)


def extract_code_block(text: str) -> str | None:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return fence.group(1) if fence else None


class _OpenAIBackend:
    """OpenAI Responses API over urllib. gpt-5.x are reasoning models: the
    output array can contain reasoning items before the message; we take the
    message item's text and budget max_output_tokens generously."""

    URL = "https://api.openai.com/v1/responses"

    def __init__(self) -> None:
        self.key = load_api_key("OPENAI_API_KEY")

    def call(self, model: str, system_text: str, user_text: str, max_tokens: int) -> tuple[str, int, int]:
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": os.environ.get("AGENT_REASONING_EFFORT", "medium")},
        }
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.load(response)
        texts = []
        for item in body.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") in ("output_text", "text"):
                        texts.append(part.get("text", ""))
        usage = body.get("usage", {})
        return "".join(texts), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))


class _AnthropicBackend:
    """Anthropic Messages via the SDK, with an ephemeral cache_control block on
    the static system prefix."""

    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=load_api_key("ANTHROPIC_API_KEY"))

    def call(self, model: str, system_text: str, user_text: str, max_tokens: int) -> tuple[str, int, int]:
        kwargs = {}
        if max_tokens >= 8000:
            # Cap internal thinking so long code replies keep text budget.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": "medium"}
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
            **kwargs,
        )
        usage = response.usage
        tokens_in = (
            usage.input_tokens
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            with open("logs/anthropic_debug.log", "a") as fh:
                fh.write(f"{model} stop={response.stop_reason} out={usage.output_tokens} "
                         f"len={len(text)} blocks={[b.type for b in response.content]}\n")
        except OSError:
            pass
        if not text.strip():
            raise RuntimeError(
                f"anthropic returned no text (stop={response.stop_reason}, "
                f"blocks={[b.type for b in response.content]}, out={usage.output_tokens})")
        return text, tokens_in, usage.output_tokens


class Brain:
    """Provider-agnostic brain used by the loop."""

    def __init__(
        self,
        menu_text: str,
        provider: str | None = None,
        model_overrides: dict[str, str] | None = None,
        max_code_tokens: int = int(os.environ.get("AGENT_MAX_CODE_TOKENS", "16000")),  # reasoning models spend thinking tokens from this budget; whole-file replies need headroom (raise when reasoning effort is high)
        budget=None,
        knowledge_mode: str = "full",
    ) -> None:
        from agent.budget import Budget

        self.budget = budget if budget is not None else Budget()
        self.usd_run = 0.0
        config = load_model_config()
        self.provider = provider or config["default_provider"]
        if self.provider not in ("openai", "anthropic"):
            raise ValueError(f"unknown provider: {self.provider}")
        self.models = dict(config[self.provider])
        self.role_max_tokens = dict(config.get("max_output_tokens", {}))
        if model_overrides:
            self.models.update(model_overrides)
        self.backend = _OpenAIBackend() if self.provider == "openai" else _AnthropicBackend()
        if knowledge_mode not in ("full", "clean"):
            raise ValueError("knowledge_mode must be 'full' or 'clean'")
        self.knowledge_mode = knowledge_mode
        self.menu_text = menu_text
        methods_path = CLEAN_METHODS_PATH if knowledge_mode == "clean" else METHODS_PATH
        self.methods_text = methods_path.read_text()
        self.method_cards = parse_method_cards(self.methods_text)
        self.max_code_tokens = max_code_tokens
        self.meter = TokenMeter()
        # Byte-identical static prefix -> prompt cache hits on both providers.
        context_heading = "Task context" if knowledge_mode == "clean" else "Improvement menu"
        self.static_prefix = prompts.TASK_BRIEF + f"\n\n## {context_heading}\n" + menu_text

    def _call(self, role: str, system_text: str, user_text: str, max_tokens: int) -> str:
        model = self.models[role]
        # Hard dollar cap: refuse the call if its worst case could exceed BUDGET_USD.
        # Conservative input estimate: ~1 token per 3 chars, plus overhead margin.
        est_input = (len(system_text) + len(user_text)) // 3 + 500
        worst_output = max_tokens
        self.budget.precheck(model, est_input, worst_output)
        text, tokens_in, tokens_out = self.backend.call(model, system_text, user_text, max_tokens)
        self.meter.add(f"{self.provider}/{model}/{role}", tokens_in, tokens_out)
        self.usd_run += self.budget.record(self.provider, model, tokens_in, tokens_out, note=role)
        return text

    @property
    def usd_total(self) -> float:
        return self.budget.total()

    def propose(
        self,
        journal_lines: list[str],
        mode: str,
        parent_id: str,
        parent_code: str,
        directive: str | None = None,
        focus_note: str | None = None,
        traceback_tail: str | None = None,
        parent_history: list | None = None,
        method_selection: dict | None = None,
        streak_state: dict | None = None,
        context_mode: str = "compact",
        full_context: str | None = None,
        prior_runs: str | None = None,
        timeout_s: int | None = None,
    ) -> dict:
        selected_card = None
        if method_selection:
            method_id = method_selection.get("chosen_method_id")
            selected_card = self.method_cards.get(method_id)
            if selected_card is None:
                raise ValueError(f"selector chose unknown method card {method_id!r}")
        user = prompts.proposer_user_prompt(
            journal_lines, mode, parent_id, parent_code, directive, focus_note, traceback_tail,
            parent_history=parent_history,
            method_selection=method_selection,
            selected_method_card=selected_card,
            streak_state=streak_state,
            context_mode=context_mode,
            full_context=full_context,
            prior_runs=prior_runs,
            timeout_s=timeout_s,
        )
        text = self._call("proposer", self.static_prefix, user, self.max_code_tokens)
        spec = extract_json_spec(text)
        expected_delta = spec.get("expected_delta")
        if (isinstance(expected_delta, bool)
                or not isinstance(expected_delta, (int, float))
                or not math.isfinite(expected_delta)):
            raise ValueError("proposer reply missing numeric expected_delta")
        spec["expected_delta"] = float(expected_delta)
        basis = spec.get("expected_delta_basis")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("proposer reply missing expected_delta_basis")
        spec.setdefault("action", mode)
        spec.setdefault("parent", parent_id)
        return spec

    def repair_farm_close_plan(self, spec: dict, error: str) -> dict:
        """Give an invalid typed plan one bounded proposer repair attempt."""
        user = prompts.farm_close_repair_user_prompt(spec, error)
        text = self._call("proposer", self.static_prefix, user, self.max_code_tokens)
        repaired = extract_json_spec(text)
        repaired.setdefault("hypothesis", spec.get("hypothesis", "farm-close repair"))
        repaired.setdefault("expected_delta", spec.get("expected_delta"))
        repaired.setdefault("expected_delta_basis", spec.get("expected_delta_basis"))
        repaired.setdefault("action", spec.get("action", "improve"))
        repaired.setdefault("parent", spec.get("parent"))
        return repaired

    def select_method(
        self,
        journal_lines: list[str],
        parent_history: list,
        streak_state: dict,
        excluded_families: list[str] | None = None,
        enforce_family_exclusion: bool = False,
        dataset: str = "pure",
        prior_runs: str | None = None,
        preference_note: str | None = None,
    ) -> dict:
        user = prompts.selector_user_prompt(
            two_tier_methods_text(method_cards_for_dataset(self.methods_text, dataset)),
            journal_lines, parent_history, streak_state,
            excluded_families=excluded_families,
            enforce_family_exclusion=enforce_family_exclusion,
            dataset=dataset,
            prior_runs=prior_runs,
            preference_note=preference_note,
        )
        text = self._call(
            "selector",
            prompts.SELECTOR_SYSTEM,
            user,
            int(self.role_max_tokens.get("selector", 1200)),
        )
        selection = extract_json_object(text)
        required = {"diagnosis", "chosen_method_id", "citation", "why", "rejected"}
        missing = required - selection.keys()
        if missing:
            raise ValueError(f"selector reply missing fields: {sorted(missing)}")
        method_id = selection["chosen_method_id"]
        if method_id not in self.method_cards:
            raise ValueError(f"selector chose unknown method card {method_id!r}")
        if not isinstance(selection["rejected"], list):
            raise ValueError("selector rejected field must be a list")
        return selection

    def plan_exploration(
        self, calibration_result: dict, max_iters: int, method_families: list[str]
    ) -> dict:
        text = self._call(
            "reflector",
            prompts.EXPLORATION_PLAN_SYSTEM,
            prompts.exploration_plan_user_prompt(
                calibration_result, max_iters, method_families
            ),
            int(self.role_max_tokens.get("reflector", 1200)),
        )
        plan = extract_json_object(text)
        required = {"initial_draft_slots", "family_priorities", "rationale"}
        missing = required - plan.keys()
        if missing:
            raise ValueError(f"exploration plan missing fields: {sorted(missing)}")
        slots = plan["initial_draft_slots"]
        if isinstance(slots, bool) or not isinstance(slots, int):
            raise ValueError("exploration plan initial_draft_slots must be an integer")
        if not isinstance(plan["family_priorities"], list) or not all(
            isinstance(family, str) and family.strip()
            for family in plan["family_priorities"]
        ):
            raise ValueError("exploration plan family_priorities must be a list of strings")
        if not isinstance(plan["rationale"], str) or not plan["rationale"].strip():
            raise ValueError("exploration plan rationale must be a non-empty string")
        return plan

    def fix(self, code: str, traceback_tail: str) -> str:
        text = self._call(
            "fixer", prompts.FIXER_SYSTEM, prompts.fixer_user_prompt(code, traceback_tail), self.max_code_tokens
        )
        fixed = extract_code_block(text)
        return fixed if fixed else text

    def reflect(self, journal_lines: list[str]) -> str:
        return self._call(
            "reflector",
            prompts.REFLECTOR_SYSTEM,
            prompts.reflector_user_prompt(self.menu_text, journal_lines, list(self.method_cards)),
            1200,
        ).strip()

    def self_critique(self, journal_summary: str) -> str:
        return self._call(
            "reflector",
            prompts.SELF_CRITIQUE_SYSTEM,
            prompts.self_critique_user_prompt(journal_summary),
            1200,
        ).strip()
