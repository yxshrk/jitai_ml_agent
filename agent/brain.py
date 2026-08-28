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
            if isinstance(spec, dict) and "code" in spec:
                return spec
        except (json.JSONDecodeError, ValueError):
            continue
    # Last resort: pull hypothesis + a python fence out of free-form text.
    code = extract_code_block(text)
    hyp = re.search(r'"hypothesis"\s*:\s*"([^"]+)"', text)
    if code and hyp:
        return {"hypothesis": hyp.group(1), "code": code}
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
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_text}],
        )
        usage = response.usage
        tokens_in = (
            usage.input_tokens
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, tokens_in, usage.output_tokens


class Brain:
    """Provider-agnostic brain used by the loop."""

    def __init__(
        self,
        menu_text: str,
        provider: str | None = None,
        model_overrides: dict[str, str] | None = None,
        max_code_tokens: int = 4000,
        budget=None,
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
        self.menu_text = menu_text
        self.methods_text = METHODS_PATH.read_text()
        self.method_cards = parse_method_cards(self.methods_text)
        self.max_code_tokens = max_code_tokens
        self.meter = TokenMeter()
        # Byte-identical static prefix -> prompt cache hits on both providers.
        self.static_prefix = prompts.TASK_BRIEF + "\n\n## Improvement menu\n" + menu_text

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
        )
        text = self._call("proposer", self.static_prefix, user, self.max_code_tokens)
        spec = extract_json_spec(text)
        expected_delta = spec.get("expected_delta")
        if (isinstance(expected_delta, bool)
                or not isinstance(expected_delta, (int, float))):
            raise ValueError("proposer reply missing numeric expected_delta")
        spec["expected_delta"] = float(expected_delta)
        spec.setdefault("action", mode)
        spec.setdefault("parent", parent_id)
        return spec

    def select_method(
        self,
        journal_lines: list[str],
        parent_history: list,
        streak_state: dict,
    ) -> dict:
        user = prompts.selector_user_prompt(
            self.methods_text, journal_lines, parent_history, streak_state
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
