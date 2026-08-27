"""Dollar spend cap, enforced in code (user-set hard requirement).

A persistent ledger at logs/spend.json is shared across ALL runs and processes:
every LLM call appends provider, model, tokens in/out, and computed cost.
Updates are atomic (fcntl lock on a sidecar lockfile + write-temp + os.replace).

Pricing lives in agent/models.toml under [pricing] — conservative ESTIMATES per
1M tokens, costs always rounded UP. An unknown model is priced at the most
expensive row. The hard cap BUDGET_USD comes from .env (default 25.0); before
each call the worst case (estimated input + max_output_tokens at the model's
output rate) is checked against ledger_total and the call is refused — no
override flag exists.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_PATH = Path(__file__).resolve().parent / "models.toml"
LEDGER_PATH = ROOT / "logs" / "spend.json"
DEFAULT_CAP_USD = 25.0


class BudgetExhausted(RuntimeError):
    """Raised instead of making an LLM call that could exceed the hard cap."""


def load_cap(env_path: Path | None = None) -> float:
    value = os.environ.get("BUDGET_USD")
    if value is None:
        path = env_path or ROOT / ".env"
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line.startswith("BUDGET_USD="):
                    value = line.split("=", 1)[1].strip()
                    break
    return float(value) if value else DEFAULT_CAP_USD


def _pricing_table() -> dict[str, dict[str, float]]:
    with MODELS_PATH.open("rb") as handle:
        table = tomllib.load(handle).get("pricing", {})
    if not table:
        raise RuntimeError("no [pricing] table in agent/models.toml")
    return table


def rates_for(model: str, table: dict[str, dict[str, float]] | None = None) -> tuple[float, float]:
    """(input, output) USD per 1M tokens; '*' suffixes are prefix patterns;
    unknown models get the most expensive row."""
    table = table or _pricing_table()
    if model in table:
        row = table[model]
    else:
        row = None
        for name, candidate in table.items():
            if name.endswith("*") and model.startswith(name[:-1]):
                row = candidate
                break
        if row is None:
            row = max(table.values(), key=lambda r: (r["out"], r["in"]))
    return float(row["in"]), float(row["out"])


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimated cost, rounded UP to the next $0.0001."""
    rate_in, rate_out = rates_for(model)
    raw = tokens_in * rate_in / 1e6 + tokens_out * rate_out / 1e6
    return math.ceil(raw * 10_000) / 10_000


class Budget:
    """Cross-process ledger + hard-cap gate."""

    def __init__(self, cap_usd: float | None = None, ledger_path: Path = LEDGER_PATH) -> None:
        self.cap_usd = load_cap() if cap_usd is None else cap_usd
        self.ledger_path = ledger_path
        self.lock_path = ledger_path.with_suffix(".lock")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _locked(self):
        handle = self.lock_path.open("w")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def _read(self) -> dict:
        if self.ledger_path.exists():
            try:
                return json.loads(self.ledger_path.read_text())
            except json.JSONDecodeError:
                pass
        return {"entries": [], "total_usd": 0.0}

    def total(self) -> float:
        lock = self._locked()
        try:
            return float(self._read()["total_usd"])
        finally:
            lock.close()

    def precheck(self, model: str, est_input_tokens: int, max_output_tokens: int) -> None:
        """Refuse the call if its worst case could exceed the hard cap."""
        worst = cost_usd(model, est_input_tokens, max_output_tokens)
        current = self.total()
        if current + worst > self.cap_usd:
            raise BudgetExhausted(
                f"refusing LLM call: ledger ${current:.4f} + worst-case ${worst:.4f} "
                f"> cap ${self.cap_usd:.2f}"
            )

    def record(self, provider: str, model: str, tokens_in: int, tokens_out: int,
               note: str = "") -> float:
        cost = cost_usd(model, tokens_in, tokens_out)
        lock = self._locked()
        try:
            ledger = self._read()
            ledger["entries"].append({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": provider,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "usd": cost,
                **({"note": note} if note else {}),
            })
            ledger["total_usd"] = round(ledger["total_usd"] + cost, 4)
            tmp = self.ledger_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(ledger, indent=1) + "\n")
            os.replace(tmp, self.ledger_path)
        finally:
            lock.close()
        return cost
