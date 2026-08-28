"""FakeBrain: canned, API-free brain for --dry-run and tests.

Serves fast numpy-only scripts that respect the CONTRACTS.md 3 interface, in a
fixed sequence; `fix` returns a canned repaired script. Tests can inject their
own script sequence.
"""

from __future__ import annotations

from agent.brain import METHODS_PATH, TokenMeter, parse_method_card_metadata, parse_method_cards

_TEMPLATE = '''\
"""Canned {name} script (FakeBrain)."""
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path({root!r})
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from harness.evaluate_provisional import evaluate


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    train = read(a.data_dir / "train.csv")
    val = read(a.data_dir / "val.csv")

    prior_n = sum(int(r["long_view"]) for r in train)
    prior = prior_n / len(train)
    counts = {{}}
    for r in train:
        key = {key_expr}
        n, s = counts.get(key, (0, 0.0))
        counts[key] = (n + 1, s + int(r["long_view"]))
    smooth = {smooth}
    scores = []
    for r in val:
        key = {key_expr}
        n, s = counts.get(key, (0, 0.0))
        scores.append((s + smooth * prior) / (n + smooth) {extra})

    metrics = evaluate(
        [int(r["user_id"]) for r in val],
        [int(r["long_view"]) for r in val],
        scores,
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (r, sc) in enumerate(zip(val, scores)):
            w.writerow([i, r["user_id"], r["video_id"], f"{{sc:.10f}}"])
    with open(a.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
'''


def canned_script(name: str, key_expr: str, smooth: float = 10.0, extra: str = "", root: str = "") -> str:
    return _TEMPLATE.format(name=name, key_expr=key_expr, smooth=smooth, extra=extra, root=root)


def default_sequence(root: str) -> list[dict]:
    return [
        {
            "hypothesis": "Smoothed per-video long_view rate should beat the FM baseline on synthetic data",
            "expected_delta": 0.004,
            "expected_delta_basis": "The item-aggregates card reports measured primary 0.6038.",
            "code": canned_script("video-rate", 'r["video_id"]', 10.0, root=root),
        },
        {
            "hypothesis": "Adding tab to the aggregation key captures context (expect +0.002)",
            "expected_delta": 0.002,
            "expected_delta_basis": "The item-aggregates card reports measured primary 0.6038.",
            "code": canned_script("video-tab-rate", '(r["video_id"], r["tab"])', 5.0, root=root),
        },
        {
            "hypothesis": "A duration<=18s indicator tilt should help the duration-defined label",
            "expected_delta": 0.003,
            "expected_delta_basis": "The duration-regime-heads card estimates +0.002-0.006 primary.",
            "code": canned_script(
                "video-dur-rate", 'r["video_id"]', 10.0,
                extra='+ (0.01 if int(r["duration_ms"]) <= 18000 else 0.0)', root=root,
            ),
        },
        {
            "hypothesis": "Weaker smoothing sharpens per-video estimates (expect small gain)",
            "expected_delta": 0.001,
            "expected_delta_basis": "The item-aggregates card reports measured primary 0.6038.",
            "code": canned_script("video-rate-s2", 'r["video_id"]', 2.0, root=root),
        },
        {
            "hypothesis": "Hour-of-day key adds temporal context",
            "expected_delta": 0.0015,
            "expected_delta_basis": "The session-time-features card estimates +0.001-0.005 primary.",
            "code": canned_script("video-hour-rate", '(r["video_id"], int(r["hourmin"]) // 100)', 5.0, root=root),
        },
    ]


class FakeBrain:
    """Deterministic stand-in for Brain. No network, canned outputs."""

    provider = "fake"
    usd_run = 0.0
    usd_total = 0.0
    models = {"selector": "fake", "proposer": "fake", "fixer": "fake", "reflector": "fake"}

    def __init__(self, menu_text: str = "", scripts: list[dict] | None = None, fixes: list[str] | None = None,
                 root: str = "") -> None:
        self.scripts = scripts if scripts is not None else default_sequence(root)
        self.fixes = fixes or []
        self.meter = TokenMeter()
        self._i = 0
        self._fix_i = 0
        self.selection_streak_states: list[dict] = []
        self.proposal_streak_states: list[dict] = []
        self.methods_text = METHODS_PATH.read_text()
        self.method_cards = parse_method_cards(self.methods_text)

    def propose(self, journal_lines, mode, parent_id, parent_code, directive=None,
                focus_note=None, traceback_tail=None, **_kwargs) -> dict:
        self.proposal_streak_states.append(dict(_kwargs.get("streak_state") or {}))
        spec = dict(self.scripts[min(self._i, len(self.scripts) - 1)])
        self._i += 1
        spec.setdefault("action", mode)
        spec.setdefault("parent", parent_id)
        spec.setdefault("expected_delta", 0.0)
        spec.setdefault(
            "expected_delta_basis",
            "Journal line node_000 provides the measured baseline for this expected delta.",
        )
        self.meter.add("fake/fake/proposer", 100, 50)
        return spec

    def select_method(self, journal_lines, parent_history, streak_state,
                      excluded_families=None, enforce_family_exclusion=False) -> dict:
        self.selection_streak_states.append(dict(streak_state))
        self.meter.add("fake/fake/selector", 80, 40)
        excluded = set(excluded_families or [])
        preferred = "regularization-schedule"
        eligible = [
            method_id for method_id, card in self.method_cards.items()
            if not parse_method_card_metadata(card)["measured_dead"]
            and not (set(parse_method_card_metadata(card)["treats"]) & excluded)
        ]
        chosen = preferred if preferred in eligible else (eligible[0] if eligible else preferred)
        return {
            "diagnosis": "overfit",
            "chosen_method_id": chosen,
            "citation": "MENU CURRENT DIRECTIVE",
            "why": "The measured learning curves peak early, so use the untried compound package.",
            "rejected": [
                {
                    "method_id": "listwise-softmax",
                    "reason": "Measured dead at 0.5991 primary.",
                }
            ],
        }

    def fix(self, code: str, traceback_tail: str) -> str:
        self.meter.add("fake/fake/fixer", 50, 25)
        if self._fix_i < len(self.fixes):
            fixed = self.fixes[self._fix_i]
            self._fix_i += 1
            return fixed
        return code  # no fix available: return unchanged (will fail again)

    def reflect(self, journal_lines) -> str:
        self.meter.add("fake/fake/reflector", 40, 20)
        return "Focus on item-side aggregate features (tier 3)."
