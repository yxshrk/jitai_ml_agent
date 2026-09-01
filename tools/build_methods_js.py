"""Regenerate site/methods.js from agent/METHODS.md (the 42-card public library).

Schema mirrors the existing file: id, title, kind, mechanism, citation, status
(status_pure text + ' - status_1k: ...'), evidence (numeric reference_primary
or '' when none). Usage: uv run python tools/build_methods_js.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.brain import parse_method_cards, parse_method_card_metadata  # noqa: E402


def field(card: str, name: str) -> str:
    m = re.search(rf"^- {re.escape(name)}:\s*(.+)$", card, re.M)
    return m.group(1).strip() if m else ""


def main():
    text = (ROOT / "agent/METHODS.md").read_text()
    cards = parse_method_cards(text)
    out = []
    for cid, card in cards.items():
        title = re.match(r"^### [a-z0-9-]+: (.+)$", card, re.M).group(1).strip()
        meta = parse_method_card_metadata(card)
        status_pure = field(card, "status_pure") or field(card, "status")
        status_1k = field(card, "status_1k")
        status = status_pure + (f" - status_1k: {status_1k}" if status_1k else "")
        ref = meta.get("reference_primary")
        evidence = "" if ref is None else f"{ref:.4f}".rstrip("0").rstrip(".")
        out.append({"id": cid, "title": title, "kind": field(card, "kind"),
                    "mechanism": field(card, "mechanism"), "citation": field(card, "citation"),
                    "status": status, "evidence": evidence})
    (ROOT / "site/methods.js").write_text("window.METHODS=" + json.dumps(out) + ";")
    print(f"methods.js: {len(out)} cards")


if __name__ == "__main__":
    main()
