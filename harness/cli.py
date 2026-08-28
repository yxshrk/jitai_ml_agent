"""CLI entry point.

  uv run python -m harness.cli run --data-dir data/synthetic --max-iters 3 \
      --max-tokens 200000 [--provider openai|anthropic] [--dry-run]

--dry-run uses FakeBrain (canned scripts, no API) so the full loop is testable
offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harness.loop import ROOT, Loop, LoopConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the autonomous loop")
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--max-iters", type=int, default=50)
    run.add_argument("--max-hours", type=float, default=6.0)
    run.add_argument("--max-tokens", type=int, default=2_000_000)
    run.add_argument("--max-usd", type=float, default=10.0,
                     help="per-run soft dollar ceiling (hard cap = BUDGET_USD in .env)")
    run.add_argument("--timeout-s", type=int, default=600)
    run.add_argument("--sigma", type=float, default=None,
                     help="skip baseline calibration and use this sigma")
    run.add_argument("--baseline-script", type=Path, default=None,
                     help="baseline node script (default: zoo/fm_torch.py)")
    run.add_argument("--draft-tiers", type=str, default=None,
                     help="comma-separated directives for initial drafts, e.g. 'Tier 4,Tier 4,CURRENT DIRECTIVE'")
    run.add_argument("--context-mode", choices=["compact", "full"], default="compact",
                     help="proposer history: journal one-liners or bounded full node evidence")
    run.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                     help="LLM provider (default: models.toml default_provider)")
    run.add_argument("--dry-run", action="store_true", help="FakeBrain, no API calls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    menu_text = (ROOT / "MENU.md").read_text()
    config = LoopConfig(
        data_dir=args.data_dir.resolve(),
        run_dir=args.run_dir,
        **({"baseline_script": args.baseline_script.resolve()} if args.baseline_script else {}),
        **({"draft_tiers": tuple(t.strip() for t in args.draft_tiers.split(","))} if args.draft_tiers else {}),
        max_iters=args.max_iters,
        max_hours=args.max_hours,
        max_tokens=args.max_tokens,
        max_usd=args.max_usd,
        timeout_s=args.timeout_s,
        sigma=args.sigma,
        context_mode=args.context_mode,
    )
    if args.dry_run:
        from agent.fake_brain import FakeBrain

        brain = FakeBrain(menu_text, root=str(ROOT))
    else:
        from agent.brain import Brain

        brain = Brain(menu_text, provider=args.provider)
    summary = Loop(config, brain).run()
    json.dump(summary, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
