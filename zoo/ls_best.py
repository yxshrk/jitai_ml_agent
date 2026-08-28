"""Clean runner for the best absolute-confirmed long-shot: mild smoothing.

The campaign mean is 0.604186 across seeds 42/43/44.  This clears the requested
absolute baseline gate but does not replace the 0.604756 strong-L0 control.

Usage:
  uv run python zoo/ls_best.py --data-dir data/real_ws --out-dir <dir> --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo.ls_campaign import parser as campaign_parser, run


def parser() -> argparse.ArgumentParser:
    # Retaining one parser keeps the artifact output/runtime contract identical
    # to the measured campaign cell.
    ap = campaign_parser()
    ap.description = __doc__
    ap.set_defaults(idea="smooth", half_epochs=5, long_pos=0.10, long_neg=0.03,
                    short_pos=0.05, short_neg=0.015, far_pos=0.01, far_neg=0.003,
                    boundary_width=0.20)
    for action in ap._actions:
        if action.dest == "idea":
            action.required = False
            action.help = argparse.SUPPRESS
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.idea != "smooth":
        raise ValueError("ls_best.py implements only the measured mild-smoothing config")
    run(args)


if __name__ == "__main__":
    main()
