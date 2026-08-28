"""Best confirmed history/data-level config: 7-day recency-weighted DCN-lite.

Validation primary (official evaluator), seeds 42/43/44:
0.6057299 / 0.6028683 / 0.6043514 = 0.6043165 +/- 0.0011685,
delta +0.0027165 over the 0.6016 baseline.

Contract CLI:
  uv run python zoo/hist_best.py --data-dir data/real_ws --out-dir <o> [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo.hist_campaign import parser, run


def main() -> None:
    args = parser(__doc__).parse_args()
    # Freeze the single confirmed winner regardless of campaign-only selectors.
    args.experiment = "recency"
    args.half_life = 7.0
    args.with_affinity = False
    run(args)


if __name__ == "__main__":
    main()
