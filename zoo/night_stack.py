"""Night campaign entry point for the frozen coral stack configuration.

All unspecified arguments retain :mod:`zoo.polish_stack` defaults.  This file
exists so the overnight campaign has an explicit, reproducible frozen point
without changing any existing entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo import polish_stack


CORAL_DEFAULTS = {
    "lr": 6.6e-4,
    "step_decay_factor": 0.73,
    "decay_every": 1.5,
    "decay_start_epoch": 1.0,
    "dropout": 0.32,
}


def parser():
    ap = polish_stack.parser(__doc__)
    ap.set_defaults(**CORAL_DEFAULTS)
    return ap


def main() -> None:
    polish_stack.run(parser().parse_args())


if __name__ == "__main__":
    main()
