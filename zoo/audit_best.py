"""Best fresh-eyes audit configuration: DCN-lite plus causal session fields.

Contract CLI: ``uv run python zoo/audit_best.py --data-dir <dir> --out-dir <dir>
[--seed 42]``. The implementation is validation-only and emits half-epoch history.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoo.audit_campaign import parser, run


def main() -> None:
    args = parser(__doc__).parse_args()
    args.lambda_weight = 0.0
    args.duration_heads = False
    args.tab_bias = False
    args.metadata_crosses = False
    args.session_features = True
    run(args)


if __name__ == "__main__":
    main()
