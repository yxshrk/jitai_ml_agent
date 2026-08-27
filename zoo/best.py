"""Best zoo config (see zoo/EXPERIMENTS.md): DCN-lite + fm_feats features + aux heads.

Frozen winner of the 2026-08-28 sweep: dcn_feats with k=16, 2 cross layers,
hidden=128 MLP, aux heads (click + effective-view proxy) at weight 0.1, hybrid
0.5*BPR + 0.5*logloss, early stopping on valid GAUC.

Validation primary, official evaluator: seeds 42/43/44 = 0.6048/0.6028/0.6042,
mean 0.6039 +- 0.0010, delta +0.0023 vs baseline 0.6016 (>= epsilon 0.002).
CLI per CONTRACTS.md section 3: uv run python zoo/best.py --data-dir real --out-dir <o> [--seed 42]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import load_for_args, make_parser, set_seed, train_and_report
from zoo.dcn_feats import DCNLite
from zoo.fm_feats import add_features

HIDDEN = 128
CROSS_LAYERS = 2
AUX_WEIGHT = 0.1
BPR_WEIGHT = 0.5


def main() -> None:
    args = make_parser(__doc__).parse_args()
    ds = add_features(load_for_args(args))
    set_seed(args.seed)
    tr = ds["train"]
    aux_targets = {
        "click": tr["click"].astype(np.float32),
        "effective_view": (tr["play_time_ms"]
                           >= np.minimum(tr["duration_ms"], 18_000)).astype(np.float32),
    }
    model = DCNLite(ds["field_dims_total"], tr["X"].shape[1], args.k,
                    n_cross=CROSS_LAYERS, hidden=HIDDEN,
                    aux_names=tuple(aux_targets))
    train_and_report(model, ds, args, aux_targets=aux_targets,
                     aux_weight=AUX_WEIGHT, bpr_weight=BPR_WEIGHT)


if __name__ == "__main__":
    main()
