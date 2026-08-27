"""DCN-lite + fm_feats features, with optional aux heads / item aggregates / loss mix.

Merges the two zoo winners (MENU #3+#9 features into the MENU #4 architecture) and
exposes the sweep knobs used in zoo/EXPERIMENTS.md: --cross-layers, --hidden,
--aux-weight (0 disables the MTL heads), --item-agg (Tier-3 train-window Bayesian-
smoothed video/author long_view rates as bucketed fields), --bpr-weight.
CLI per CONTRACTS.md section 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import (encode_extra_column, load_for_args, make_parser, set_seed,
                        train_and_report)
from zoo.fm_feats import add_features

SPLITS = ("train", "valid", "test")


def add_item_aggregates(ds: dict, prior_strength: float = 20.0,
                        n_buckets: int = 20) -> dict:
    """Video / author train-window long_view rates, Bayesian-smoothed toward the
    global train rate (strength ~20), quantile-bucketed as extra categorical fields.
    Keys are the OFFICIAL encoded ids (X col 1 = video, col 2 = author), which are
    consistent across splits with a shared UNK slot. TRAIN WINDOW ONLY."""
    tr = ds["train"]
    g = float(tr["y"].mean())

    def rates(col: int) -> dict[str, np.ndarray]:
        keys = tr["X"][:, col].astype(np.int64)
        cnt = np.bincount(keys)
        pos = np.bincount(keys, weights=tr["y"].astype(np.float64))
        rate = (pos + prior_strength * g) / (cnt + prior_strength)
        out = {}
        for n in SPLITS:
            k = ds[n]["X"][:, col].astype(np.int64)
            r = np.full(len(k), g)
            seen = k < len(cnt)
            r[seen] = rate[k[seen]]  # cnt==0 keys already smooth to exactly g
            out[n] = r
        return out

    offset = ds["field_dims_total"]
    for col in (1, 2):
        rr = rates(col)
        edges = np.quantile(rr["train"], np.linspace(0, 1, n_buckets + 1)[1:-1])
        bucketed = {n: np.searchsorted(edges, rr[n]).astype(np.int64) for n in SPLITS}
        enc, offset = encode_extra_column(bucketed["train"], bucketed, offset)
        for n in SPLITS:
            ds[n]["X"] = np.hstack([ds[n]["X"].astype(np.int64), enc[n][:, None]])
    ds["field_dims_total"] = offset
    return ds


class DCNLite(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int, n_cross: int = 2,
                 hidden: int = 64, aux_names: tuple[str, ...] = ()):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        d = n_fields * k
        self.cross_w = nn.ModuleList([nn.Linear(d, d) for _ in range(n_cross)])
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, 1)
                                    for name in ("main", *aux_names)})

    def forward(self, x: torch.Tensor):
        x0 = self.emb(x).flatten(1)
        xl = x0
        for w in self.cross_w:
            xl = x0 * w(xl) + xl
        h = self.mlp(xl)
        if len(self.heads) == 1:
            return self.heads["main"](h).squeeze(1)
        return {name: head(h).squeeze(1) for name, head in self.heads.items()}


def main() -> None:
    ap = make_parser(__doc__)
    ap.add_argument("--cross-layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help=">0 enables click + effective-view aux heads")
    ap.add_argument("--item-agg", action="store_true")
    ap.add_argument("--bpr-weight", type=float, default=0.5)
    args = ap.parse_args()
    ds = add_features(load_for_args(args))
    if args.item_agg:
        ds = add_item_aggregates(ds)
    set_seed(args.seed)
    tr = ds["train"]
    aux_targets = None
    aux_names: tuple[str, ...] = ()
    if args.aux_weight > 0:
        aux_targets = {
            "click": tr["click"].astype(np.float32),
            "effective_view": (tr["play_time_ms"]
                               >= np.minimum(tr["duration_ms"], 18_000)).astype(np.float32),
        }
        aux_names = tuple(aux_targets)
    model = DCNLite(ds["field_dims_total"], tr["X"].shape[1], args.k,
                    n_cross=args.cross_layers, hidden=args.hidden, aux_names=aux_names)
    train_and_report(model, ds, args, aux_targets=aux_targets,
                     aux_weight=args.aux_weight, bpr_weight=args.bpr_weight)


if __name__ == "__main__":
    main()
