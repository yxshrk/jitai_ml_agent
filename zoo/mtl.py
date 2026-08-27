"""Shared-bottom multi-task model (MENU #5).

Shared embeddings + shared MLP trunk; main long_view head (hybrid BPR+logloss)
plus auxiliary heads for click and effective-view proxy
(play_time_ms >= min(duration_ms, 18s)) at aux weight 0.2. Auxiliary signals are
used as TARGETS only, never as inputs. CLI per CONTRACTS.md section 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import load_for_args, make_parser, set_seed, train_and_report

AUX_WEIGHT = 0.2


class SharedBottomMTL(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.trunk = nn.Sequential(nn.Linear(n_fields * k, hidden), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(hidden, 64), nn.ReLU())
        self.heads = nn.ModuleDict({name: nn.Linear(64, 1)
                                    for name in ("main", "click", "effective_view")})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(self.emb(x).flatten(1))
        return {name: head(h).squeeze(1) for name, head in self.heads.items()}


def main() -> None:
    args = make_parser(__doc__).parse_args()
    ds = load_for_args(args)
    set_seed(args.seed)
    tr = ds["train"]
    aux_targets = {
        "click": tr["click"].astype(np.float32),
        "effective_view": (tr["play_time_ms"]
                           >= np.minimum(tr["duration_ms"], 18_000)).astype(np.float32),
    }
    n_fields = tr["X"].shape[1]
    model = SharedBottomMTL(ds["field_dims_total"], n_fields, args.k)
    train_and_report(model, ds, args, aux_targets=aux_targets, aux_weight=AUX_WEIGHT)


if __name__ == "__main__":
    main()
