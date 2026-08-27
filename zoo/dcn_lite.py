"""DCN-lite (MENU #4): field embeddings + 2 cross layers + small MLP head.

Official 5-field encoding, same hybrid 0.5*BPR + 0.5*logloss loss and early
stopping on valid GAUC as fm_bpr. CLI per CONTRACTS.md section 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import load_for_args, make_parser, set_seed, train_and_report


class DCNLite(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int, n_cross: int = 2,
                 hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, std=0.01)
        d = n_fields * k
        self.cross_w = nn.ModuleList([nn.Linear(d, d) for _ in range(n_cross)])
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.emb(x).flatten(1)                     # (B, F*k)
        xl = x0
        for w in self.cross_w:                          # DCNv2 cross: x0 * (W xl + b) + xl
            xl = x0 * w(xl) + xl
        return self.mlp(xl).squeeze(1)


def main() -> None:
    args = make_parser(__doc__).parse_args()
    ds = load_for_args(args)
    set_seed(args.seed)
    n_fields = ds["train"]["X"].shape[1]
    model = DCNLite(ds["field_dims_total"], n_fields, args.k)
    train_and_report(model, ds, args)


if __name__ == "__main__":
    main()
