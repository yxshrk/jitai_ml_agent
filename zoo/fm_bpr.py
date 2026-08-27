"""FM with hybrid 0.5*within-user-BPR + 0.5*logloss and early stopping on valid GAUC.

MENU #1 + #2 over the official 5-field encoding. CLI per CONTRACTS.md section 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import load_for_args, make_parser, set_seed, train_and_report


class FM(nn.Module):
    def __init__(self, dim: int, k: int):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        self.lin = nn.Embedding(dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.lin.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)                                   # (B,F,k)
        s = e.sum(1)
        inter = 0.5 * (s.square() - e.square().sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + inter


def main() -> None:
    args = make_parser(__doc__).parse_args()
    ds = load_for_args(args)
    set_seed(args.seed)
    model = FM(ds["field_dims_total"], args.k)
    train_and_report(model, ds, args)


if __name__ == "__main__":
    main()
