"""Frozen best single-model schedule-sweep config.

DCN-lite, k=16, hidden=128, two cross layers, hybrid 0.5 BPR/logloss with aux
weight 0.1, MLP dropout 0.2, embedding dropout 0.1, AdamW weight decay 1e-5,
step LR decay 0.5/epoch, with raw-or-EMA checkpoint selection (EMA decay 0.9,
start epoch 2). Validation primary over seeds 42/43/44: 0.605040 ± 0.000314
(official evaluator). EMA is selected only for seed 42; raw wins seeds 43/44.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from zoo.sweep_train import parser as sweep_parser
from zoo.sweep_train import train


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=None,
                    help="train row cap for smoke tests only")
    return ap


def frozen_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = sweep_parser().parse_args(("--data-dir", cli.data_dir,
                                      "--out-dir", cli.out_dir,
                                      "--seed", str(cli.seed)))
    args.subsample = cli.subsample
    args.dropout = 0.2
    args.embedding_dropout = 0.1
    args.weight_decay = 1e-5
    args.embedding_weight_decay = 1e-5
    args.schedule = "step"
    args.average = "ema"
    args.ema_decay = 0.9
    args.ema_start = 2
    return args


def main() -> None:
    train(frozen_args(parser().parse_args()))


if __name__ == "__main__":
    main()
