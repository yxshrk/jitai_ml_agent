"""fm_bpr + MENU #3 duration features + MENU #9 temporal context.

Extra fields appended to the official 5: 50 train-quantile duration buckets,
duration<=18s indicator, dur50-bucket x tab cross, hour-of-day bucket, day-of-week.
Same hybrid loss and early stopping as fm_bpr. CLI per CONTRACTS.md section 3.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zoo.common import (encode_extra_column, load_for_args, make_parser, set_seed,
                        train_and_report)
from zoo.fm_bpr import FM

SPLITS = ("train", "valid", "test")


def add_features(ds: dict) -> dict:
    tr = ds["train"]
    # 50 equal-frequency duration buckets from the TRAIN window only.
    edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 51)[1:-1])
    b50 = {n: np.searchsorted(edges, ds[n]["duration_ms"]).astype(np.int64) for n in SPLITS}
    short = {n: (ds[n]["duration_ms"] <= 18_000).astype(np.int64) for n in SPLITS}
    cross = {n: b50[n] * 100 + ds[n]["tab"].astype(np.int64) for n in SPLITS}
    hour = {n: (ds[n]["hourmin"] // 100).astype(np.int64) for n in SPLITS}

    def dow(dates: np.ndarray) -> np.ndarray:
        uniq = {int(d): datetime.date(int(d) // 10000, int(d) // 100 % 100,
                                      int(d) % 100).weekday() for d in np.unique(dates)}
        return np.asarray([uniq[int(d)] for d in dates], dtype=np.int64)

    dows = {n: dow(ds[n]["date"]) for n in SPLITS}

    offset = ds["field_dims_total"]
    cols = []
    for raw in (b50, short, cross, hour, dows):
        enc, offset = encode_extra_column(raw["train"], raw, offset)
        cols.append(enc)
    for n in SPLITS:
        extra = np.column_stack([c[n] for c in cols])
        ds[n]["X"] = np.hstack([ds[n]["X"].astype(np.int64), extra])
    ds["field_dims_total"] = offset
    return ds


def main() -> None:
    args = make_parser(__doc__).parse_args()
    ds = add_features(load_for_args(args))
    set_seed(args.seed)
    model = FM(ds["field_dims_total"], args.k)
    train_and_report(model, ds, args)


if __name__ == "__main__":
    main()
