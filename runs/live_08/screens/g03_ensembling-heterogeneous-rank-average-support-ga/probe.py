import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train = pd.read_csv(
        os.path.join(args.data_dir, "train.csv"),
        usecols=["user_id", "tab", "long_view"],
    )
    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["user_id", "tab"],
    )

    grouped = train.groupby(["user_id", "tab"], sort=False, observed=True)["long_view"]
    support = grouped.agg(ut_pos_support="sum", ut_total_support="count")
    support["ut_neg_support"] = support["ut_total_support"] - support["ut_pos_support"]

    valid_keys = pd.MultiIndex.from_frame(valid[["user_id", "tab"]])
    matched = support.reindex(valid_keys)

    pos = matched["ut_pos_support"].fillna(0).to_numpy(dtype=np.float64)
    neg = matched["ut_neg_support"].fillna(0).to_numpy(dtype=np.float64)

    features = pd.DataFrame(
        {
            "row_id": np.arange(len(valid), dtype=np.int64),
            "user_tab_positive_support": pos,
            "user_tab_negative_support": neg,
        }
    )

    values = features.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Generated features contain NaN or infinite values")

    os.makedirs(args.out_dir, exist_ok=True)
    features.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
