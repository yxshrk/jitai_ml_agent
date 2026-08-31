import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["row_id", "user_id"],
    )
    users = pd.read_csv(
        os.path.join(args.data_dir, "user_features.csv"),
        usecols=[
            "user_id",
            "user_active_degree",
            "is_lowactive_period",
            "register_days_range",
        ],
    )

    if users["user_id"].duplicated().any():
        raise ValueError("user_features.csv contains duplicate user_id values")

    attrs = [
        "user_active_degree",
        "is_lowactive_period",
        "register_days_range",
    ]
    for col in attrs:
        users[col] = users[col].astype("string").fillna("__MISSING__")

    users["_cohort_key"] = (
        users["user_active_degree"]
        + "\x1f"
        + users["is_lowactive_period"]
        + "\x1f"
        + users["register_days_range"]
    )

    keys = sorted(users["_cohort_key"].unique().tolist())
    cohort_map = {key: idx + 1 for idx, key in enumerate(keys)}
    users["lifecycle_cohort"] = (
        users["_cohort_key"].map(cohort_map).astype(np.int64)
    )

    merged = valid.merge(
        users[["user_id", "lifecycle_cohort"]],
        on="user_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    cohort = merged["lifecycle_cohort"].fillna(0).to_numpy(dtype=np.int64)

    n = len(valid)
    expected_row_id = np.arange(n, dtype=np.int64)
    if not np.array_equal(valid["row_id"].to_numpy(dtype=np.int64), expected_row_id):
        raise ValueError("valid.csv row_id must be contiguous and in file order")
    if len(merged) != n or not np.isfinite(cohort).all():
        raise ValueError("Invalid lifecycle cohort output")

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": expected_row_id,
            "lifecycle_cohort": cohort,
        }
    ).to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
