import argparse
import os

import numpy as np
import pandas as pd


SMOOTHING = 5.0
DURATION_EDGES_MS = np.array(
    [6_000, 10_000, 14_000, 18_000, 30_000, 60_000, 120_000],
    dtype=np.float64,
)


def duration_bucket(values):
    duration = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(np.float64)
    duration = np.maximum(duration, 0.0)
    bucket = np.zeros(len(duration), dtype=np.int16)
    known = duration > 0
    bucket[known] = (
        1 + np.searchsorted(DURATION_EDGES_MS, duration[known], side="right")
    ).astype(np.int16)
    return bucket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train = pd.read_csv(
        os.path.join(args.data_dir, "train.csv"),
        usecols=[
            "user_id",
            "tab",
            "duration_ms",
            "is_click",
            "long_view",
        ],
    )
    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["row_id", "user_id", "tab", "duration_ms"],
    )

    train["tab"] = pd.to_numeric(train["tab"], errors="coerce").fillna(-1).astype(np.int16)
    valid["tab"] = pd.to_numeric(valid["tab"], errors="coerce").fillna(-1).astype(np.int16)
    train["dur_bucket"] = duration_bucket(train["duration_ms"])
    valid["dur_bucket"] = duration_bucket(valid["duration_ms"])

    train["is_click"] = (
        pd.to_numeric(train["is_click"], errors="coerce").fillna(0).clip(0, 1).astype(np.int8)
    )
    train["long_view"] = (
        pd.to_numeric(train["long_view"], errors="coerce").fillna(0).clip(0, 1).astype(np.int8)
    )
    train["clicked_long_view"] = (
        train["is_click"].to_numpy(np.int8) * train["long_view"].to_numpy(np.int8)
    )

    global_click_rate = float(train["is_click"].mean())
    total_clicks = float(train["is_click"].sum())
    global_conversion = (
        float(train["clicked_long_view"].sum()) / total_clicks
        if total_clicks > 0
        else 0.0
    )

    user_stats = (
        train.groupby("user_id", sort=False, observed=True)
        .agg(
            user_impressions=("is_click", "size"),
            user_clicks=("is_click", "sum"),
            user_clicked_long=("clicked_long_view", "sum"),
        )
        .reset_index()
    )
    user_stats["user_click_prior"] = (
        user_stats["user_clicks"] + SMOOTHING * global_click_rate
    ) / (user_stats["user_impressions"] + SMOOTHING)
    user_stats["user_conversion_prior"] = (
        user_stats["user_clicked_long"] + SMOOTHING * global_conversion
    ) / (user_stats["user_clicks"] + SMOOTHING)

    user_priors = user_stats[
        ["user_id", "user_click_prior", "user_conversion_prior"]
    ]

    user_tab = (
        train.groupby(["user_id", "tab"], sort=False, observed=True)
        .agg(
            ut_impressions=("is_click", "size"),
            ut_clicks=("is_click", "sum"),
        )
        .reset_index()
    )

    user_duration = (
        train.groupby(["user_id", "dur_bucket"], sort=False, observed=True)
        .agg(
            ud_clicks=("is_click", "sum"),
            ud_clicked_long=("clicked_long_view", "sum"),
        )
        .reset_index()
    )

    features = valid[["row_id", "user_id", "tab", "dur_bucket"]].merge(
        user_priors, on="user_id", how="left", sort=False
    )
    features = features.merge(
        user_tab, on=["user_id", "tab"], how="left", sort=False
    )
    features = features.merge(
        user_duration, on=["user_id", "dur_bucket"], how="left", sort=False
    )

    features["user_click_prior"] = features["user_click_prior"].fillna(global_click_rate)
    features["user_conversion_prior"] = features["user_conversion_prior"].fillna(
        global_conversion
    )
    for column in ["ut_impressions", "ut_clicks", "ud_clicks", "ud_clicked_long"]:
        features[column] = features[column].fillna(0.0)

    features["user_tab_click_rate"] = (
        features["ut_clicks"] + SMOOTHING * features["user_click_prior"]
    ) / (features["ut_impressions"] + SMOOTHING)

    features["user_duration_click_to_long_rate"] = (
        features["ud_clicked_long"]
        + SMOOTHING * features["user_conversion_prior"]
    ) / (features["ud_clicks"] + SMOOTHING)

    output = features[
        [
            "row_id",
            "user_tab_click_rate",
            "ut_impressions",
            "user_duration_click_to_long_rate",
            "ud_clicks",
        ]
    ].copy()
    output["row_id"] = output["row_id"].astype(np.int64)
    output["ut_impressions"] = output["ut_impressions"].astype(np.float64)
    output["ud_clicks"] = output["ud_clicks"].astype(np.float64)

    expected_row_id = np.arange(len(valid), dtype=np.int64)
    if not np.array_equal(output["row_id"].to_numpy(), expected_row_id):
        raise ValueError("valid.csv row_id must be contiguous and in file order")

    numeric = output.drop(columns="row_id").to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite probe feature generated")

    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
