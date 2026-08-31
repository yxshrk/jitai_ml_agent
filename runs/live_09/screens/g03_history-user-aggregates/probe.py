import argparse
import os

import numpy as np
import pandas as pd


SMOOTHING = 5.0


def duration_bucket(values):
    """Coarse 10-second buckets, capped at 300 seconds; zero remains bucket 0."""
    x = pd.to_numeric(values, errors="coerce").fillna(0).to_numpy(np.int64)
    return np.minimum(np.maximum(x, 0) // 10_000, 30).astype(np.int16)


def attach_history_stats(valid, aggregates, keys, prefix, user_prior, global_prior):
    count_col = f"{prefix}_support"
    pos_col = f"{prefix}_positives"

    valid = valid.merge(aggregates, how="left", on=keys, sort=False)
    support = valid[count_col].fillna(0).to_numpy(np.float64)
    positives = valid[pos_col].fillna(0).to_numpy(np.float64)
    prior = valid["user_id"].map(user_prior).fillna(global_prior).to_numpy(np.float64)

    valid[f"{prefix}_rate"] = (
        positives + SMOOTHING * prior
    ) / (support + SMOOTHING)
    valid[count_col] = support
    valid.drop(columns=[pos_col], inplace=True)
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train.csv")
    valid_path = os.path.join(args.data_dir, "valid.csv")
    video_path = os.path.join(args.data_dir, "video_features_basic.csv")

    train = pd.read_csv(
        train_path,
        usecols=["user_id", "video_id", "tab", "duration_ms", "long_view"],
    )
    valid = pd.read_csv(
        valid_path,
        usecols=["row_id", "user_id", "video_id", "tab", "duration_ms"],
    )
    videos = (
        pd.read_csv(video_path, usecols=["video_id", "author_id"])
        .drop_duplicates("video_id", keep="first")
    )

    train = train.merge(videos, how="left", on="video_id", sort=False)
    valid = valid.merge(videos, how="left", on="video_id", sort=False)
    train["author_id"] = train["author_id"].fillna(-1)
    valid["author_id"] = valid["author_id"].fillna(-1)

    train["dur_bucket"] = duration_bucket(train["duration_ms"])
    valid["dur_bucket"] = duration_bucket(valid["duration_ms"])

    labels = pd.to_numeric(train["long_view"], errors="coerce").fillna(0).astype(np.float64)
    train["long_view"] = labels
    global_prior = float(labels.mean())

    user_stats = train.groupby("user_id", sort=False, observed=True)["long_view"].agg(
        ["sum", "count"]
    )
    user_prior = (
        (user_stats["sum"] + SMOOTHING * global_prior)
        / (user_stats["count"] + SMOOTHING)
    )

    author_stats = (
        train.groupby(["user_id", "author_id"], sort=False, observed=True)["long_view"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "author_positives",
                "count": "author_support",
            }
        )
    )
    tab_stats = (
        train.groupby(["user_id", "tab"], sort=False, observed=True)["long_view"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "tab_positives",
                "count": "tab_support",
            }
        )
    )
    duration_stats = (
        train.groupby(["user_id", "dur_bucket"], sort=False, observed=True)["long_view"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "duration_positives",
                "count": "duration_support",
            }
        )
    )

    valid = attach_history_stats(
        valid,
        author_stats,
        ["user_id", "author_id"],
        "author",
        user_prior,
        global_prior,
    )
    valid = attach_history_stats(
        valid,
        tab_stats,
        ["user_id", "tab"],
        "tab",
        user_prior,
        global_prior,
    )
    valid = attach_history_stats(
        valid,
        duration_stats,
        ["user_id", "dur_bucket"],
        "duration",
        user_prior,
        global_prior,
    )

    feature_columns = [
        "author_rate",
        "author_support",
        "tab_rate",
        "tab_support",
        "duration_rate",
        "duration_support",
    ]
    output = valid[["row_id"] + feature_columns].copy()
    output.sort_values("row_id", kind="stable", inplace=True)
    output.reset_index(drop=True, inplace=True)

    expected_ids = np.arange(len(output), dtype=np.int64)
    if not np.array_equal(output["row_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("valid row_id must be contiguous and ordered from zero")

    values = output[feature_columns].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature value generated")

    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
