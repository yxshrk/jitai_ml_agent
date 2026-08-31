import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        data_dir / "train.csv",
        usecols=["user_id", "video_id", "long_view"],
    )
    valid = pd.read_csv(
        data_dir / "valid.csv",
        usecols=["row_id", "user_id", "video_id"],
    )
    videos = pd.read_csv(
        data_dir / "video_features_basic.csv",
        usecols=["video_id", "music_id"],
    )

    expected_row_id = np.arange(len(valid), dtype=np.int64)
    if not np.array_equal(valid["row_id"].to_numpy(dtype=np.int64), expected_row_id):
        raise ValueError("valid.csv row_id must be contiguous and in file order")

    # Missing music is retained as its own legal category.
    videos["music_id"] = (
        pd.to_numeric(videos["music_id"], errors="coerce")
        .fillna(-1)
        .astype(np.int64)
    )
    videos = videos.drop_duplicates("video_id", keep="first")

    train = train.merge(videos, on="video_id", how="left", sort=False, validate="many_to_one")
    valid = valid.merge(videos, on="video_id", how="left", sort=False, validate="many_to_one")
    train["music_id"] = train["music_id"].fillna(-1).astype(np.int64)
    valid["music_id"] = valid["music_id"].fillna(-1).astype(np.int64)

    y = train["long_view"].astype(np.float64)
    global_rate = float(y.mean())
    if not np.isfinite(global_rate):
        raise ValueError("Could not estimate global long-view rate")

    # Hierarchical empirical-Bayes prior:
    # first shrink each user's rate toward the train-wide rate, then shrink
    # each user×music rate toward that user's posterior rate.
    user_stats = (
        train.groupby("user_id", sort=False, observed=True)["long_view"]
        .agg(user_pos="sum", user_support="size")
        .reset_index()
    )
    user_prior_strength = 10.0
    user_stats["user_prior_rate"] = (
        user_stats["user_pos"].astype(np.float64)
        + user_prior_strength * global_rate
    ) / (
        user_stats["user_support"].astype(np.float64)
        + user_prior_strength
    )

    pair_stats = (
        train.groupby(["user_id", "music_id"], sort=False, observed=True)["long_view"]
        .agg(user_music_pos="sum", user_music_support="size")
        .reset_index()
    )

    features = valid[["row_id", "user_id", "music_id"]].merge(
        user_stats[["user_id", "user_prior_rate"]],
        on="user_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    features = features.merge(
        pair_stats,
        on=["user_id", "music_id"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    features["user_prior_rate"] = features["user_prior_rate"].fillna(global_rate)
    features["user_music_pos"] = features["user_music_pos"].fillna(0.0).astype(np.float64)
    features["user_music_support"] = (
        features["user_music_support"].fillna(0.0).astype(np.float64)
    )
    features["user_music_neg"] = (
        features["user_music_support"] - features["user_music_pos"]
    )

    pair_prior_strength = 5.0
    features["user_music_rate"] = (
        features["user_music_pos"]
        + pair_prior_strength * features["user_prior_rate"]
    ) / (
        features["user_music_support"] + pair_prior_strength
    )
    features["user_music_affinity"] = (
        features["user_music_rate"] - features["user_prior_rate"]
    )

    output = features[
        [
            "row_id",
            "user_music_rate",
            "user_music_affinity",
            "user_music_support",
            "user_music_pos",
            "user_music_neg",
        ]
    ].copy()

    values = output.drop(columns="row_id").to_numpy(dtype=np.float64)
    if len(output) != len(valid):
        raise ValueError("Feature row count does not match valid.csv")
    if not np.isfinite(values).all():
        raise ValueError("Generated features contain NaN or infinite values")

    output.to_csv(out_dir / "features.csv", index=False)


if __name__ == "__main__":
    main()
