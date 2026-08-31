import argparse
import os

import numpy as np
import pandas as pd


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
    videos = pd.read_csv(
        video_path,
        usecols=["video_id", "author_id"],
    ).drop_duplicates("video_id", keep="last")

    expected_row_id = np.arange(len(valid), dtype=np.int64)
    if not np.array_equal(valid["row_id"].to_numpy(dtype=np.int64), expected_row_id):
        raise ValueError("valid.csv row_id must be contiguous and in file order")

    author_map = videos.set_index("video_id")["author_id"]
    train["author_id"] = train["video_id"].map(author_map).fillna(-1).astype(np.int64)
    valid["author_id"] = valid["video_id"].map(author_map).fillna(-1).astype(np.int64)

    # Coarse ten-second duration cohorts, matching the historical-rate
    # hypothesis rather than exposing a bucket identifier as the feature.
    train_duration = pd.to_numeric(train["duration_ms"], errors="coerce").fillna(0)
    valid_duration = pd.to_numeric(valid["duration_ms"], errors="coerce").fillna(0)
    train["duration_bucket"] = (
        np.maximum(train_duration.to_numpy(dtype=np.float64), 0.0) // 10000.0
    ).astype(np.int64)
    valid["duration_bucket"] = (
        np.maximum(valid_duration.to_numpy(dtype=np.float64), 0.0) // 10000.0
    ).astype(np.int64)

    train["long_view"] = (
        pd.to_numeric(train["long_view"], errors="raise").astype(np.float64)
    )
    global_prior = float(train["long_view"].mean())
    smoothing = 5.0

    user_stats = train.groupby("user_id", sort=False, observed=True)["long_view"].agg(
        user_count="size",
        user_positive="sum",
    )
    user_count = valid["user_id"].map(user_stats["user_count"]).fillna(0.0)
    user_positive = valid["user_id"].map(user_stats["user_positive"]).fillna(0.0)
    user_prior = np.where(
        user_count.to_numpy(dtype=np.float64) > 0,
        user_positive.to_numpy(dtype=np.float64)
        / np.maximum(user_count.to_numpy(dtype=np.float64), 1.0),
        global_prior,
    )

    valid["_order"] = np.arange(len(valid), dtype=np.int64)

    def add_history_rate(frame, keys, prefix):
        stats = (
            train.groupby(keys, sort=False, observed=True)["long_view"]
            .agg(**{f"{prefix}_count": "size", f"{prefix}_positive": "sum"})
            .reset_index()
        )
        frame = frame.merge(stats, on=keys, how="left", sort=False, validate="many_to_one")
        count_col = f"{prefix}_count"
        positive_col = f"{prefix}_positive"
        frame[count_col] = frame[count_col].fillna(0.0).astype(np.float64)
        frame[positive_col] = frame[positive_col].fillna(0.0).astype(np.float64)
        frame[f"{prefix}_rate"] = (
            frame[positive_col].to_numpy(dtype=np.float64) + smoothing * user_prior
        ) / (frame[count_col].to_numpy(dtype=np.float64) + smoothing)
        return frame.drop(columns=[positive_col])

    valid = add_history_rate(valid, ["user_id", "author_id"], "author_history")
    valid = add_history_rate(valid, ["user_id", "tab"], "tab_history")
    valid = add_history_rate(
        valid, ["user_id", "duration_bucket"], "duration_history"
    )
    valid = valid.sort_values("_order", kind="stable")

    output = valid[
        [
            "row_id",
            "author_history_rate",
            "author_history_count",
            "tab_history_rate",
            "tab_history_count",
            "duration_history_rate",
            "duration_history_count",
        ]
    ].copy()

    feature_values = output.drop(columns=["row_id"]).to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Generated history features contain NaN or infinity")

    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
