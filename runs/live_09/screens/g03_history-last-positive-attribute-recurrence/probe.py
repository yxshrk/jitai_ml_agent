import argparse
import os

import numpy as np
import pandas as pd


UNK = "__UNK__"


def normalize_id(series):
    missing = series.isna()
    values = series.astype("string").str.strip()
    values = values.mask(
        missing
        | values.isna()
        | values.str.lower().isin(["", "nan", "none", "null", "<na>"]),
        UNK,
    )
    return values.fillna(UNK)


def normalize_first_tag(series):
    missing = series.isna()
    values = series.astype("string").str.strip()
    values = (
        values.str.replace(r"^[\[\(\{]\s*", "", regex=True)
        .str.replace(r"\s*[\]\)\}]$", "", regex=True)
        .str.split(r"\s*[,|;]\s*", n=1, regex=True)
        .str[0]
        .str.strip()
        .str.strip("'\"")
    )
    values = values.mask(
        missing
        | values.isna()
        | values.str.lower().isin(["", "nan", "none", "null", "<na>"]),
        UNK,
    )
    return values.fillna(UNK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    side = pd.read_csv(
        os.path.join(args.data_dir, "video_features_basic.csv"),
        usecols=["video_id", "video_type", "music_id", "tag"],
    )
    side = side.drop_duplicates("video_id", keep="last")
    side["tag_attr"] = normalize_first_tag(side["tag"])
    side["music_attr"] = normalize_id(side["music_id"])
    side["type_attr"] = normalize_id(side["video_type"])
    side = side[["video_id", "tag_attr", "music_attr", "type_attr"]]

    train = pd.read_csv(
        os.path.join(args.data_dir, "train.csv"),
        usecols=["user_id", "video_id", "time_ms", "long_view"],
    )
    train["_order"] = np.arange(len(train), dtype=np.int64)
    positives = train.loc[
        train["long_view"].to_numpy() == 1,
        ["user_id", "video_id", "time_ms", "_order"],
    ].copy()
    positives = positives.merge(side, on="video_id", how="left", sort=False)

    for col in ["tag_attr", "music_attr", "type_attr"]:
        positives[col] = positives[col].fillna(UNK)

    # Stable ordering makes the final positive state deterministic. All validation
    # rows occur after training; equal-time training rows are treated as a group
    # before the state is available to later timestamps.
    positives = positives.sort_values(
        ["user_id", "time_ms", "_order"], kind="mergesort"
    )
    latest = positives.drop_duplicates("user_id", keep="last")
    latest = latest[
        ["user_id", "time_ms", "tag_attr", "music_attr", "type_attr"]
    ].rename(
        columns={
            "time_ms": "latest_positive_time_ms",
            "tag_attr": "latest_tag",
            "music_attr": "latest_music",
            "type_attr": "latest_type",
        }
    )

    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["row_id", "user_id", "video_id", "time_ms"],
    )
    expected_row_id = np.arange(len(valid), dtype=np.int64)
    if not np.array_equal(valid["row_id"].to_numpy(), expected_row_id):
        raise ValueError("valid.csv row_id must be contiguous and in file order")

    valid = valid.merge(side, on="video_id", how="left", sort=False)
    valid = valid.merge(latest, on="user_id", how="left", sort=False)

    for col in ["tag_attr", "music_attr", "type_attr"]:
        valid[col] = valid[col].fillna(UNK)

    has_latest = valid["latest_positive_time_ms"].notna().to_numpy()
    valid_time = pd.to_numeric(valid["time_ms"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    latest_time = pd.to_numeric(
        valid["latest_positive_time_ms"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    strictly_earlier = has_latest & (latest_time < valid_time)
    recency_seconds = np.full(len(valid), -1.0, dtype=np.float64)
    recency_seconds[strictly_earlier] = (
        valid_time[strictly_earlier] - latest_time[strictly_earlier]
    ) / 1000.0

    output = pd.DataFrame({"row_id": expected_row_id})
    output["has_latest_positive"] = strictly_earlier.astype(np.int8)
    output["latest_positive_gap_seconds"] = recency_seconds

    pairs = [
        ("tag", "tag_attr", "latest_tag"),
        ("music", "music_attr", "latest_music"),
        ("type", "type_attr", "latest_type"),
    ]
    for name, current_col, latest_col in pairs:
        current = valid[current_col].astype("string").fillna(UNK)
        previous = valid[latest_col].astype("string").fillna(UNK)
        known_pair = (
            strictly_earlier
            & current.ne(UNK).to_numpy()
            & previous.ne(UNK).to_numpy()
        )
        match = known_pair & current.eq(previous).to_numpy()
        output[f"latest_positive_{name}_match"] = match.astype(np.int8)
        output[f"latest_positive_{name}_known_pair"] = known_pair.astype(np.int8)

    numeric = output.drop(columns=["row_id"]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Probe generated non-finite feature values")

    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
