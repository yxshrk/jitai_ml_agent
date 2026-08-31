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
        usecols=["user_id", "video_id", "time_ms", "duration_ms"],
    )
    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["row_id", "user_id", "video_id", "time_ms", "duration_ms"],
    )
    videos = pd.read_csv(
        os.path.join(args.data_dir, "video_features_basic.csv"),
        usecols=["video_id", "author_id"],
    ).drop_duplicates("video_id", keep="last")

    n_train = len(train)
    n_valid = len(valid)

    combined = pd.concat(
        [
            train.assign(_valid_pos=-1),
            valid.assign(_valid_pos=np.arange(n_valid, dtype=np.int64)),
        ],
        ignore_index=True,
        sort=False,
    )

    author_map = videos.set_index("video_id")["author_id"]
    author_values = combined["video_id"].map(author_map)
    author_code, _ = pd.factorize(author_values, sort=False)
    combined["_author_code"] = author_code.astype(np.int64)
    combined["_sequence"] = np.arange(n_train + n_valid, dtype=np.int64)

    combined["time_ms"] = pd.to_numeric(
        combined["time_ms"], errors="coerce"
    ).fillna(0).astype(np.int64)
    combined["duration_ms"] = pd.to_numeric(
        combined["duration_ms"], errors="coerce"
    ).fillna(0.0).astype(np.float64)

    ordered = combined.sort_values(
        ["user_id", "time_ms", "_sequence"],
        kind="mergesort",
    ).reset_index(drop=True)

    users = ordered["user_id"].to_numpy()
    times = ordered["time_ms"].to_numpy(dtype=np.int64)
    durations = ordered["duration_ms"].to_numpy(dtype=np.float64)
    authors = ordered["_author_code"].to_numpy(dtype=np.int64)
    valid_pos = ordered["_valid_pos"].to_numpy(dtype=np.int64)

    n = len(ordered)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (users[1:] != users[:-1]) | (times[1:] != times[:-1])

    group_starts = np.flatnonzero(change)
    group_ends = np.r_[group_starts[1:] - 1, n - 1]
    group_sizes = group_ends - group_starts + 1
    row_group = np.repeat(np.arange(len(group_starts)), group_sizes)

    # Every equal-time group observes only the final stable row from the
    # preceding distinct timestamp, so equal-time rows never observe each other.
    previous_group_end = np.full(len(group_starts), -1, dtype=np.int64)
    if len(group_starts) > 1:
        has_same_user_predecessor = (
            users[group_starts[1:]] == users[group_ends[:-1]]
        )
        previous_group_end[1:] = np.where(
            has_same_user_predecessor,
            group_ends[:-1],
            -1,
        )

    previous_index = previous_group_end[row_group]
    predecessor_exists = previous_index >= 0
    safe_previous = np.maximum(previous_index, 0)

    gap_ms = np.where(
        predecessor_exists,
        times - times[safe_previous],
        -1000,
    ).astype(np.float64)
    previous_duration = np.where(
        predecessor_exists,
        durations[safe_previous],
        -1.0,
    )
    threshold_ms = np.where(
        predecessor_exists,
        np.minimum(np.maximum(previous_duration, 0.0), 18000.0),
        -1000.0,
    )

    same_author = (
        predecessor_exists
        & (authors >= 0)
        & (authors[safe_previous] >= 0)
        & (authors == authors[safe_previous])
    )

    # State: 0=no predecessor, 1=inferred skip, 2=inferred completion,
    # 3=ambiguous because the previous duration is unknown or the gap exceeds
    # the established 30-minute session boundary.
    ambiguous = predecessor_exists & (
        (previous_duration <= 0.0) | (gap_ms > 30.0 * 60.0 * 1000.0)
    )
    inferred_completion = (
        predecessor_exists
        & ~ambiguous
        & (gap_ms >= threshold_ms)
    )
    inferred_skip = predecessor_exists & ~ambiguous & ~inferred_completion

    state = np.zeros(n, dtype=np.int64)
    state[inferred_skip] = 1
    state[inferred_completion] = 2
    state[ambiguous] = 3

    pseudo_completion = np.full(n, -1.0, dtype=np.float64)
    pseudo_completion[inferred_skip] = 0.0
    pseudo_completion[inferred_completion] = 1.0

    # Explicit categorical cross requested by the hypothesis.
    state_author_cross = state * 2 + same_author.astype(np.int64)

    selected = valid_pos >= 0
    positions = valid_pos[selected]

    output = pd.DataFrame(
        {
            "row_id": valid["row_id"].to_numpy(dtype=np.int64),
            "previous_gap_seconds": np.empty(n_valid, dtype=np.float64),
            "previous_threshold_seconds": np.empty(n_valid, dtype=np.float64),
            "gap_minus_threshold_seconds": np.empty(n_valid, dtype=np.float64),
            "previous_same_author": np.empty(n_valid, dtype=np.float64),
            "inferred_previous_completion": np.empty(n_valid, dtype=np.float64),
            "inferred_previous_state": np.empty(n_valid, dtype=np.float64),
            "state_same_author_cross": np.empty(n_valid, dtype=np.float64),
        }
    )

    output.loc[positions, "previous_gap_seconds"] = gap_ms[selected] / 1000.0
    output.loc[positions, "previous_threshold_seconds"] = (
        threshold_ms[selected] / 1000.0
    )
    output.loc[positions, "gap_minus_threshold_seconds"] = (
        gap_ms[selected] - threshold_ms[selected]
    ) / 1000.0
    output.loc[positions, "previous_same_author"] = same_author[selected].astype(float)
    output.loc[positions, "inferred_previous_completion"] = pseudo_completion[selected]
    output.loc[positions, "inferred_previous_state"] = state[selected].astype(float)
    output.loc[positions, "state_same_author_cross"] = state_author_cross[
        selected
    ].astype(float)

    feature_columns = output.columns[1:]
    output[feature_columns] = output[feature_columns].replace(
        [np.inf, -np.inf], -1.0
    ).fillna(-1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    output.to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
