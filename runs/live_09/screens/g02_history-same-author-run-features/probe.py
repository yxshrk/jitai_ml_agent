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
        usecols=["user_id", "video_id", "time_ms"],
    )
    valid = pd.read_csv(
        os.path.join(args.data_dir, "valid.csv"),
        usecols=["row_id", "user_id", "video_id", "time_ms"],
    )
    videos = pd.read_csv(
        os.path.join(args.data_dir, "video_features_basic.csv"),
        usecols=["video_id", "author_id"],
    ).drop_duplicates("video_id", keep="first")

    if not np.array_equal(valid["row_id"].to_numpy(), np.arange(len(valid))):
        raise ValueError("valid row_id must be contiguous and in file order")

    # Encode known authors, then give videos with unknown authors distinct
    # synthetic author identities so all missing authors are not conflated.
    known_author_codes, known_authors = pd.factorize(videos["author_id"], sort=False)
    videos = videos.copy()
    videos["_author_code"] = known_author_codes.astype(np.int64)
    video_to_author = videos.set_index("video_id")["_author_code"]

    combined_video = pd.concat(
        [train["video_id"], valid["video_id"]], ignore_index=True
    )
    author_code = combined_video.map(video_to_author).to_numpy(dtype=np.float64)

    missing = author_code < 0
    if missing.any():
        missing_codes, _ = pd.factorize(combined_video[missing], sort=False)
        author_code[missing] = len(known_authors) + missing_codes

    author_code = author_code.astype(np.int64)
    users = pd.concat(
        [train["user_id"], valid["user_id"]], ignore_index=True
    ).to_numpy(dtype=np.int64)
    times = pd.concat(
        [train["time_ms"], valid["time_ms"]], ignore_index=True
    ).to_numpy(dtype=np.int64)

    n_train = len(train)
    n_total = len(users)
    original_order = np.arange(n_total, dtype=np.int64)

    # Stable chronological order within user. File order is used only to define
    # state after timestamp ties; rows sharing a timestamp do not observe one
    # another when their own feature is computed.
    order = np.lexsort((original_order, times, users))
    su = users[order]
    st = times[order]
    sa = author_code[order]
    idx = np.arange(n_total, dtype=np.int64)

    # Ordinary stable-order same-author run position. This is used only to
    # summarize the state immediately before each distinct timestamp group.
    run_boundary = np.ones(n_total, dtype=bool)
    if n_total > 1:
        run_boundary[1:] = (su[1:] != su[:-1]) | (sa[1:] != sa[:-1])
    run_start = np.maximum.accumulate(np.where(run_boundary, idx, 0))
    stable_run_so_far = idx - run_start

    time_boundary = np.ones(n_total, dtype=bool)
    if n_total > 1:
        time_boundary[1:] = (su[1:] != su[:-1]) | (st[1:] != st[:-1])
    time_start = np.maximum.accumulate(np.where(time_boundary, idx, 0))
    prev_idx = time_start - 1

    run_so_far = np.zeros(n_total, dtype=np.int64)
    has_strictly_earlier = prev_idx >= 0
    rows = np.flatnonzero(has_strictly_earlier)
    prior = prev_idx[rows]
    same_author = sa[rows] == sa[prior]
    matched_rows = rows[same_author]
    matched_prior = prior[same_author]
    run_so_far[matched_rows] = stable_run_so_far[matched_prior] + 1
    np.minimum(run_so_far, 5, out=run_so_far)

    # Return to concatenated file order and retain validation rows only.
    run_original = np.empty(n_total, dtype=np.int64)
    run_original[order] = run_so_far
    valid_feature = run_original[n_train:]

    if len(valid_feature) != len(valid) or not np.isfinite(valid_feature).all():
        raise RuntimeError("invalid feature output")

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": np.arange(len(valid), dtype=np.int64),
            "same_author_run_so_far_cap5": valid_feature,
        }
    ).to_csv(os.path.join(args.out_dir, "features.csv"), index=False)


if __name__ == "__main__":
    main()
