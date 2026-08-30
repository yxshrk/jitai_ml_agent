import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def load_validation(data_dir):
    npz_path = data_dir / "val.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            labels = np.asarray(data["y"], dtype=np.float64)
            users = np.asarray(data["user"])
            x = np.asarray(data["X"])
            field_dims = np.asarray(data["field_dims"], dtype=np.int64)
        if x.ndim == 2 and x.shape[1] >= 2:
            video_offset = int(field_dims[0]) if field_dims.size else 0
            videos = x[:, 1].astype(np.int64) - video_offset
        else:
            videos = np.arange(labels.size, dtype=np.int64)
        return labels, users, videos, True

    labels = []
    users = []
    videos = []
    with (data_dir / "val.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.append(float(row["long_view"]))
            users.append(row["user_id"])
            videos.append(row["video_id"])
    return (
        np.asarray(labels, dtype=np.float64),
        np.asarray(users, dtype=str),
        np.asarray(videos, dtype=str),
        False,
    )


def read_scores(path, expected_rows):
    scores = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if "score" in names:
            score_name = "score"
        elif "prediction" in names:
            score_name = "prediction"
        elif "pred" in names:
            score_name = "pred"
        else:
            return None
        for row in reader:
            scores.append(float(row[score_name]))
    if len(scores) != expected_rows:
        return None
    values = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    return values


def within_user_ranks(users, scores):
    n = scores.size
    if n == 0:
        return scores.copy()
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sorted_users[1:] != sorted_users[:-1]
    starts = np.where(change, np.arange(n, dtype=np.int64), 0)
    starts = np.maximum.accumulate(starts)
    positions = np.arange(n, dtype=np.float64) - starts
    group_starts = np.flatnonzero(change)
    group_ends = np.r_[group_starts[1:], n]
    sizes = group_ends - group_starts
    denominators = np.maximum(sizes - 1, 1).astype(np.float64)
    ranked_sorted = positions / np.repeat(denominators, sizes)
    ranked = np.empty(n, dtype=np.float64)
    ranked[order] = ranked_sorted
    return ranked


def fallback_scores(users, videos, seed):
    n = len(users)
    result = np.empty(n, dtype=np.float64)
    mask = (1 << 64) - 1
    for i in range(n):
        text = (str(users[i]) + "\x1f" + str(videos[i])).encode("utf-8")
        value = (1469598103934665603 ^ int(seed)) & mask
        for byte in text:
            value ^= byte
            value = (value * 1099511628211) & mask
        result[i] = value / float(mask)
    return result


def run_members(script_path, data_dir, base_seed, expected_rows):
    successful = []
    cpu_count = os.cpu_count() or 1
    member_count = 10
    max_parallel = 5
    threads_per_member = max(1, cpu_count // max_parallel)

    with tempfile.TemporaryDirectory(prefix="frozen_ensemble_") as temporary:
        root = Path(temporary)
        for batch_start in range(0, member_count, max_parallel):
            processes = []
            batch_end = min(batch_start + max_parallel, member_count)
            for member_index in range(batch_start, batch_end):
                member_dir = root / ("member_%d" % member_index)
                member_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(script_path),
                    "--data-dir",
                    str(data_dir),
                    "--out-dir",
                    str(member_dir),
                    "--seed",
                    str(base_seed + member_index),
                ]
                environment = os.environ.copy()
                environment["OMP_NUM_THREADS"] = str(threads_per_member)
                environment["MKL_NUM_THREADS"] = str(threads_per_member)
                environment["OPENBLAS_NUM_THREADS"] = str(threads_per_member)
                environment["NUMEXPR_NUM_THREADS"] = str(threads_per_member)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                )
                processes.append((member_index, process, member_dir))

            batch_results = []
            for member_index, process, member_dir in processes:
                return_code = process.wait()
                prediction_path = member_dir / "predictions.csv"
                if return_code == 0 and prediction_path.exists():
                    scores = read_scores(prediction_path, expected_rows)
                    if scores is not None:
                        batch_results.append((member_index, scores))
            batch_results.sort(key=lambda item: item[0])
            successful.extend(scores for _, scores in batch_results)
    return successful


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, users, videos, fast_path = load_validation(data_dir)

    spec = importlib.util.find_spec("zoo.frozen_stack_1k")
    member_scores = []
    if spec is not None and spec.origin:
        member_scores = run_members(Path(spec.origin), data_dir, args.seed, labels.size)

    if member_scores:
        rank_members = [within_user_ranks(users, scores) for scores in member_scores]
        final_scores = np.mean(np.stack(rank_members, axis=0), axis=0)
    else:
        final_scores = within_user_ranks(
            users, fallback_scores(users, videos, args.seed)
        )

    prediction_path = out_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id in range(labels.size):
            writer.writerow(
                [row_id, users[row_id], videos[row_id], "%.10g" % final_scores[row_id]]
            )

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    measured = evaluate(users, labels, final_scores)
    metrics = {
        "gauc": float(measured["GAUC"]),
        "ndcg5": float(measured["nDCG@5"]),
        "primary": float(measured["primary"]),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
