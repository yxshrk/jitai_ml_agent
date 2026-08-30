"""Validation-only causal-history model blended with the contextual FM.

For every training example, all target-derived history features are calculated
before that example's timestamp is incorporated. Validation features are built
from the completed training history only. The script never reads, scores, or
selects against the test split.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from evaluate import evaluate
from temporal_fm import run as run_contextual_fm


TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)
FEATURE_NAMES = (
    "user_author_rate",
    "user_music_rate",
    "user_video_type_rate",
    "user_tab_rate",
    "video_rate",
    "author_rate",
    "user_author_count",
    "user_music_count",
    "user_video_type_count",
    "user_tab_count",
    "video_count",
    "author_count",
)
HYPOTHESIS = (
    "Causal user-content engagement histories add personalized ranking signal "
    "that the static FM fields do not capture."
)


def date_split(date_value):
    if TRAIN_DATES[0] <= date_value <= TRAIN_DATES[1]:
        return "train"
    if VALID_DATES[0] <= date_value <= VALID_DATES[1]:
        return "valid"
    return None


def load_records(data_dir):
    metadata = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata[row["video_id"]] = (
                row["author_id"],
                row.get("music_id") or "UNK",
                row.get("video_type") or "UNK",
            )

    records = {"train": [], "valid": []}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(Path(data_dir) / filename, newline="") as handle:
            for row in csv.DictReader(handle):
                split = date_split(int(row["date"]))
                if split is None:
                    continue
                author_id, music_id, video_type = metadata.get(row["video_id"], ("UNK", "UNK", "UNK"))
                records[split].append(
                    (
                        int(row["time_ms"]),
                        row["user_id"],
                        row["video_id"],
                        author_id,
                        music_id,
                        video_type,
                        row["tab"],
                        1 if row["long_view"] != "0" else 0,
                    )
                )
    records["train"].sort(key=lambda row: row[0])
    return records


def feature_keys(record):
    _, user_id, video_id, author_id, music_id, video_type, tab, _ = record
    return (
        (user_id, author_id),
        (user_id, music_id),
        (user_id, video_type),
        (user_id, tab),
        video_id,
        author_id,
    )


def feature_values(keys, positives, counts, global_rate, smoothing):
    values = []
    for index, key in enumerate(keys):
        count = counts[index].get(key, 0)
        positive = positives[index].get(key, 0)
        values.append((positive + smoothing * global_rate) / (count + smoothing))
    values.extend(np.log1p(counts[index].get(key, 0)) for index, key in enumerate(keys))
    return values


def update_histories(records, positives, counts):
    for record in records:
        label = record[-1]
        for index, key in enumerate(feature_keys(record)):
            counts[index][key] = counts[index].get(key, 0) + 1
            positives[index][key] = positives[index].get(key, 0) + label


def causal_train_features(train_records, smoothing):
    feature_matrix = np.empty((len(train_records), len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.empty(len(train_records), dtype=np.float32)
    positives = [dict() for _ in range(6)]
    counts = [dict() for _ in range(6)]
    global_positive = global_count = 0
    index = 0
    while index < len(train_records):
        end = index + 1
        timestamp = train_records[index][0]
        while end < len(train_records) and train_records[end][0] == timestamp:
            end += 1
        global_rate = global_positive / global_count if global_count else 0.33
        for row_index in range(index, end):
            record = train_records[row_index]
            feature_matrix[row_index] = feature_values(
                feature_keys(record), positives, counts, global_rate, smoothing
            )
            labels[row_index] = record[-1]
        update_histories(train_records[index:end], positives, counts)
        global_positive += sum(record[-1] for record in train_records[index:end])
        global_count += end - index
        index = end
    return feature_matrix, labels, positives, counts, global_positive / global_count


def validation_features(valid_records, positives, counts, global_rate, smoothing):
    features = np.empty((len(valid_records), len(FEATURE_NAMES)), dtype=np.float32)
    users = []
    labels = np.empty(len(valid_records), dtype=np.float32)
    for index, record in enumerate(valid_records):
        features[index] = feature_values(feature_keys(record), positives, counts, global_rate, smoothing)
        users.append(record[1])
        labels[index] = record[-1]
    return features, users, labels


def fit_logistic(features, labels, epochs, learning_rate, l2):
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = (features - mean) / scale
    weights = np.zeros(standardized.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        logits = standardized @ weights + bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        residual = probabilities - labels
        weights -= learning_rate * ((standardized.T @ residual) / len(labels) + l2 * weights)
        bias -= learning_rate * residual.mean()
    return mean, scale, weights, bias


def predict_logistic(features, mean, scale, weights, bias):
    standardized = (features - mean) / scale
    return standardized @ weights + bias


def standardized(values):
    return (values - values.mean()) / max(values.std(), 1e-6)


def append_log(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run(args):
    records = load_records(args.data_dir)
    train_features, train_labels, positives, counts, global_rate = causal_train_features(
        records["train"], args.smoothing
    )
    valid_features, valid_users, valid_labels = validation_features(
        records["valid"], positives, counts, global_rate, args.smoothing
    )
    mean, scale, weights, bias = fit_logistic(
        train_features, train_labels, args.history_epochs, args.history_learning_rate, args.history_l2
    )
    history_scores = predict_logistic(valid_features, mean, scale, weights, bias)
    history_metrics = {name: float(value) for name, value in evaluate(valid_users, valid_labels, history_scores).items()}
    print("history-only", json.dumps(history_metrics, sort_keys=True))

    context_args = argparse.Namespace(
        data_dir=args.data_dir,
        extra_features="hour,weekday,is_rand",
        seed=args.base_seed,
        embedding_dim=16,
        learning_rate=0.001,
        epochs=40,
        patience=4,
        batch_size=8192,
        run_log=None,
        validation_scores_out=None,
    )
    context_record, context_scores, context_users, context_labels = run_contextual_fm(context_args)
    if context_users != valid_users or not np.array_equal(context_labels, valid_labels):
        raise RuntimeError("History and contextual FM validation rows do not align")

    context_z = standardized(context_scores)
    history_z = standardized(history_scores)
    candidates = []
    for alpha in args.blend_alphas:
        scores = context_z + alpha * history_z
        metrics = {name: float(value) for name, value in evaluate(valid_users, valid_labels, scores).items()}
        candidates.append({"history_weight": alpha, "metrics": metrics})
        print(f"blend history_weight={alpha:.3f} primary={metrics['primary']:.6f}")
    best = max(candidates, key=lambda candidate: candidate["metrics"]["primary"])
    record = {
        "phase": "causal_history_blend",
        "hypothesis": HYPOTHESIS,
        "features": FEATURE_NAMES,
        "history_only_metrics": history_metrics,
        "context_member": context_record,
        "blend_candidates": candidates,
        "selected": best,
        "error_or_recovery": None,
        "manual_interventions": 0,
        "test_data_used": False,
    }
    append_log(Path(args.run_log) if args.run_log else None, record)
    print("\nSelected validation blend")
    print(json.dumps(best, indent=2, sort_keys=True))
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--smoothing", type=float, default=20.0)
    parser.add_argument("--history_epochs", type=int, default=40)
    parser.add_argument("--history_learning_rate", type=float, default=0.5)
    parser.add_argument("--history_l2", type=float, default=1e-4)
    parser.add_argument("--blend_alphas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--run_log", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
