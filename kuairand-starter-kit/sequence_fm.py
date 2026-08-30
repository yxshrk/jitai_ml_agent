"""Validation-only FM with leakage-safe short-term user context.

Each example receives the user's most recent author, music, video type, tab,
and an elapsed-time bucket. Training contexts are built before an event's
timestamp is committed; validation uses only the completed training history.
"""

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import time

import numpy as np

import baseline as organizer_baseline
from evaluate import evaluate


SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428)}
FIELDS = (
    "user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday", "is_rand",
    "previous_author", "previous_music", "previous_video_type", "previous_tab", "previous_gap",
)
HYPOTHESIS = (
    "Causal short-term user context will improve within-user long_view ranking "
    "beyond a static user embedding."
)


def split_for(date_value):
    for name, (start, end) in SPLITS.items():
        if start <= date_value <= end:
            return name
    return None


def weekday(date_value):
    text = str(date_value)
    return str(date(int(text[:4]), int(text[4:6]), int(text[6:])).weekday())


def gap_bucket(milliseconds):
    if milliseconds is None:
        return "UNK"
    seconds = max(0, milliseconds // 1000)
    if seconds < 60:
        return "under_1m"
    if seconds < 10 * 60:
        return "1_10m"
    if seconds < 60 * 60:
        return "10_60m"
    if seconds < 6 * 60 * 60:
        return "1_6h"
    if seconds < 24 * 60 * 60:
        return "6_24h"
    return "over_24h"


def load_rows(data_dir):
    metadata = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            metadata[row["video_id"]] = (
                row["author_id"], row.get("music_id") or "UNK", row.get("video_type") or "UNK"
            )

    rows = {"train": [], "valid": []}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(Path(data_dir) / filename, newline="") as handle:
            for source_row in csv.DictReader(handle):
                date_value = int(source_row["date"])
                split = split_for(date_value)
                if split is None:
                    continue
                author_id, music_id, video_type = metadata.get(source_row["video_id"], ("UNK", "UNK", "UNK"))
                rows[split].append(
                    {
                        "timestamp": int(source_row["time_ms"]),
                        "user_id": source_row["user_id"],
                        "video_id": source_row["video_id"],
                        "author_id": author_id,
                        "music_id": music_id,
                        "video_type": video_type,
                        "tab": source_row["tab"],
                        "duration_ms": float(source_row["duration_ms"]),
                        "hour": str(int(source_row["hourmin"]) // 100),
                        "weekday": weekday(date_value),
                        "is_rand": source_row["is_rand"],
                        "label": 1 if source_row["long_view"] != "0" else 0,
                    }
                )
    return rows


def add_causal_context(rows):
    train_order = sorted(range(len(rows["train"])), key=lambda index: rows["train"][index]["timestamp"])
    user_state = {}
    start = 0
    while start < len(train_order):
        end = start + 1
        timestamp = rows["train"][train_order[start]]["timestamp"]
        while end < len(train_order) and rows["train"][train_order[end]]["timestamp"] == timestamp:
            end += 1
        for order_index in range(start, end):
            row = rows["train"][train_order[order_index]]
            previous = user_state.get(row["user_id"])
            attach_context(row, previous)
        for order_index in range(start, end):
            row = rows["train"][train_order[order_index]]
            user_state[row["user_id"]] = row
        start = end

    # Validation contexts are frozen at the training boundary. This intentionally
    # avoids using any validation impression, label, or outcome as input.
    for row in rows["valid"]:
        attach_context(row, user_state.get(row["user_id"]))


def attach_context(row, previous):
    if previous is None:
        row["previous_author"] = "UNK"
        row["previous_music"] = "UNK"
        row["previous_video_type"] = "UNK"
        row["previous_tab"] = "UNK"
        row["previous_gap"] = "UNK"
        return
    row["previous_author"] = previous["author_id"]
    row["previous_music"] = previous["music_id"]
    row["previous_video_type"] = previous["video_type"]
    row["previous_tab"] = previous["tab"]
    row["previous_gap"] = gap_bucket(row["timestamp"] - previous["timestamp"])


def encode(rows):
    duration_edges = np.quantile(
        np.asarray([row["duration_ms"] for row in rows["train"]]), np.linspace(0, 1, 11)[1:-1]
    )

    def raw(row):
        return [
            row["user_id"], row["video_id"], row["author_id"], row["tab"],
            str(int(np.searchsorted(duration_edges, row["duration_ms"]))), row["hour"], row["weekday"],
            row["is_rand"], row["previous_author"], row["previous_music"], row["previous_video_type"],
            row["previous_tab"], row["previous_gap"],
        ]

    vocabularies = [dict() for _ in FIELDS]
    for row in rows["train"]:
        for field_index, value in enumerate(raw(row)):
            if value not in vocabularies[field_index]:
                vocabularies[field_index][value] = len(vocabularies[field_index])
    unknowns = [len(vocabulary) for vocabulary in vocabularies]
    dimensions = [len(vocabulary) + 1 for vocabulary in vocabularies]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)
    encoded = {}
    for split, split_rows in rows.items():
        features = np.empty((len(split_rows), len(FIELDS)), dtype=np.int32)
        labels = np.empty(len(split_rows), dtype=np.float32)
        users = []
        for row_index, row in enumerate(split_rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = (
                    vocabularies[field_index].get(value, unknowns[field_index]) + offsets[field_index]
                )
            labels[row_index] = row["label"]
            users.append(row["user_id"])
        encoded[split] = (features, labels, users)
    return encoded, int(sum(dimensions))


def append_log(path, record):
    if path is None:
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run(args):
    rows = load_rows(args.data_dir)
    add_causal_context(rows)
    encoded, dimension = encode(rows)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = organizer_baseline.FM(dimension, k=args.embedding_dim, lr=args.learning_rate, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    trajectory = []
    best_primary = -np.inf
    best_state = None
    best_record = None
    stalled_epochs = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train_y))
        losses = [
            model.step(train_x[order[start : start + args.batch_size]], train_y[order[start : start + args.batch_size]])
            for start in range(0, len(order), args.batch_size)
        ]
        metrics = {name: float(value) for name, value in evaluate(valid_users, valid_y, model.predict(valid_x)).items()}
        event = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 7), "metrics": metrics}
        trajectory.append(event)
        print(
            f"epoch {epoch:2d} | loss {event['train_loss']:.4f} | GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            best_record = event
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    record = {
        "phase": "causal_sequence_fm",
        "hypothesis": HYPOTHESIS,
        "fields": FIELDS,
        "best": best_record,
        "trajectory": trajectory,
        "error_or_recovery": None,
        "manual_interventions": 0,
        "test_data_used": False,
    }
    append_log(Path(args.run_log) if args.run_log else None, record)
    print("\nBest validation result")
    print(json.dumps(best_record, indent=2, sort_keys=True))
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
