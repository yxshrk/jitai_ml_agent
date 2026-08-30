"""Validation-only temporal/context FM experiment for KuaiRand-Pure.

The organizer baseline uses five categorical fields. This candidate leaves those
fields intact and tests three deliberately small, previously-unused context
signals: time of day, day of week, and randomized-exposure status. It never
loads or scores the test split and always chooses checkpoints on validation.
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
BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
ALLOWED_EXTRAS = {
    "hour", "weekday", "is_rand", "video_age",
    "user_tab", "author_tab", "tab_hour", "tab_is_rand",
}
ENGAGEMENT_TARGETS = ("long_view", "is_click", "is_like", "is_follow", "is_comment", "is_forward")
HYPOTHESIS = (
    "Time of day, day of week, and randomized-exposure status add ranking signal "
    "beyond the organizer's five static FM fields."
)


def weekday(date_value):
    text = str(date_value)
    return str(date(int(text[:4]), int(text[4:6]), int(text[6:])).weekday())


def date_ordinal(date_value):
    text = str(date_value)
    return date(int(text[:4]), int(text[4:6]), int(text[6:])).toordinal()


def video_age_bucket(impression_day, upload_day):
    if upload_day is None:
        return "UNK"
    age = max(0, impression_day - upload_day)
    if age <= 1:
        return "0_1"
    if age <= 3:
        return "2_3"
    if age <= 7:
        return "4_7"
    if age <= 30:
        return "8_30"
    if age <= 90:
        return "31_90"
    return "91_plus"


def load_rows(data_dir):
    video_metadata = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                upload_day = date.fromisoformat(row["upload_dt"]).toordinal()
            except (TypeError, ValueError):
                upload_day = None
            video_metadata[row["video_id"]] = (row["author_id"], upload_day)

    rows = {name: [] for name in SPLITS}
    for filename in (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    ):
        with open(Path(data_dir) / filename, newline="") as handle:
            for record in csv.DictReader(handle):
                date_value = int(record["date"])
                split = next(
                    (
                        name
                        for name, (start, end) in SPLITS.items()
                        if start <= date_value <= end
                    ),
                    None,
                )
                if split is None:
                    continue
                author_id, upload_day = video_metadata.get(record["video_id"], ("UNK", None))
                user_id, tab = record["user_id"], record["tab"]
                hour, is_rand = str(int(record["hourmin"]) // 100), record["is_rand"]
                rows[split].append(
                    {
                        "user_id": user_id,
                        "video_id": record["video_id"],
                        "author_id": author_id,
                        "tab": tab,
                        "duration_ms": float(record["duration_ms"]),
                        "hour": hour,
                        "weekday": weekday(date_value),
                        "is_rand": is_rand,
                        "video_age": video_age_bucket(date_ordinal(date_value), upload_day),
                        # Wide crosses are all derived from columns available at
                        # impression time; they do not use outcomes or test data.
                        "user_tab": f"{user_id}|{tab}",
                        "author_tab": f"{author_id}|{tab}",
                        "tab_hour": f"{tab}|{hour}",
                        "tab_is_rand": f"{tab}|{is_rand}",
                        "label": 1 if record["long_view"] != "0" else 0,
                        "targets": {
                            target: 1 if record[target] != "0" else 0 for target in ENGAGEMENT_TARGETS
                        },
                    }
                )
    return rows


def encode(rows, extra_fields):
    duration_edges = np.quantile(
        np.asarray([row["duration_ms"] for row in rows["train"]]), np.linspace(0, 1, 11)[1:-1]
    )
    fields = BASE_FIELDS + list(extra_fields)

    def raw(row):
        values = [
            row["user_id"],
            row["video_id"],
            row["author_id"],
            row["tab"],
            str(int(np.searchsorted(duration_edges, row["duration_ms"]))),
        ]
        values.extend(row[field] for field in extra_fields)
        return values

    vocabularies = [dict() for _ in fields]
    for row in rows["train"]:
        for field_index, value in enumerate(raw(row)):
            if value not in vocabularies[field_index]:
                vocabularies[field_index][value] = len(vocabularies[field_index])
    unknowns = [len(vocabulary) for vocabulary in vocabularies]
    dimensions = [len(vocabulary) + 1 for vocabulary in vocabularies]
    offsets = np.cumsum([0] + dimensions[:-1]).astype(np.int32)

    encoded = {}
    for split, split_rows in rows.items():
        features = np.empty((len(split_rows), len(fields)), dtype=np.int32)
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
    return encoded, int(sum(dimensions)), fields


def append_log(path, record):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run(args):
    extra_fields = tuple(field.strip() for field in args.extra_features.split(",") if field.strip())
    invalid_fields = set(extra_fields) - ALLOWED_EXTRAS
    if invalid_fields:
        raise ValueError(f"Unsupported extra fields: {sorted(invalid_fields)}")
    rows = load_rows(args.data_dir)
    encoded, dimension, fields = encode(rows, extra_fields)
    train_x, train_y, _ = encoded["train"]
    valid_x, valid_y, valid_users = encoded["valid"]
    model = organizer_baseline.FM(dimension, k=args.embedding_dim, lr=args.learning_rate, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    log_path = Path(args.run_log) if args.run_log else None
    if log_path is not None and log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {log_path}")

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
        record = {
            "phase": "temporal_fm",
            "iteration": epoch,
            "hypothesis": HYPOTHESIS,
            "change": f"FM fields: {fields}",
            "train_loss": round(float(np.mean(losses)), 7),
            "metrics": metrics,
            "error_or_recovery": None,
            "manual_interventions": 0,
        }
        append_log(log_path, record)
        print(
            f"epoch {epoch:2d} | loss {record['train_loss']:.4f} | GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
            best_record = record
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    model.V, model.W, model.b = best_state
    if args.validation_scores_out:
        score_path = Path(args.validation_scores_out)
        if score_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing validation scores: {score_path}")
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, model.predict(valid_x))
    print("\nBest validation result")
    print(json.dumps(best_record, indent=2, sort_keys=True))
    return best_record, model.predict(valid_x), valid_users, valid_y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--extra_features", default="hour,weekday,is_rand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None, help="Optional JSONL iteration log path")
    parser.add_argument(
        "--validation_scores_out",
        default=None,
        help="Optional .npy path for validation logits from the selected checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
