"""Validation-only neural sequence ranker for KuaiRand-Pure.

The model combines FM interactions, a small deep tower, and mean-pooled recent
author history. Context for both train and validation is strictly causal: a
record never sees its own outcome, and validation receives only train history.
"""

import argparse
import csv
from collections import defaultdict, deque
from datetime import date
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from evaluate import evaluate


SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428)}
FIELDS = ("user_id", "video_id", "author_id", "tab", "dur_bucket", "hour", "weekday", "is_rand")
HYPOTHESIS = (
    "A pooled causal author-history representation will capture short-term user "
    "interests that static FM embeddings miss."
)


def split_for(date_value):
    for name, (start, end) in SPLITS.items():
        if start <= date_value <= end:
            return name
    return None


def weekday(date_value):
    text = str(date_value)
    return str(date(int(text[:4]), int(text[4:6]), int(text[6:])).weekday())


def load_rows(data_dir):
    authors = {}
    with open(Path(data_dir) / "video_features_basic_pure.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            authors[row["video_id"]] = row["author_id"]
    rows = {"train": [], "valid": []}
    for filename in ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"):
        with open(Path(data_dir) / filename, newline="") as handle:
            for source_row in csv.DictReader(handle):
                split = split_for(int(source_row["date"]))
                if split is None:
                    continue
                rows[split].append(
                    {
                        "timestamp": int(source_row["time_ms"]),
                        "user_id": source_row["user_id"],
                        "video_id": source_row["video_id"],
                        "author_id": authors.get(source_row["video_id"], "UNK"),
                        "tab": source_row["tab"],
                        "duration_ms": float(source_row["duration_ms"]),
                        "hour": str(int(source_row["hourmin"]) // 100),
                        "weekday": weekday(int(source_row["date"])),
                        "is_rand": source_row["is_rand"],
                        "label": 1 if source_row["long_view"] != "0" else 0,
                    }
                )
    return rows


def add_history(rows, history_length):
    histories = defaultdict(lambda: deque(maxlen=history_length))
    order = sorted(range(len(rows["train"])), key=lambda index: rows["train"][index]["timestamp"])
    start = 0
    while start < len(order):
        end = start + 1
        timestamp = rows["train"][order[start]]["timestamp"]
        while end < len(order) and rows["train"][order[end]]["timestamp"] == timestamp:
            end += 1
        for order_index in range(start, end):
            row = rows["train"][order[order_index]]
            row["history_authors"] = tuple(histories[row["user_id"]])
        for order_index in range(start, end):
            row = rows["train"][order[order_index]]
            histories[row["user_id"]].append(rows["train"][order[order_index]]["author_id"])
        start = end
    for row in rows["valid"]:
        row["history_authors"] = tuple(histories[row["user_id"]])


def encode(rows, history_length):
    duration_edges = np.quantile(
        np.asarray([row["duration_ms"] for row in rows["train"]]), np.linspace(0, 1, 11)[1:-1]
    )

    def raw(row):
        return (
            row["user_id"], row["video_id"], row["author_id"], row["tab"],
            str(int(np.searchsorted(duration_edges, row["duration_ms"]))), row["hour"], row["weekday"], row["is_rand"],
        )

    vocabularies = [dict() for _ in FIELDS]
    for row in rows["train"]:
        for field_index, value in enumerate(raw(row)):
            if value not in vocabularies[field_index]:
                vocabularies[field_index][value] = len(vocabularies[field_index])
    unknowns = [len(vocabulary) for vocabulary in vocabularies]
    dimensions = [len(vocabulary) + 1 for vocabulary in vocabularies]
    author_padding = dimensions[2]
    encoded = {}
    for split, split_rows in rows.items():
        features = np.empty((len(split_rows), len(FIELDS)), dtype=np.int64)
        history = np.full((len(split_rows), history_length), author_padding, dtype=np.int64)
        labels = np.empty(len(split_rows), dtype=np.float32)
        users = []
        for row_index, row in enumerate(split_rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = vocabularies[field_index].get(value, unknowns[field_index])
            for history_index, author_id in enumerate(row["history_authors"][-history_length:]):
                history[row_index, history_index] = vocabularies[2].get(author_id, unknowns[2])
            labels[row_index] = row["label"]
            users.append(row["user_id"])
        encoded[split] = (features, history, labels, users)
    return encoded, dimensions, author_padding


class SequenceDeepFM(nn.Module):
    def __init__(self, field_dimensions, author_padding, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(dimension + (1 if index == 2 else 0), embedding_dim, padding_idx=author_padding if index == 2 else None)
                for index, dimension in enumerate(field_dimensions)
            ]
        )
        self.linear = nn.ModuleList(
            [nn.Embedding(dimension + (1 if index == 2 else 0), 1, padding_idx=author_padding if index == 2 else None)
             for index, dimension in enumerate(field_dimensions)]
        )
        input_dimension = embedding_dim * (len(field_dimensions) + 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dimension, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1)
        )
        self.bias = nn.Parameter(torch.zeros(()))
        # PyTorch's default unit-variance embedding initialization makes the FM
        # interaction term explode with eight categorical fields.  Match the
        # organizer FM's small 0.01 initialization so the neural extension
        # starts from a calibrated logit scale.
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
            if embedding.padding_idx is not None:
                embedding.weight.data[embedding.padding_idx].zero_()
        for linear in self.linear:
            nn.init.zeros_(linear.weight)
            if linear.padding_idx is not None:
                linear.weight.data[linear.padding_idx].zero_()
        for layer in self.deep:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features, history_authors):
        embeddings = [embedding(features[:, index]) for index, embedding in enumerate(self.embeddings)]
        stacked = torch.stack(embeddings, dim=1)
        fm = 0.5 * ((stacked.sum(dim=1) ** 2).sum(dim=1) - (stacked**2).sum(dim=(1, 2)))
        linear = sum(layer(features[:, index]).squeeze(1) for index, layer in enumerate(self.linear))
        history_embeddings = self.embeddings[2](history_authors)
        history_mask = (history_authors != self.embeddings[2].padding_idx).unsqueeze(-1)
        history_mean = (history_embeddings * history_mask).sum(dim=1) / history_mask.sum(dim=1).clamp_min(1)
        sequence_match = (history_mean * (embeddings[1] + embeddings[2])).sum(dim=1)
        deep = self.deep(torch.cat(embeddings + [history_mean], dim=1)).squeeze(1)
        return self.bias + linear + fm + sequence_match + deep


def predict(model, features, history, device, batch_size):
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch_features = torch.from_numpy(features[start : start + batch_size]).to(device)
            batch_history = torch.from_numpy(history[start : start + batch_size]).to(device)
            values.append(model(batch_features, batch_history).cpu().numpy())
    return np.concatenate(values)


def append_log(path, record):
    if path is None:
        return
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run log: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    rows = load_rows(args.data_dir)
    add_history(rows, args.history_length)
    encoded, field_dimensions, author_padding = encode(rows, args.history_length)
    train_x, train_history, train_y, _ = encoded["train"]
    valid_x, valid_history, valid_y, valid_users = encoded["valid"]
    model = SequenceDeepFM(field_dimensions, author_padding, args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best_state = None
    best_event = None
    best_primary = -np.inf
    stalled_epochs = 0
    trajectory = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_y))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            features = torch.from_numpy(train_x[indices]).to(device)
            history = torch.from_numpy(train_history[indices]).to(device)
            labels = torch.from_numpy(train_y[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(model(features, history), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = predict(model, valid_x, valid_history, device, args.batch_size)
        metrics = {name: float(value) for name, value in evaluate(valid_users, valid_y, scores).items()}
        event = {"epoch": epoch, "train_loss": round(float(np.mean(losses)), 7), "metrics": metrics}
        trajectory.append(event)
        print(
            f"epoch {epoch:2d} | loss {event['train_loss']:.4f} | GAUC {metrics['GAUC']:.4f} | "
            f"nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}"
        )
        if metrics["primary"] > best_primary + 1e-5:
            best_primary = metrics["primary"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_event = event
            stalled_epochs = 0
        else:
            stalled_epochs += 1
            if stalled_epochs >= args.patience:
                print(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    selected_scores = predict(model, valid_x, valid_history, device, args.batch_size)
    if args.validation_scores_out:
        score_path = Path(args.validation_scores_out)
        if score_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing validation scores: {score_path}")
        score_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(score_path, selected_scores)
    record = {
        "phase": "causal_sequence_deepfm",
        "hypothesis": HYPOTHESIS,
        "fields": FIELDS,
        "history_length": args.history_length,
        "best": best_event,
        "trajectory": trajectory,
        "error_or_recovery": None,
        "manual_interventions": 0,
        "test_data_used": False,
        "device": str(device),
    }
    append_log(Path(args.run_log) if args.run_log else None, record)
    print("\nBest validation result")
    print(json.dumps(best_event, indent=2, sort_keys=True))
    return record


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history_length", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--run_log", default=None)
    parser.add_argument("--validation_scores_out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    started = time.time()
    run(parse_args())
    print(f"elapsed_seconds={time.time() - started:.1f}")
