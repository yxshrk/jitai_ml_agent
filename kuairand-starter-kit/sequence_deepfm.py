"""Validation-only neural sequence ranker for KuaiRand-Pure.

The model combines FM interactions, a small deep tower, and mean-pooled recent
author history. Context is strictly causal: a record never sees its own
outcome. The default validation history is frozen at train end; an explicit
metadata-only rolling mode can additionally use earlier impressions.
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
AUXILIARY_TASKS = ("is_click", "is_profile_enter", "is_like", "is_follow")
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
                        "auxiliary": tuple(1 if source_row[name] != "0" else 0 for name in AUXILIARY_TASKS),
                    }
                )
    return rows


def add_history(rows, history_length, validation_history_mode, history_source):
    if history_source not in ("all_exposures", "positive_long_view"):
        raise ValueError(f"Unknown history source: {history_source}")
    if history_source == "positive_long_view" and validation_history_mode == "rolling_metadata":
        raise ValueError("positive_long_view history cannot consume validation metadata without labels")
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
            if history_source == "all_exposures" or row["label"]:
                histories[row["user_id"]].append(row["author_id"])
        start = end
    if validation_history_mode == "frozen_train":
        for row in rows["valid"]:
            row["history_authors"] = tuple(histories[row["user_id"]])
        return
    if validation_history_mode != "rolling_metadata":
        raise ValueError(f"Unknown validation history mode: {validation_history_mode}")
    # The feature history contains only authors from strictly earlier
    # impressions. Outcomes are never read, and rows at one timestamp are
    # assigned their histories before any of their authors are appended.
    order = sorted(range(len(rows["valid"])), key=lambda index: rows["valid"][index]["timestamp"])
    start = 0
    while start < len(order):
        end = start + 1
        timestamp = rows["valid"][order[start]]["timestamp"]
        while end < len(order) and rows["valid"][order[end]]["timestamp"] == timestamp:
            end += 1
        for order_index in range(start, end):
            row = rows["valid"][order[order_index]]
            row["history_authors"] = tuple(histories[row["user_id"]])
        for order_index in range(start, end):
            row = rows["valid"][order[order_index]]
            histories[row["user_id"]].append(row["author_id"])
        start = end


def build_pair_index(users, labels):
    """Build train-only positive/negative pairs from the same user."""
    negatives = defaultdict(list)
    positives = []
    for index, (user_id, label) in enumerate(zip(users, labels)):
        if label:
            positives.append((index, user_id))
        else:
            negatives[user_id].append(index)
    usable = [(index, user_id) for index, user_id in positives if negatives[user_id]]
    return np.asarray([index for index, _ in usable], dtype=np.int64), [user_id for _, user_id in usable], negatives


def pair_batches(rng, positive_indices, positive_users, negatives, batch_size):
    order = rng.permutation(len(positive_indices))
    for start in range(0, len(order), batch_size):
        selection = order[start : start + batch_size]
        positives = positive_indices[selection]
        negative_batch = np.fromiter(
            (negatives[positive_users[index]][rng.integers(len(negatives[positive_users[index]]))] for index in selection),
            dtype=np.int64, count=len(selection),
        )
        yield positives, negative_batch


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
        auxiliary = np.empty((len(split_rows), len(AUXILIARY_TASKS)), dtype=np.float32)
        users = []
        for row_index, row in enumerate(split_rows):
            for field_index, value in enumerate(raw(row)):
                features[row_index, field_index] = vocabularies[field_index].get(value, unknowns[field_index])
            for history_index, author_id in enumerate(row["history_authors"][-history_length:]):
                history[row_index, history_index] = vocabularies[2].get(author_id, unknowns[2])
            labels[row_index] = row["label"]
            auxiliary[row_index] = row["auxiliary"]
            users.append(row["user_id"])
        encoded[split] = (features, history, labels, auxiliary, users)
    return encoded, dimensions, author_padding


class SequenceDeepFM(nn.Module):
    def __init__(self, field_dimensions, author_padding, embedding_dim, hidden_dim, dropout, history_recency_decay):
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
        self.deep_hidden = nn.Sequential(nn.Linear(input_dimension, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.deep_out = nn.Linear(hidden_dim, 1)
        self.auxiliary_out = nn.Linear(hidden_dim, len(AUXILIARY_TASKS))
        self.bias = nn.Parameter(torch.zeros(()))
        self.history_recency_decay = history_recency_decay
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
        for layer in list(self.deep_hidden) + [self.deep_out, self.auxiliary_out]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features, history_authors, return_auxiliary=False):
        embeddings = [embedding(features[:, index]) for index, embedding in enumerate(self.embeddings)]
        stacked = torch.stack(embeddings, dim=1)
        fm = 0.5 * ((stacked.sum(dim=1) ** 2).sum(dim=1) - (stacked**2).sum(dim=(1, 2)))
        linear = sum(layer(features[:, index]).squeeze(1) for index, layer in enumerate(self.linear))
        history_embeddings = self.embeddings[2](history_authors)
        history_mask = (history_authors != self.embeddings[2].padding_idx).unsqueeze(-1)
        if self.history_recency_decay:
            positions = torch.arange(history_authors.shape[1], device=history_authors.device).view(1, -1, 1)
            lengths = history_mask.sum(dim=1, keepdim=True)
            weights = torch.exp((positions - lengths + 1) * self.history_recency_decay) * history_mask
            history_mean = (history_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        else:
            history_mean = (history_embeddings * history_mask).sum(dim=1) / history_mask.sum(dim=1).clamp_min(1)
        sequence_match = (history_mean * (embeddings[1] + embeddings[2])).sum(dim=1)
        hidden = self.deep_hidden(torch.cat(embeddings + [history_mean], dim=1))
        primary = self.bias + linear + fm + sequence_match + self.deep_out(hidden).squeeze(1)
        return (primary, self.auxiliary_out(hidden)) if return_auxiliary else primary


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
    add_history(rows, args.history_length, args.validation_history_mode, args.history_source)
    encoded, field_dimensions, author_padding = encode(rows, args.history_length)
    train_x, train_history, train_y, train_auxiliary, train_users = encoded["train"]
    valid_x, valid_history, valid_y, _, valid_users = encoded["valid"]
    model = SequenceDeepFM(
        field_dimensions, author_padding, args.embedding_dim, args.hidden_dim, args.dropout, args.history_recency_decay
    ).to(device)
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
            primary_logits, auxiliary_logits = model(features, history, return_auxiliary=True)
            loss = functional.binary_cross_entropy_with_logits(primary_logits, labels)
            if args.auxiliary_weight:
                auxiliary_labels = torch.from_numpy(train_auxiliary[indices]).to(device)
                loss = loss + args.auxiliary_weight * functional.binary_cross_entropy_with_logits(
                    auxiliary_logits, auxiliary_labels
                )
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
    if args.ranking_finetune_epochs:
        # Start ranking refinement from the selected BCE checkpoint, never from
        # a later overfit epoch.  Pair construction uses train labels only.
        model.load_state_dict(best_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ranking_learning_rate, weight_decay=args.weight_decay)
        positive_indices, positive_users, negatives = build_pair_index(train_users, train_y)
        for epoch in range(1, args.ranking_finetune_epochs + 1):
            model.train()
            losses = []
            for positive_indices_batch, negative_indices_batch in pair_batches(
                rng, positive_indices, positive_users, negatives, args.ranking_batch_size
            ):
                optimizer.zero_grad(set_to_none=True)
                pos_scores = model(
                    torch.from_numpy(train_x[positive_indices_batch]).to(device),
                    torch.from_numpy(train_history[positive_indices_batch]).to(device),
                )
                neg_scores = model(
                    torch.from_numpy(train_x[negative_indices_batch]).to(device),
                    torch.from_numpy(train_history[negative_indices_batch]).to(device),
                )
                loss = functional.softplus(-(pos_scores - neg_scores)).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scores = predict(model, valid_x, valid_history, device, args.batch_size)
            metrics = {name: float(value) for name, value in evaluate(valid_users, valid_y, scores).items()}
            event = {"phase": "pairwise_finetune", "epoch": epoch, "train_loss": round(float(np.mean(losses)), 7), "metrics": metrics}
            trajectory.append(event)
            print(f"pairwise {epoch:2d} | loss {event['train_loss']:.4f} | primary {metrics['primary']:.4f}")
            if metrics["primary"] > best_primary + 1e-5:
                best_primary = metrics["primary"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_event = event
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
    parser.add_argument(
        "--validation_history_mode", choices=("frozen_train", "rolling_metadata"), default="frozen_train",
        help="Use only train history, or earlier validation/test impression metadata (never labels).",
    )
    parser.add_argument(
        "--history_source", choices=("all_exposures", "positive_long_view"), default="all_exposures",
        help="Retain all causal train impressions, or only earlier train long-view authors.",
    )
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--history_recency_decay", type=float, default=0.0,
        help="Exponential preference for newer authors in the causal history; zero is uniform pooling.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--ranking_finetune_epochs", type=int, default=0)
    parser.add_argument("--ranking_learning_rate", type=float, default=0.0001)
    parser.add_argument("--ranking_batch_size", type=int, default=8192)
    parser.add_argument(
        "--auxiliary_weight", type=float, default=0.0,
        help="Training-only loss weight for click/profile-entry/like/follow prediction heads.",
    )
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
