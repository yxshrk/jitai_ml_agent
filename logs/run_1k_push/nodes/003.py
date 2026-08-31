import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_ordinal(value):
    s = str(value).strip()
    try:
        if "." in s:
            s = str(int(float(s)))
        s = s.replace("-", "")
        return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return 0


def load_fast(data_dir):
    train = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    Xtr = np.asarray(train["X"], dtype=np.int64)
    ytr = np.asarray(train["y"], dtype=np.float32)
    Xva = np.asarray(val["X"], dtype=np.int64)
    yva = np.asarray(val["y"], dtype=np.float32)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    dates = np.asarray(train["date"]) if "date" in train.files else np.zeros(len(ytr), dtype=np.int64)
    return {
        "Xtr": Xtr,
        "ytr": ytr,
        "utr": np.asarray(train["user"]),
        "dates": dates,
        "Xva": Xva,
        "yva": yva,
        "uva": np.asarray(val["user"]),
        "video_out": Xva[:, 1] - offsets[1],
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row.get("tab", "0"),
                "duration_ms": float(row.get("duration_ms", 0) or 0),
                "date": row.get("date", "0"),
                "long_view": float(row["long_view"]),
            })
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv")
    val_rows = read_csv_rows(Path(data_dir) / "val.csv")
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    if len(durations):
        boundaries = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    else:
        boundaries = np.zeros(9, dtype=np.float64)

    def raw_fields(rows):
        result = []
        for row in rows:
            bucket = int(np.searchsorted(boundaries, row["duration_ms"], side="right"))
            result.append([
                row["user_id"], row["video_id"], row["author_id"], row["tab"], str(bucket)
            ])
        return result

    raw_train = raw_fields(train_rows)
    raw_val = raw_fields(val_rows)
    mappings = []
    dims = []
    for field in range(5):
        values = sorted({row[field] for row in raw_train})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        mappings.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(raw):
        X = np.empty((len(raw), 5), dtype=np.int64)
        for i, row in enumerate(raw):
            for field in range(5):
                X[i, field] = mappings[field].get(row[field], 0) + offsets[field]
        return X

    return {
        "Xtr": encode(raw_train),
        "ytr": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "utr": np.asarray([row["user_id"] for row in train_rows]),
        "dates": np.asarray([row["date"] for row in train_rows]),
        "Xva": encode(raw_val),
        "yva": np.asarray([row["long_view"] for row in val_rows], dtype=np.float32),
        "uva": np.asarray([row["user_id"] for row in val_rows]),
        "video_out": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def make_recency_weights(dates, half_life_days):
    ordinals = np.asarray([date_ordinal(value) for value in dates], dtype=np.int64)
    valid = ordinals > 0
    if not np.any(valid):
        return np.ones(len(dates), dtype=np.float32)
    newest = int(np.max(ordinals[valid]))
    ages = np.maximum(0, newest - ordinals)
    weights = np.exp(-math.log(2.0) * ages / half_life_days).astype(np.float32)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    positive_parts = []
    negative_parts = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_users[end] == sorted_users[start]:
            end += 1
        indices = order[start:end]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if len(positives) and len(negatives):
            sampled_negatives = negatives[rng.integers(0, len(negatives), size=len(positives))]
            positive_parts.append(positives.astype(np.int64, copy=False))
            negative_parts.append(sampled_negatives.astype(np.int64, copy=False))
        start = end
    if not positive_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class RankModel(nn.Module):
    def __init__(self, n_vocab, n_fields, embedding_dim=16, dropout=0.21):
        super().__init__()
        self.n_fields = n_fields
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.linear = nn.Embedding(n_vocab, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.embedding_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        width = n_fields * embedding_dim
        self.cross_scalar = nn.Linear(width, 1, bias=False)
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.deep = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(width + 32, 1)

    def forward(self, x):
        embeddings = self.embedding_dropout(self.embedding(x))
        linear = self.linear(x).sum(dim=1).squeeze(-1) + self.bias
        x0 = embeddings.reshape(embeddings.shape[0], -1)
        cross = x0 * self.cross_scalar(x0) + x0 + self.cross_bias
        deep = self.deep(x0)
        return linear + self.head(torch.cat([cross, deep], dim=1)).squeeze(-1)


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def score_model(model, X, batch_size, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
            parts.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_curriculum(data, seed, total_epochs, device, evaluator, pair_pos, pair_neg):
    seed_all(seed)
    Xtr = data["Xtr"]
    ytr = data["ytr"]
    n_vocab = max(
        int(np.sum(data["field_dims"])),
        int(Xtr.max()) + 1,
        int(data["Xva"].max()) + 1,
    )
    model = RankModel(n_vocab, Xtr.shape[1], embedding_dim=16, dropout=0.21).to(device)
    base_lr = 0.00168
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=3.7e-5)
    batch_size = 16384 if device.type == "cuda" else 8192
    rng = np.random.default_rng(seed + 991)
    weights_by_phase = {
        "uniform": data["uniform"],
        "recency-7d": data["recency7"],
        "recency-3.5d": data["recency35"],
    }
    nominal_schedule = [
        {"name": "uniform", "lr_factor": 1.0},
        {"name": "recency-7d", "lr_factor": 1.0},
        {"name": "recency-7d", "lr_factor": 1.0},
        {"name": "recency-3.5d", "lr_factor": 0.3},
    ]
    segment_count = min(len(nominal_schedule), max(1, int(total_epochs) * 2))
    schedule = nominal_schedule[:segment_count]
    best_primary = -float("inf")
    best_state = None
    best_predictions = None
    best_metrics = None
    best_checkpoint = 0.0
    trajectory = []
    current_order = None
    pair_order = np.arange(len(pair_pos), dtype=np.int64)
    pair_cursor = 0

    for segment_id, phase in enumerate(schedule):
        if segment_id % 2 == 0 or current_order is None:
            current_order = rng.permutation(len(Xtr))
        midpoint = (len(current_order) + 1) // 2
        if segment_id % 2 == 0:
            segment_indices = current_order[:midpoint]
        else:
            segment_indices = current_order[midpoint:]
        if len(segment_indices) == 0:
            segment_indices = current_order
        if len(pair_order):
            pair_order = rng.permutation(len(pair_pos))
            pair_cursor = 0
        for group in optimizer.param_groups:
            group["lr"] = base_lr * float(phase["lr_factor"])
        recency = weights_by_phase[phase["name"]]
        model.train()
        train_loss_sum = 0.0
        train_examples = 0

        for start in range(0, len(segment_indices), batch_size):
            indices = segment_indices[start:start + batch_size]
            xb = torch.as_tensor(Xtr[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[indices], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()

            if len(pair_pos):
                need = len(indices)
                if pair_cursor + need > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + need]
                pair_cursor += len(selected)
                if len(selected):
                    positive_indices = pair_pos[selected]
                    negative_indices = pair_neg[selected]
                    xp = torch.as_tensor(Xtr[positive_indices], dtype=torch.long, device=device)
                    xn = torch.as_tensor(Xtr[negative_indices], dtype=torch.long, device=device)
                    pair_weights = 0.5 * (recency[positive_indices] + recency[negative_indices])
                    pair_weights_t = torch.as_tensor(pair_weights, dtype=torch.float32, device=device)
                    pair_loss = (
                        F.softplus(-(model(xp) - model(xn))) * pair_weights_t
                    ).mean()
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(indices)
            train_examples += len(indices)

        predictions = score_model(model, data["Xva"], batch_size, device)
        metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))
        checkpoint = 0.5 * float(segment_id + 1)
        trajectory.append({
            "checkpoint": checkpoint,
            "phase": phase["name"],
            "lr": base_lr * float(phase["lr_factor"]),
            "train_loss": train_loss_sum / max(1, train_examples),
            **metrics,
        })
        if metrics["primary"] > best_primary:
            best_primary = metrics["primary"]
            best_metrics = metrics
            best_checkpoint = checkpoint
            best_predictions = predictions.copy()
            best_state = copy.deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_predictions is None:
        best_predictions = score_model(model, data["Xva"], batch_size, device)
        best_metrics = normalize_metrics(evaluator(data["uva"], data["yva"], best_predictions))
    result = {
        "predictions": best_predictions,
        "metrics": best_metrics,
        "best_checkpoint": best_checkpoint,
        "trajectory": trajectory,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    fast = (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists()
    data = load_fast(data_dir) if fast else load_csv(data_dir)
    data["uniform"] = np.ones(len(data["ytr"]), dtype=np.float32)
    data["recency7"] = make_recency_weights(data["dates"], 7.0)
    data["recency35"] = make_recency_weights(data["dates"], 3.5)
    evaluator = get_evaluator(fast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_pos, pair_neg = make_pairs(data["utr"], data["ytr"], args.seed + 17)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    if smoke_value is None:
        total_epochs = 2
    else:
        total_epochs = max(1, min(2, int(smoke_value)))

    result = train_curriculum(
        data=data,
        seed=args.seed,
        total_epochs=total_epochs,
        device=device,
        evaluator=evaluator,
        pair_pos=pair_pos,
        pair_neg=pair_neg,
    )
    predictions = result["predictions"]
    final_metrics = normalize_metrics(evaluator(data["uva"], data["yva"], predictions))

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(
            zip(data["uva"], data["video_out"], predictions)
        ):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])

    config = {
        "architecture": "dcn-lite",
        "embedding_dim": 16,
        "loss": "0.5-logloss+0.5-bpr",
        "dropout": 0.21,
        "weight_decay": 3.7e-5,
        "base_lr": 0.00168,
        "curriculum": [
            {"interval": "0.0-0.5", "weighting": "uniform", "lr_factor": 1.0},
            {"interval": "0.5-1.5", "weighting": "recency-7d", "lr_factor": 1.0},
            {"interval": "1.5-2.0", "weighting": "recency-3.5d", "lr_factor": 0.3},
        ],
        "checkpoint_interval_epochs": 0.5,
        "optimizer_reset_between_phases": False,
    }
    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": config,
        "history": [{
            "phase": "final",
            "seed": args.seed,
            "config": config,
            "best_checkpoint": result["best_checkpoint"],
            "metrics": result["metrics"],
            "trajectory": result["trajectory"],
        }],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)


if __name__ == "__main__":
    main()
