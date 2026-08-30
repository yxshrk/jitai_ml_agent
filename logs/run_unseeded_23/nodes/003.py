import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def scalar_metric(metrics, names):
    for name in names:
        if name in metrics:
            return float(metrics[name])
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        key = name.lower()
        if key in lowered:
            return float(lowered[key])
    raise KeyError(str(names))


def load_npz(data_dir):
    train_file = np.load(Path(data_dir) / "train.npz", allow_pickle=True)
    val_file = np.load(Path(data_dir) / "val.npz", allow_pickle=True)

    x_train = np.asarray(train_file["X"], dtype=np.int64)
    y_train = np.asarray(train_file["y"], dtype=np.float32).reshape(-1)
    train_user = np.asarray(train_file["user"]).reshape(-1)
    play_train = np.asarray(train_file["play_time_ms"], dtype=np.float32).reshape(-1)
    duration_train = np.asarray(train_file["duration_ms"], dtype=np.float32).reshape(-1)

    x_val = np.asarray(val_file["X"], dtype=np.int64)
    y_val = np.asarray(val_file["y"], dtype=np.float32).reshape(-1)
    val_user = np.asarray(val_file["user"]).reshape(-1)

    if "field_dims" in train_file:
        field_dims = np.asarray(train_file["field_dims"], dtype=np.int64).reshape(-1)
    else:
        offsets = np.min(x_train, axis=0)
        field_dims = np.empty(x_train.shape[1], dtype=np.int64)
        for j in range(x_train.shape[1]):
            upper = int(max(np.max(x_train[:, j]), np.max(x_val[:, j])))
            if j + 1 < x_train.shape[1]:
                field_dims[j] = int(np.min(x_train[:, j + 1])) - int(offsets[j])
            else:
                field_dims[j] = upper - int(offsets[j]) + 1

    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_codes = x_val[:, 1] - offsets[1]
    row_ids = np.arange(x_val.shape[0], dtype=np.int64)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_user": train_user,
        "play_train": play_train,
        "duration_train": duration_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_user": val_user,
        "val_video": video_codes,
        "row_ids": row_ids,
        "field_dims": field_dims,
        "npz": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for position, row in enumerate(reader):
            item = {
                "row_id": row.get("row_id", str(position)),
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", row["video_id"]),
                "tab": row.get("tab", "0"),
                "duration": float(row.get("duration_ms", "0") or 0.0),
                "label": float(row["long_view"]),
            }
            if training:
                item["play"] = float(row.get("play_time_ms", "0") or 0.0)
            rows.append(item)
    return rows


def make_mapping(values):
    unique = sorted(set(values))
    return {value: i + 1 for i, value in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)

    user_map = make_mapping([r["user"] for r in train_rows])
    video_map = make_mapping([r["video"] for r in train_rows])
    author_map = make_mapping([r["author"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])

    train_duration = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        11,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right")) + 1
            x[i, 0] = user_map.get(row["user"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = bucket + offsets[4]
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    return {
        "x_train": x_train,
        "y_train": np.asarray([r["label"] for r in train_rows], dtype=np.float32),
        "train_user": np.asarray([r["user"] for r in train_rows], dtype=object),
        "play_train": np.asarray([r["play"] for r in train_rows], dtype=np.float32),
        "duration_train": np.asarray([r["duration"] for r in train_rows], dtype=np.float32),
        "x_val": x_val,
        "y_val": np.asarray([r["label"] for r in val_rows], dtype=np.float32),
        "val_user": np.asarray([r["user"] for r in val_rows], dtype=object),
        "val_video": np.asarray([r["video"] for r in val_rows], dtype=object),
        "row_ids": np.asarray([r["row_id"] for r in val_rows], dtype=object),
        "field_dims": field_dims,
        "npz": False,
    }


def build_pairs(users, labels, rng, max_pairs=500000):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    pos_parts = []
    neg_parts = []
    total = 0
    for start, end in zip(starts, ends):
        group = order[start:end]
        positive = group[labels[group] > 0.5]
        negative = group[labels[group] <= 0.5]
        if positive.size == 0 or negative.size == 0:
            continue
        chosen_neg = negative[rng.integers(0, negative.size, size=positive.size)]
        pos_parts.append(positive)
        neg_parts.append(chosen_neg)
        total += positive.size
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    positive = np.concatenate(pos_parts).astype(np.int64, copy=False)
    negative = np.concatenate(neg_parts).astype(np.int64, copy=False)
    if positive.size > max_pairs:
        keep = rng.choice(positive.size, size=max_pairs, replace=False)
        positive = positive[keep]
        negative = negative[keep]
    return positive, negative


class DCNCensored(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dim=128, dropout=0.25):
        super().__init__()
        total_features = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total_features, embedding_dim)
        input_dim = len(field_dims) * embedding_dim
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim)) for _ in range(2)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(2)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.main_head = nn.Linear(input_dim + hidden_dim // 2, 1)
        self.watch_head = nn.Linear(input_dim + hidden_dim // 2, 1)
        self.embedding_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.main_head.weight)
        nn.init.zeros_(self.main_head.bias)
        nn.init.xavier_uniform_(self.watch_head.weight)
        nn.init.zeros_(self.watch_head.bias)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        x0 = self.embedding_dropout(x0)
        cross = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            projection = torch.sum(cross * weight, dim=1, keepdim=True)
            cross = x0 * projection + bias + cross
        deep = self.mlp(x0)
        shared = torch.cat([cross, deep], dim=1)
        main_logit = self.main_head(shared).squeeze(1)
        watch_ratio = F.softplus(self.watch_head(shared).squeeze(1))
        return main_logit, watch_ratio


def censored_watch_loss(prediction, play_time, duration):
    valid_duration = torch.clamp(duration, min=1.0)
    observed_ratio = torch.clamp(play_time / valid_duration, min=0.0, max=1.0)
    completed = (play_time >= duration) & (duration > 0.0)
    exact = ~completed
    losses = torch.zeros_like(prediction)
    if torch.any(exact):
        losses[exact] = F.smooth_l1_loss(
            prediction[exact], observed_ratio[exact], reduction="none", beta=0.1
        )
    if torch.any(completed):
        losses[completed] = torch.square(F.relu(1.0 - prediction[completed]))
    return losses.mean()


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (Path(args.data_dir) / "train.npz").exists() and (Path(args.data_dir) / "val.npz").exists():
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate

    rng = np.random.default_rng(args.seed)
    pair_pos, pair_neg = build_pairs(data["train_user"], data["y_train"], rng)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNCensored(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.65)

    epochs = 10
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    point_batch = 8192
    pair_batch = 4096
    best_gauc = -float("inf")
    best_state = None
    patience = 3
    stale = 0

    for _ in range(epochs):
        model.train()
        point_order = rng.permutation(data["x_train"].shape[0])
        for start in range(0, point_order.size, point_batch):
            idx = point_order[start:start + point_batch]
            xb = torch.as_tensor(data["x_train"][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["y_train"][idx], dtype=torch.float32, device=device)
            play = torch.as_tensor(data["play_train"][idx], dtype=torch.float32, device=device)
            duration = torch.as_tensor(data["duration_train"][idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits, watch_prediction = model(xb)
            primary_loss = F.binary_cross_entropy_with_logits(logits, yb)
            watch_loss = censored_watch_loss(watch_prediction, play, duration)
            loss = 0.5 * primary_loss + 0.08 * watch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if pair_pos.size:
            pair_order = rng.permutation(pair_pos.size)
            for start in range(0, pair_order.size, pair_batch):
                pair_idx = pair_order[start:start + pair_batch]
                pos_idx = pair_pos[pair_idx]
                neg_idx = pair_neg[pair_idx]
                xp = torch.as_tensor(data["x_train"][pos_idx], dtype=torch.long, device=device)
                xn = torch.as_tensor(data["x_train"][neg_idx], dtype=torch.long, device=device)

                optimizer.zero_grad(set_to_none=True)
                pos_score, _ = model(xp)
                neg_score, _ = model(xn)
                pair_loss = -F.logsigmoid(pos_score - neg_score).mean()
                loss = 0.5 * pair_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        scheduler.step()
        validation_scores = predict(model, data["x_val"], device)
        validation_metrics = evaluate(data["val_user"], data["y_val"], validation_scores)
        current_gauc = scalar_metric(validation_metrics, ["GAUC", "gauc"])
        if current_gauc > best_gauc + 1e-8:
            best_gauc = current_gauc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, data["x_val"], device)
    metrics = evaluate(data["val_user"], data["y_val"], scores)
    output_metrics = {
        "gauc": scalar_metric(metrics, ["GAUC", "gauc"]),
        "ndcg5": scalar_metric(metrics, ["nDCG@5", "ndcg5", "ndcg@5"]),
        "primary": scalar_metric(metrics, ["primary"]),
    }

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user_id, video_id, score in zip(
            data["row_ids"], data["val_user"], data["val_video"], scores
        ):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
