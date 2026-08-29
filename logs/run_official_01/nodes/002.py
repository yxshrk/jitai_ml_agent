import argparse
import copy
import csv
import json
import os
import random

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


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None

    with np.load(train_path, allow_pickle=False) as tr:
        x_train = np.asarray(tr["X"], dtype=np.int64)
        y_train = np.asarray(tr["y"], dtype=np.float32)
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    with np.load(val_path, allow_pickle=False) as va:
        x_val = np.asarray(va["X"], dtype=np.int64)
        y_val = np.asarray(va["y"], dtype=np.float32)
        val_users = np.asarray(va["user"])

    video_offset = int(field_dims[0])
    val_videos = x_val[:, 1] - video_offset
    total_features = max(int(field_dims.sum()), int(x_train.max()) + 1, int(x_val.max()) + 1)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_users": val_users,
        "val_videos": val_videos,
        "pair_users": x_train[:, 0],
        "total_features": total_features,
        "npz": True,
    }


def read_csv_columns(path, training):
    users = []
    videos = []
    tabs = []
    durations = []
    labels = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            try:
                durations.append(float(row["duration_ms"]))
            except (ValueError, TypeError):
                durations.append(0.0)
            labels.append(float(row["long_view"]))
    return users, videos, tabs, np.asarray(durations, dtype=np.float64), np.asarray(labels, dtype=np.float32)


def encode_field(train_values, val_values, offset):
    mapping = {}
    next_code = 1
    train_codes = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        code = mapping.get(value)
        if code is None:
            code = next_code
            mapping[value] = code
            next_code += 1
        train_codes[i] = offset + code
    val_codes = np.empty(len(val_values), dtype=np.int64)
    for i, value in enumerate(val_values):
        val_codes[i] = offset + mapping.get(value, 0)
    dim = next_code
    return train_codes, val_codes, offset + dim


def load_csv_data(data_dir):
    tr = read_csv_columns(os.path.join(data_dir, "train.csv"), True)
    va = read_csv_columns(os.path.join(data_dir, "val.csv"), False)
    tr_users, tr_videos, tr_tabs, tr_duration, y_train = tr
    va_users, va_videos, va_tabs, va_duration, y_val = va

    finite_duration = tr_duration[np.isfinite(tr_duration)]
    if finite_duration.size:
        edges = np.unique(np.quantile(finite_duration, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)
    tr_buckets = np.searchsorted(edges, np.nan_to_num(tr_duration), side="right").astype(str).tolist()
    va_buckets = np.searchsorted(edges, np.nan_to_num(va_duration), side="right").astype(str).tolist()

    offset = 0
    train_columns = []
    val_columns = []
    for train_values, val_values in (
        (tr_users, va_users),
        (tr_videos, va_videos),
        (tr_videos, va_videos),
        (tr_tabs, va_tabs),
        (tr_buckets, va_buckets),
    ):
        train_codes, val_codes, offset = encode_field(train_values, val_values, offset)
        train_columns.append(train_codes)
        val_columns.append(val_codes)

    x_train = np.stack(train_columns, axis=1)
    x_val = np.stack(val_columns, axis=1)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "val_users": np.asarray(va_users),
        "val_videos": np.asarray(va_videos),
        "pair_users": x_train[:, 0],
        "total_features": offset,
        "npz": False,
    }


def build_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts = []
    negative_parts = []
    for j in range(len(boundaries) - 1):
        indices = order[boundaries[j]:boundaries[j + 1]]
        positive = indices[labels[indices] > 0.5]
        negative = indices[labels[indices] <= 0.5]
        if positive.size and negative.size:
            positive_parts.append(positive)
            negative_parts.append(rng.choice(negative, size=positive.size, replace=True))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class RegularizedDCN(nn.Module):
    def __init__(self, total_features, fields=5, embedding_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.linear_embedding = nn.Embedding(total_features, 1)
        input_dim = fields * embedding_dim
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.deep = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.cross_output = nn.Linear(input_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for weight in self.cross_weights:
            nn.init.normal_(weight, std=0.01)

    def forward(self, x, return_embeddings=False):
        embedded = self.embedding(x)
        x0 = embedded.flatten(1)
        crossed = x0
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            scale = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = x0 * scale + bias + crossed
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        logits = first_order + self.cross_output(crossed).squeeze(-1) + self.deep(x0).squeeze(-1) + self.global_bias
        if return_embeddings:
            return logits, embedded
        return logits


def predict(model, x, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(outputs).astype(np.float64)


def metric_value(metrics, name):
    candidates = {
        "gauc": ("GAUC", "gauc"),
        "ndcg5": ("nDCG@5", "ndcg5", "NDCG@5"),
        "primary": ("primary", "PRIMARY"),
    }[name]
    for key in candidates:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(name)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_npz_data(args.data_dir)
    if data is None:
        data = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate
    else:
        from data.official.evaluate import evaluate

    x_train = np.ascontiguousarray(data["x_train"], dtype=np.int64)
    y_train = np.ascontiguousarray(data["y_train"], dtype=np.float32)
    x_val = np.ascontiguousarray(data["x_val"], dtype=np.int64)
    y_val = np.ascontiguousarray(data["y_val"], dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    pair_positive, pair_negative = build_pairs(np.asarray(data["pair_users"]), y_train, rng)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(data["total_features"]).to(device)
    embedding_parameters = list(model.embedding.parameters()) + list(model.linear_embedding.parameters())
    embedding_ids = {id(p) for p in embedding_parameters}
    dense_parameters = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_parameters, "weight_decay": 0.0},
            {"params": dense_parameters, "weight_decay": 1e-3},
        ],
        lr=2e-3,
    )

    epochs = 20
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(0, int(smoke_epochs)))
    batch_size = 16384
    row_l2_weight = 1e-4
    best_gauc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())

    train_order = np.arange(len(x_train), dtype=np.int64)
    pair_order = np.arange(len(pair_positive), dtype=np.int64)

    for _ in range(epochs):
        rng.shuffle(train_order)
        if pair_order.size:
            rng.shuffle(pair_order)
        model.train()
        pair_cursor = 0

        for start in range(0, len(train_order), batch_size):
            batch_indices = train_order[start:start + batch_size]
            xb = torch.from_numpy(x_train[batch_indices]).to(device)
            yb = torch.from_numpy(y_train[batch_indices]).to(device)
            logits, accessed_embeddings = model(xb, return_embeddings=True)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if pair_order.size:
                needed = len(batch_indices)
                chosen_parts = []
                while needed > 0:
                    available = len(pair_order) - pair_cursor
                    take = min(needed, available)
                    chosen_parts.append(pair_order[pair_cursor:pair_cursor + take])
                    pair_cursor += take
                    needed -= take
                    if pair_cursor == len(pair_order):
                        pair_cursor = 0
                chosen = np.concatenate(chosen_parts)
                pos_x = torch.from_numpy(x_train[pair_positive[chosen]]).to(device)
                neg_x = torch.from_numpy(x_train[pair_negative[chosen]]).to(device)
                pos_logits, pos_embeddings = model(pos_x, return_embeddings=True)
                neg_logits, neg_embeddings = model(neg_x, return_embeddings=True)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                row_l2 = (
                    accessed_embeddings.square().sum(dim=-1).mean()
                    + pos_embeddings.square().sum(dim=-1).mean()
                    + neg_embeddings.square().sum(dim=-1).mean()
                ) / 3.0
                task_loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                row_l2 = accessed_embeddings.square().sum(dim=-1).mean()
                task_loss = point_loss

            loss = task_loss + row_l2_weight * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_scores = predict(model, x_val, device)
        epoch_metrics = evaluate(data["val_users"], y_val, val_scores)
        epoch_gauc = metric_value(epoch_metrics, "gauc")
        if epoch_gauc > best_gauc:
            best_gauc = epoch_gauc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for group in optimizer.param_groups:
            group["lr"] *= 0.5

    model.load_state_dict(best_state)
    model.to(device)
    scores = predict(model, x_val, device)
    final_metrics = evaluate(data["val_users"], y_val, scores)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_users"], data["val_videos"], scores)):
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, format(float(score), ".10g")])

    output_metrics = {
        "gauc": metric_value(final_metrics, "gauc"),
        "ndcg5": metric_value(final_metrics, "ndcg5"),
        "primary": metric_value(final_metrics, "primary"),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(output_metrics, f)


if __name__ == "__main__":
    main()
