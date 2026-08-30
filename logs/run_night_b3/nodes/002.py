import argparse
import csv
import json
import os
import random
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as z:
        x_train = np.asarray(z["X"], dtype=np.int64)
        y_train = np.asarray(z["y"], dtype=np.float32)
        train_users = np.asarray(z["user"])
        field_dims = np.asarray(z["field_dims"], dtype=np.int64)
    with np.load(val_path, allow_pickle=False) as z:
        x_val = np.asarray(z["X"], dtype=np.int64)
        y_val = np.asarray(z["y"], dtype=np.float32)
        val_users = np.asarray(z["user"])
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_videos = x_val[:, 1] - offsets[1]
    return x_train, y_train, train_users, x_val, y_val, val_users, val_users, val_videos, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__unknown_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    keys = ("user_id", "video_id", "author_id", "tab")
    maps = []
    for key in keys:
        values = sorted({r[key] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})
    field_dims = np.asarray([len(m) + 1 for m in maps] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for j, key in enumerate(keys):
            mapping = maps[j]
            x[:, j] = np.fromiter((mapping.get(r[key], 0) for r in rows), dtype=np.int64, count=len(rows))
        raw_durations = np.asarray([r["duration_ms"] for r in rows], dtype=np.float64)
        x[:, 4] = np.searchsorted(quantiles, raw_durations, side="right")
        x += offsets[None, :]
        y = np.asarray([r["long_view"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user_id"] for r in rows])
        videos = np.asarray([r["video_id"] for r in rows])
        return x, y, users, videos

    x_train, y_train, train_users, _ = encode(train_rows)
    x_val, y_val, val_users, val_videos = encode(val_rows)
    return x_train, y_train, train_users, x_val, y_val, val_users, val_users, val_videos, field_dims, False


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x + x0 * scale + self.bias


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.30):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear_embedding.weight)
        self.input_dropout = nn.Dropout(dropout)
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim), CrossLayer(input_dim)])
        self.cross_dropout = nn.Dropout(dropout)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + 128, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        embeddings = self.embedding(x)
        flat = self.input_dropout(embeddings.flatten(1))
        cross = flat
        for layer in self.cross_layers:
            cross = self.cross_dropout(layer(flat, cross))
        deep = self.deep(flat)
        interaction = self.output(torch.cat([cross, deep], dim=1)).squeeze(1)
        linear_values = self.linear_embedding(x).squeeze(-1)
        linear = linear_values.sum(dim=1)
        return interaction + linear + self.bias, embeddings, linear_values


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.split(order, boundaries)
    pos_parts = []
    neg_parts = []
    for group in groups:
        group_labels = labels[group] >= 0.5
        pos = group[group_labels]
        neg = group[~group_labels]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _, _ = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(x_train, y_train, train_users, x_val, y_val, val_users, field_dims, evaluator, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(field_dims, embed_dim=16, dropout=0.30).to(device)
    embedding_params = list(model.embedding.parameters()) + list(model.linear_embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=0, threshold=2e-4, min_lr=2e-5
    )
    pair_pos, pair_neg = make_pairs(train_users, y_train, seed + 17)
    rng = np.random.default_rng(seed + 31)
    batch_size = 8192 if device.type == "cuda" else 4096
    max_epochs = 20
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    best_gauc = -float("inf")
    best_state = clone_state(model)
    stale = 0

    for _ in range(max_epochs):
        model.train()
        order = rng.permutation(len(x_train))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        for step, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            logits, emb, linear_values = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)
            row_l2 = emb.square().sum(dim=2).mean() + linear_values.square().mean()

            if len(pair_order):
                pair_start = (step * batch_size) % len(pair_order)
                pair_idx = pair_order[pair_start:pair_start + batch_size]
                if len(pair_idx) < min(batch_size, len(pair_order)):
                    needed = min(batch_size, len(pair_order)) - len(pair_idx)
                    pair_idx = np.concatenate([pair_idx, pair_order[:needed]])
                pos_x = torch.as_tensor(x_train[pair_pos[pair_idx]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[pair_neg[pair_idx]], dtype=torch.long, device=device)
                pos_logits, pos_emb, pos_linear = model(pos_x)
                neg_logits, neg_emb, neg_linear = model(neg_x)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                row_l2 = row_l2 + 0.5 * (
                    pos_emb.square().sum(dim=2).mean()
                    + neg_emb.square().sum(dim=2).mean()
                    + pos_linear.square().mean()
                    + neg_linear.square().mean()
                )
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-4 * row_l2
            else:
                loss = point_loss + 1e-4 * row_l2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, x_val, device, batch_size * 2)
        metrics = evaluator(val_users, y_val, val_scores)
        gauc = float(metrics["GAUC"])
        scheduler.step(gauc)
        if gauc > best_gauc + 1e-5:
            best_gauc = gauc
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break

    model.load_state_dict(best_state)
    return model, device, batch_size


def write_outputs(out_dir, row_users, row_videos, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(row_users, row_videos, scores)):
            writer.writerow([i, user, video, format(float(score), ".10g")])
    payload = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast_path:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate

    x_train, y_train, train_users, x_val, y_val, val_users, row_users, row_videos, field_dims, _ = data
    model, device, batch_size = train_model(
        x_train, y_train, train_users, x_val, y_val, val_users, field_dims, evaluate, args.seed
    )
    scores = predict(model, x_val, device, batch_size * 2)
    metrics = evaluate(val_users, y_val, scores)
    write_outputs(args.out_dir, row_users, row_videos, scores, metrics)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        main()
