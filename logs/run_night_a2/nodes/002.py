import argparse
import csv
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DCNLite(nn.Module):
    def __init__(self, total_dim, n_fields=5, emb_dim=16, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, emb_dim)
        self.linear_embedding = nn.Embedding(total_dim, 1)
        width = n_fields * emb_dim
        self.cross_weight = nn.Parameter(torch.empty(width))
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.cross_out = nn.Linear(width, 1, bias=False)
        self.mlp_fc = nn.Linear(width, 128)
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_out = nn.Linear(128, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.xavier_uniform_(self.mlp_fc.weight)
        nn.init.zeros_(self.mlp_fc.bias)
        nn.init.xavier_uniform_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, x):
        e = self.embedding(x)
        x0 = e.flatten(1)
        cross_scale = torch.sum(x0 * self.cross_weight, dim=1, keepdim=True)
        cross = x0 * cross_scale + self.cross_bias + x0
        hidden = F.relu(self.mlp_fc(x0))
        hidden = self.mlp_dropout(hidden)
        linear = self.linear_embedding(x).sum(dim=1)
        return (linear + self.cross_out(cross) + self.mlp_out(hidden) + self.bias).squeeze(1)


def encode_values(values, mapping, allow_new):
    out = np.empty(len(values), dtype=np.int64)
    unknown = len(mapping)
    for i, value in enumerate(values):
        if value in mapping:
            out[i] = mapping[value]
        elif allow_new:
            mapping[value] = len(mapping)
            out[i] = mapping[value]
        else:
            out[i] = unknown
    return out


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_columns = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")}
    with open(train_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_columns["user_id"].append(row["user_id"])
            train_columns["video_id"].append(row["video_id"])
            train_columns["author_id"].append(row["author_id"] if has_author else row["video_id"])
            train_columns["tab"].append(row["tab"])
            train_columns["duration_ms"].append(float(row["duration_ms"] or 0.0))
            train_columns["long_view"].append(float(row["long_view"]))

    val_columns = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")}
    with open(val_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_columns["user_id"].append(row["user_id"])
            val_columns["video_id"].append(row["video_id"])
            val_columns["author_id"].append(row["author_id"] if has_author else row["video_id"])
            val_columns["tab"].append(row["tab"])
            val_columns["duration_ms"].append(float(row["duration_ms"] or 0.0))
            val_columns["long_view"].append(float(row["long_view"]))

    train_duration = np.asarray(train_columns["duration_ms"], dtype=np.float64)
    val_duration = np.asarray(val_columns["duration_ms"], dtype=np.float64)
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(edges, train_duration, side="right").astype(np.int64)
    val_bucket = np.searchsorted(edges, val_duration, side="right").astype(np.int64)

    train_parts = []
    val_parts = []
    dims = []
    field_names = ("user_id", "video_id", "author_id", "tab")
    for name in field_names:
        mapping = {}
        train_encoded = encode_values(train_columns[name], mapping, True)
        val_encoded = encode_values(val_columns[name], mapping, False)
        dim = len(mapping) + 1
        train_parts.append(train_encoded)
        val_parts.append(val_encoded)
        dims.append(dim)
    train_parts.append(train_bucket)
    val_parts.append(val_bucket)
    dims.append(max(10, int(train_bucket.max(initial=0)) + 1))

    offsets = np.cumsum(np.asarray([0] + dims[:-1], dtype=np.int64))
    train_x = np.stack(train_parts, axis=1) + offsets
    val_x = np.stack(val_parts, axis=1) + offsets
    train_y = np.asarray(train_columns["long_view"], dtype=np.float32)
    val_y = np.asarray(val_columns["long_view"], dtype=np.float32)
    train_users = np.asarray(train_columns["user_id"], dtype=object)
    val_users = np.asarray(val_columns["user_id"], dtype=object)
    val_videos = np.asarray(val_columns["video_id"], dtype=object)
    return train_x, train_y, train_users, val_x, val_y, val_users, val_videos, np.asarray(dims), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=False) as tr:
            train_x = np.asarray(tr["X"], dtype=np.int64)
            train_y = np.asarray(tr["y"], dtype=np.float32)
            train_users = np.asarray(tr["user"])
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
        with np.load(val_npz, allow_pickle=False) as va:
            val_x = np.asarray(va["X"], dtype=np.int64)
            val_y = np.asarray(va["y"], dtype=np.float32)
            val_users = np.asarray(va["user"])
        video_offset = int(field_dims[0])
        val_videos = val_x[:, 1].astype(np.int64) - video_offset
        return train_x, train_y, train_users, val_x, val_y, val_users, val_videos, field_dims, True
    return read_csv_data(data_dir)


def build_pairs(users, labels, rng):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_chunks = []
    negative_chunks = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positive_chunks.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negative_chunks.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positive_chunks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_chunks), np.concatenate(negative_chunks)


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def metric_values(result):
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5"))
    primary = result.get("primary")
    return float(gauc), float(ndcg), float(primary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed = int(args.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    train_x, train_y, train_users, val_x, val_y, val_users, val_videos, field_dims, fast_path = load_data(args.data_dir)
    total_dim = int(np.sum(field_dims))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(total_dim=total_dim, n_fields=5, emb_dim=16, dropout=0.30).to(device)

    embedding_params = [model.embedding.weight, model.linear_embedding.weight]
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epochs = 20
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192 if device.type == "cuda" else 4096
    rng = np.random.default_rng(seed)
    pair_pos, pair_neg = build_pairs(train_users, train_y, rng)

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    best_gauc = -float("inf")
    best_scores = None
    best_metrics = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(train_x))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        point_steps = (len(point_order) + batch_size - 1) // batch_size
        pair_steps = (len(pair_order) + batch_size - 1) // batch_size if len(pair_order) else 0
        steps = max(point_steps, pair_steps)

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            accessed = []

            if step < point_steps:
                idx = point_order[step * batch_size:(step + 1) * batch_size]
                xb = torch.from_numpy(train_x[idx]).to(device=device, dtype=torch.long)
                yb = torch.from_numpy(train_y[idx]).to(device=device, dtype=torch.float32)
                point_loss = F.binary_cross_entropy_with_logits(model(xb), yb)
                losses.append(0.5 * point_loss if pair_steps else point_loss)
                accessed.append(xb.reshape(-1))

            if step < pair_steps:
                selected = pair_order[step * batch_size:(step + 1) * batch_size]
                pos_x = torch.from_numpy(train_x[pair_pos[selected]]).to(device=device, dtype=torch.long)
                neg_x = torch.from_numpy(train_x[pair_neg[selected]]).to(device=device, dtype=torch.long)
                pair_loss = F.softplus(-(model(pos_x) - model(neg_x))).mean()
                losses.append(0.5 * pair_loss)
                accessed.extend((pos_x.reshape(-1), neg_x.reshape(-1)))

            loss = torch.stack(losses).sum()
            row_ids = torch.unique(torch.cat(accessed))
            row_l2 = model.embedding(row_ids).pow(2).sum(dim=1).mean()
            row_l2 = row_l2 + model.linear_embedding(row_ids).pow(2).sum(dim=1).mean()
            loss = loss + 1e-3 * row_l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, val_x, device, batch_size * 2)
        current = evaluate(val_users, val_y, scores)
        gauc, ndcg, primary = metric_values(current)
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_scores = scores.copy()
            best_metrics = (gauc, ndcg, primary)
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 6:
            break

    if best_scores is None:
        best_scores = predict(model, val_x, device, batch_size * 2)
        best_metrics = metric_values(evaluate(val_users, val_y, best_scores))

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_users, val_videos, best_scores)):
            if isinstance(user_id, np.generic):
                user_id = user_id.item()
            if isinstance(video_id, np.generic):
                video_id = video_id.item()
            writer.writerow([i, user_id, video_id, float(score)])

    gauc, ndcg, primary = best_metrics
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg, "primary": primary}, f)


if __name__ == "__main__":
    main()
