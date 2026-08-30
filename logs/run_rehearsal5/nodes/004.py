import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=8, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.total_dim = int(np.sum(field_dims))
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        input_dim = self.num_fields * embed_dim

        self.embedding = nn.Embedding(self.total_dim, embed_dim)
        self.linear_embedding = nn.Embedding(self.total_dim, 1)
        self.cross_linear1 = nn.Linear(input_dim, 1)
        self.cross_linear2 = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + hidden_dim // 2, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, x):
        emb = self.embedding(x).reshape(x.shape[0], -1)
        x0 = emb
        cross1 = x0 * self.cross_linear1(emb) + emb
        cross2 = x0 * self.cross_linear2(cross1) + cross1
        deep = self.mlp(emb)
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        return self.output(torch.cat([cross2, deep], dim=1)).squeeze(-1) + first_order + self.bias


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def scalar_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None

    with np.load(train_path, allow_pickle=False) as tr, np.load(val_path, allow_pickle=False) as va:
        x_train = np.asarray(tr["X"], dtype=np.int64)
        y_train = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
        x_val = np.asarray(va["X"], dtype=np.int64)
        y_val = np.asarray(va["y"], dtype=np.float32).reshape(-1)
        train_users = np.asarray(tr["user"]).reshape(-1) if "user" in tr.files else x_train[:, 0]
        val_users = np.asarray(va["user"]).reshape(-1) if "user" in va.files else x_val[:, 0]
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)

        if "video" in va.files:
            val_videos = np.asarray(va["video"]).reshape(-1)
        elif "video_id" in va.files:
            val_videos = np.asarray(va["video_id"]).reshape(-1)
        else:
            video_offset = int(field_dims[0])
            val_videos = x_val[:, 1].astype(np.int64) - video_offset

    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims


def parse_duration(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def duration_bucket(values, boundaries):
    return np.searchsorted(boundaries, values, side="right").astype(np.int64)


def load_csv(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    train_durations = []
    with open(train_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = row.get("user_id", "")
            video = row.get("video_id", "")
            author = row.get("author_id", "0")
            tab = row.get("tab", "")
            duration = parse_duration(row.get("duration_ms", "0"))
            label = float(row.get("long_view", "0"))
            train_rows.append((user, video, author, tab, duration, label))
            train_durations.append(duration)

    duration_array = np.asarray(train_durations, dtype=np.float64)
    if duration_array.size:
        boundaries = np.unique(np.quantile(duration_array, np.arange(1, 10) / 10.0))
    else:
        boundaries = np.empty(0, dtype=np.float64)

    maps = [dict() for _ in range(4)]
    for row in train_rows:
        for j, value in enumerate(row[:4]):
            if value not in maps[j]:
                maps[j][value] = len(maps[j]) + 1

    field_dims = np.asarray([len(m) + 1 for m in maps] + [len(boundaries) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    x_train = np.empty((len(train_rows), 5), dtype=np.int64)
    y_train = np.empty(len(train_rows), dtype=np.float32)
    train_users = np.empty(len(train_rows), dtype=object)
    train_buckets = duration_bucket(duration_array, boundaries)
    for i, row in enumerate(train_rows):
        for j in range(4):
            x_train[i, j] = maps[j][row[j]] + offsets[j]
        x_train[i, 4] = train_buckets[i] + offsets[4]
        y_train[i] = row[5]
        train_users[i] = row[0]

    val_basic = []
    val_durations = []
    val_labels = []
    with open(val_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = row.get("user_id", "")
            video = row.get("video_id", "")
            author = row.get("author_id", "0")
            tab = row.get("tab", "")
            duration = parse_duration(row.get("duration_ms", "0"))
            label = float(row.get("long_view", "0"))
            val_basic.append((user, video, author, tab))
            val_durations.append(duration)
            val_labels.append(label)

    x_val = np.empty((len(val_basic), 5), dtype=np.int64)
    val_buckets = duration_bucket(np.asarray(val_durations, dtype=np.float64), boundaries)
    val_users = np.empty(len(val_basic), dtype=object)
    val_videos = np.empty(len(val_basic), dtype=object)
    for i, row in enumerate(val_basic):
        for j in range(4):
            x_val[i, j] = maps[j].get(row[j], 0) + offsets[j]
        x_val[i, 4] = val_buckets[i] + offsets[4]
        val_users[i] = row[0]
        val_videos[i] = row[1]

    return (
        x_train,
        y_train,
        train_users,
        x_val,
        np.asarray(val_labels, dtype=np.float32),
        val_users,
        val_videos,
        field_dims,
    )


def make_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.split(order, boundaries)
    positives = []
    negatives = []
    for group in groups:
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if pos.size == 0 or neg.size == 0:
            continue
        count = max(pos.size, neg.size)
        positives.append(pos[rng.integers(0, pos.size, size=count)])
        negatives.append(neg[rng.integers(0, neg.size, size=count)])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_evaluate(users, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5"))
    primary = result.get("primary")
    return {"gauc": float(gauc), "ndcg5": float(ndcg), "primary": float(primary)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = load_npz(args.data_dir)
    fast_path = loaded is not None
    if loaded is None:
        loaded = load_csv(args.data_dir)
    x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims = loaded

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = DCNLite(field_dims, embed_dim=8, hidden_dim=128, dropout=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(args.seed)
    pair_pos, pair_neg = make_pairs(np.asarray(train_users), y_train, rng)

    batch_size = 8192
    pair_batch_size = 4096
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(8):
        model.train()
        point_order = rng.permutation(x_train.shape[0])
        if pair_pos.size:
            pair_order = rng.permutation(pair_pos.size)
        else:
            pair_order = np.empty(0, dtype=np.int64)

        steps = (x_train.shape[0] + batch_size - 1) // batch_size
        for step in range(steps):
            start = step * batch_size
            idx = point_order[start:min(start + batch_size, x_train.shape[0])]
            xb = torch.from_numpy(x_train[idx]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y_train[idx]).to(device=device, dtype=torch.float32)
            point_loss = F.binary_cross_entropy_with_logits(model(xb), yb)

            if pair_order.size:
                pstart = (step * pair_batch_size) % pair_order.size
                pend = pstart + pair_batch_size
                if pend <= pair_order.size:
                    chosen = pair_order[pstart:pend]
                else:
                    chosen = np.concatenate((pair_order[pstart:], pair_order[:pend - pair_order.size]))
                pos_x = torch.from_numpy(x_train[pair_pos[chosen]]).to(device=device, dtype=torch.long)
                neg_x = torch.from_numpy(x_train[pair_neg[chosen]]).to(device=device, dtype=torch.long)
                pair_loss = F.softplus(-(model(pos_x) - model(neg_x))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_scores = predict(model, x_val, device)
        current = official_evaluate(val_users, y_val, val_scores, fast_path)
        if current["gauc"] > best_gauc + 1.0e-7:
            best_gauc = current["gauc"]
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, device)
    metrics = official_evaluate(val_users, y_val, scores, fast_path)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, scalar_text(val_users[i]), scalar_text(val_videos[i]), format(float(score), ".10g")])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
