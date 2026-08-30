import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(result):
    def get(*names):
        for name in names:
            if name in result:
                return float(result[name])
        raise KeyError(names[0])
    return {
        "gauc": get("GAUC", "gauc"),
        "ndcg5": get("nDCG@5", "ndcg5", "ndcg@5"),
        "primary": get("primary", "Primary"),
    }


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None
    with np.load(train_path, allow_pickle=False) as z:
        x_train = np.asarray(z["X"], dtype=np.int64)
        y_train = np.asarray(z["y"], dtype=np.float32)
        train_user = np.asarray(z["user"])
        train_duration = np.asarray(z["duration_ms"], dtype=np.float32)
        field_dims = np.asarray(z["field_dims"], dtype=np.int64)
    with np.load(val_path, allow_pickle=False) as z:
        x_val = np.asarray(z["X"], dtype=np.int64)
        y_val = np.asarray(z["y"], dtype=np.float32)
        val_user = np.asarray(z["user"])
        val_duration = np.asarray(z["duration_ms"], dtype=np.float32)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    video_ids = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_user": train_user,
        "train_duration": train_duration,
        "x_val": x_val,
        "y_val": y_val,
        "val_user": val_user,
        "val_duration": val_duration,
        "val_video": video_ids,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
            }
            if not training:
                item["long_view"] = float(row["long_view"])
            else:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {"__UNK__": 0}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32)
    if train_duration.size:
        edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float32)
    field_dims = np.asarray([
        len(user_map), len(video_map), 1, len(tab_map), len(edges) + 1
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        durations = np.asarray([r["duration_ms"] for r in rows], dtype=np.float32)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(edges, row["duration_ms"], side="right")) + offsets[4]
        y = np.asarray([r["long_view"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user_id"] for r in rows])
        videos = np.asarray([r["video_id"] for r in rows])
        return x, y, users, videos, durations

    x_train, y_train, train_user, _, train_duration = encode(train_rows)
    x_val, y_val, val_user, val_video, val_duration = encode(val_rows)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_user": train_user,
        "train_duration": train_duration,
        "x_val": x_val,
        "y_val": y_val,
        "val_user": val_user,
        "val_duration": val_duration,
        "val_video": val_video,
        "field_dims": field_dims,
        "fast": False,
    }


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and sorted_users[end] == sorted_users[start]:
            end += 1
        indices = order[start:end]
        pos = indices[labels[indices] >= 0.5]
        neg = indices[labels[indices] < 0.5]
        if pos.size and neg.size:
            pos_parts.append(pos)
            neg_parts.append(rng.choice(neg, size=pos.size, replace=True))
        start = end
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class DurationRegimeDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.15):
        super().__init__()
        total = int(np.sum(field_dims))
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        width = self.num_fields * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        self.field_dropout = nn.Dropout(dropout)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rep_dim = width + hidden_dim
        self.shared_head = nn.Linear(rep_dim, 1)
        self.short_residual_head = nn.Linear(rep_dim, 1, bias=True)
        self.long_residual_head = nn.Linear(rep_dim, 1, bias=True)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)
        nn.init.zeros_(self.short_residual_head.weight)
        nn.init.zeros_(self.short_residual_head.bias)
        nn.init.zeros_(self.long_residual_head.weight)
        nn.init.zeros_(self.long_residual_head.bias)

    def forward(self, x, regime):
        emb = self.embedding(x)
        emb = self.field_dropout(emb)
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1) + self.global_bias
        summed = emb.sum(dim=1)
        fm = 0.5 * (summed.square() - emb.square().sum(dim=1)).sum(dim=1)
        x0 = emb.reshape(emb.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(cross * w, dim=1, keepdim=True)
            cross = x0 * scale + b + cross
        deep = self.mlp(x0)
        rep = torch.cat([cross, deep], dim=1)
        shared = self.shared_head(rep).squeeze(1)
        short_delta = self.short_residual_head(rep).squeeze(1)
        long_delta = self.long_residual_head(rep).squeeze(1)
        delta = torch.where(regime <= 0, short_delta, long_delta)
        return linear + fm + shared + delta

    def regime_penalty(self):
        return (
            self.short_residual_head.weight.square().mean()
            + self.long_residual_head.weight.square().mean()
            + self.short_residual_head.bias.square().mean()
            + self.long_residual_head.bias.square().mean()
        )


def predict(model, x, duration, device, batch_size):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            rb = torch.as_tensor((duration[start:end] > 18000.0).astype(np.int64), dtype=torch.long, device=device)
            logits = model(xb, rb)
            parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_npz(args.data_dir)
    if data is None:
        data = load_csv(args.data_dir)

    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    x_train = data["x_train"]
    y_train = data["y_train"]
    duration_train = data["train_duration"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    duration_val = data["val_duration"]

    pair_pos, pair_neg = build_pairs(data["train_user"], y_train, args.seed + 17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 8192
    rng = np.random.default_rng(args.seed + 101)
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        point_order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        pair_cursor = 0
        for start in range(0, len(point_order), batch_size):
            idx = point_order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            rb = torch.as_tensor((duration_train[idx] > 18000.0).astype(np.int64), dtype=torch.long, device=device)
            point_logits = model(xb, rb)
            point_loss = F.binary_cross_entropy_with_logits(point_logits, yb)

            if len(pair_pos):
                take = min(len(idx), len(pair_pos))
                if pair_cursor + take > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + take]
                pair_cursor += take
                pi = pair_pos[selected]
                ni = pair_neg[selected]
                px = torch.as_tensor(x_train[pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(x_train[ni], dtype=torch.long, device=device)
                pr = torch.as_tensor((duration_train[pi] > 18000.0).astype(np.int64), dtype=torch.long, device=device)
                nr = torch.as_tensor((duration_train[ni] > 18000.0).astype(np.int64), dtype=torch.long, device=device)
                pair_loss = F.softplus(-(model(px, pr) - model(nx, nr))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss = loss + 1.0e-3 * model.regime_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, x_val, duration_val, device, batch_size)
        result = evaluate(data["val_user"], y_val, val_scores)
        gauc = metric_values(result)["gauc"]
        if gauc > best_gauc + 1.0e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, duration_val, device, batch_size)
    metrics = metric_values(evaluate(data["val_user"], y_val, scores))

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(data["val_user"], data["val_video"], scores)):
            writer.writerow([i, user_id.item() if hasattr(user_id, "item") else user_id,
                             video_id.item() if hasattr(video_id, "item") else video_id,
                             float(score)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
