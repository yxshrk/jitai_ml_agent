import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def sigmoid_np(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_npz(data_dir):
    train_file = Path(data_dir) / "train.npz"
    val_file = Path(data_dir) / "val.npz"
    if not train_file.exists() or not val_file.exists():
        return None
    tr = np.load(train_file, allow_pickle=False)
    va = np.load(val_file, allow_pickle=False)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    y_val = np.asarray(va["y"], dtype=np.float32)
    user_train = np.asarray(tr["user"])
    user_val = np.asarray(va["user"])
    duration_train = np.asarray(tr["duration_ms"], dtype=np.float32)
    duration_val = np.asarray(va["duration_ms"], dtype=np.float32)
    video_offset = int(field_dims[0])
    video_val = x_val[:, 1].astype(np.int64) - video_offset
    return {
        "x_train": x_train,
        "y_train": y_train,
        "user_train": user_train,
        "duration_train": duration_train,
        "x_val": x_val,
        "y_val": y_val,
        "user_val": user_val,
        "video_val": video_val,
        "duration_val": duration_val,
        "field_dims": field_dims,
        "npz": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row.get("user_id", ""),
                "video_id": row.get("video_id", ""),
                "author_id": row.get("author_id", ""),
                "tab": row.get("tab", ""),
                "duration_ms": float(row.get("duration_ms", 0.0) or 0.0),
                "long_view": float(row.get("long_view", 0.0) or 0.0),
            }
            rows.append(item)
    return rows


def fit_mapping(rows, name):
    values = sorted({r[name] for r in rows})
    return {v: i + 1 for i, v in enumerate(values)}


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    user_map = fit_mapping(train_rows, "user_id")
    video_map = fit_mapping(train_rows, "video_id")
    author_map = fit_mapping(train_rows, "author_id")
    tab_map = fit_mapping(train_rows, "tab")
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    if len(train_duration):
        quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
        quantiles = np.maximum.accumulate(quantiles)
    else:
        quantiles = np.zeros(9, dtype=np.float64)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        duration = np.asarray([r["duration_ms"] for r in rows], dtype=np.float32)
        x[:, 0] = np.asarray([user_map.get(r["user_id"], 0) for r in rows]) + offsets[0]
        x[:, 1] = np.asarray([video_map.get(r["video_id"], 0) for r in rows]) + offsets[1]
        x[:, 2] = np.asarray([author_map.get(r["author_id"], 0) for r in rows]) + offsets[2]
        x[:, 3] = np.asarray([tab_map.get(r["tab"], 0) for r in rows]) + offsets[3]
        x[:, 4] = np.searchsorted(quantiles, duration, side="right") + offsets[4]
        y = np.asarray([r["long_view"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user_id"] for r in rows])
        videos = np.asarray([r["video_id"] for r in rows])
        return x, y, users, videos, duration

    x_train, y_train, user_train, _, duration_train = encode(train_rows)
    x_val, y_val, user_val, video_val, duration_val = encode(val_rows)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "user_train": user_train,
        "duration_train": duration_train,
        "x_val": x_val,
        "y_val": y_val,
        "user_val": user_val,
        "video_val": video_val,
        "duration_val": duration_val,
        "field_dims": field_dims,
        "npz": False,
    }


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DurationRegimeDCN(nn.Module):
    def __init__(self, total_features, n_fields=5, embed_dim=16):
        super().__init__()
        width = n_fields * embed_dim
        self.embedding = nn.Embedding(total_features, embed_dim)
        self.linear = nn.Embedding(total_features, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        self.input_dropout = nn.Dropout(0.15)
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        trunk_width = width + 64
        self.shared_head = nn.Linear(trunk_width, 1)
        self.short_residual_head = nn.Linear(trunk_width, 1, bias=True)
        self.long_residual_head = nn.Linear(trunk_width, 1, bias=True)
        nn.init.zeros_(self.short_residual_head.weight)
        nn.init.zeros_(self.short_residual_head.bias)
        nn.init.zeros_(self.long_residual_head.weight)
        nn.init.zeros_(self.long_residual_head.bias)

    def forward(self, x, short_regime):
        emb = self.embedding(x).flatten(1)
        emb = self.input_dropout(emb)
        crossed = self.cross1(emb, emb)
        crossed = self.cross2(emb, crossed)
        deep = self.mlp(emb)
        trunk = torch.cat([crossed, deep], dim=1)
        shared = self.shared_head(trunk).squeeze(1)
        short_delta = self.short_residual_head(trunk).squeeze(1)
        long_delta = self.long_residual_head(trunk).squeeze(1)
        routed_delta = torch.where(short_regime, short_delta, long_delta)
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return first_order + shared + routed_delta

    def regime_penalty(self):
        return (
            self.short_residual_head.weight.square().mean()
            + self.short_residual_head.bias.square().mean()
            + self.long_residual_head.weight.square().mean()
            + self.long_residual_head.bias.square().mean()
        )


def prepare_user_groups(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    groups = []
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos, neg))
    return groups


def make_pairs(groups, rng):
    positive_parts = []
    negative_parts = []
    for pos, neg in groups:
        count = max(len(pos), len(neg))
        positive_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negative_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def predict(model, x, duration, device, batch_size=32768):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            short = torch.as_tensor(duration[start:end] <= 18000.0, dtype=torch.bool, device=device)
            outputs.append(model(xb, short).cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return sigmoid_np(np.concatenate(outputs)).astype(np.float32)


def metric_values(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5"))
    primary = result.get("primary")
    return float(gauc), float(ndcg), float(primary)


def train_model(data, seed):
    x_train = data["x_train"]
    y_train = data["y_train"]
    duration_train = data["duration_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    duration_val = data["duration_val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(int(np.sum(data["field_dims"]))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    groups = prepare_user_groups(np.asarray(data["user_train"]), y_train)
    smoke = os.environ.get("SMOKE_EPOCHS")
    max_epochs = 7
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    batch_size = 16384
    rng = np.random.default_rng(seed)
    best_gauc = -math.inf
    best_state = None
    bad_checks = 0

    for epoch in range(max_epochs):
        point_order = rng.permutation(len(x_train))
        pair_pos, pair_neg = make_pairs(groups, rng)
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
            pair_pos = pair_pos[pair_order]
            pair_neg = pair_neg[pair_order]
        midpoint = (len(point_order) + 1) // 2
        halves = (point_order[:midpoint], point_order[midpoint:])
        for half_indices in halves:
            model.train()
            for local_start in range(0, len(half_indices), batch_size):
                idx = half_indices[local_start:local_start + batch_size]
                xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
                yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
                short_b = torch.as_tensor(duration_train[idx] <= 18000.0, dtype=torch.bool, device=device)
                logits = model(xb, short_b)
                point_loss = F.binary_cross_entropy_with_logits(logits, yb)

                if len(pair_pos):
                    pair_start = (local_start + epoch * batch_size) % len(pair_pos)
                    pair_ids = np.arange(pair_start, pair_start + len(idx), dtype=np.int64) % len(pair_pos)
                    pi = pair_pos[pair_ids]
                    ni = pair_neg[pair_ids]
                    px = torch.as_tensor(x_train[pi], dtype=torch.long, device=device)
                    nx = torch.as_tensor(x_train[ni], dtype=torch.long, device=device)
                    ps = torch.as_tensor(duration_train[pi] <= 18000.0, dtype=torch.bool, device=device)
                    ns = torch.as_tensor(duration_train[ni] <= 18000.0, dtype=torch.bool, device=device)
                    pos_logits = model(px, ps)
                    neg_logits = model(nx, ns)
                    pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                loss = loss + 1.0e-3 * model.regime_penalty()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            val_scores = predict(model, x_val, duration_val, device)
            gauc, _, _ = metric_values(data["npz"], data["user_val"], y_val, val_scores)
            if gauc > best_gauc + 1.0e-7:
                best_gauc = gauc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
            if bad_checks >= 6:
                break
        scheduler.step()
        if bad_checks >= 6:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, data, scores):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(data["user_val"], data["video_val"], scores)):
            writer.writerow([i, u.item() if isinstance(u, np.generic) else u, v.item() if isinstance(v, np.generic) else v, float(s)])
    gauc, ndcg, primary = metric_values(data["npz"], data["user_val"], data["y_val"], scores)
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg, "primary": primary}, f)


def main():
    args = parse_args()
    set_seed(args.seed)
    data = load_npz(args.data_dir)
    if data is None:
        data = load_csv(args.data_dir)
    model, device = train_model(data, args.seed)
    scores = predict(model, data["x_val"], data["duration_val"], device)
    write_outputs(args.out_dir, data, scores)


if __name__ == "__main__":
    main()
