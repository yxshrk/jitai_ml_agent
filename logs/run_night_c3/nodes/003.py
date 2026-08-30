import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    train_path = data_dir / "train.npz"
    val_path = data_dir / "val.npz"
    with np.load(train_path, allow_pickle=False) as z:
        tr = {k: np.asarray(z[k]) for k in z.files}
    with np.load(val_path, allow_pickle=False) as z:
        va = {k: np.asarray(z[k]) for k in z.files}

    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    train_x = np.asarray(tr["X"], dtype=np.int64)
    val_x = np.asarray(va["X"], dtype=np.int64)
    train_y = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
    val_y = np.asarray(va["y"], dtype=np.float32).reshape(-1)
    train_users = np.asarray(tr["user"]).reshape(-1)
    val_users = np.asarray(va["user"]).reshape(-1)
    train_duration = np.asarray(tr["duration_ms"], dtype=np.float32).reshape(-1)
    val_duration = np.asarray(va["duration_ms"], dtype=np.float32).reshape(-1)

    if "video_id" in va:
        val_videos = np.asarray(va["video_id"]).reshape(-1)
    elif "video" in va:
        val_videos = np.asarray(va["video"]).reshape(-1)
    else:
        video_offset = int(field_dims[0])
        val_videos = val_x[:, 1].astype(np.int64) - video_offset

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_users": train_users,
        "train_duration": train_duration,
        "val_x": val_x,
        "val_y": val_y,
        "val_users": val_users,
        "val_duration": val_duration,
        "val_videos": val_videos,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", ""),
                "tab": row.get("tab", ""),
                "duration": float(row.get("duration_ms", 0.0) or 0.0),
                "label": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def build_mapping(values):
    unique = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(unique)}


def load_csv(data_dir):
    train_rows = read_csv_rows(data_dir / "train.csv", True)
    val_rows = read_csv_rows(data_dir / "val.csv", False)

    user_map = build_mapping([r["user"] for r in train_rows])
    video_map = build_mapping([r["video"] for r in train_rows])
    author_map = build_mapping([r["author"] for r in train_rows])
    tab_map = build_mapping([r["tab"] for r in train_rows])

    train_duration = np.asarray([r["duration"] for r in train_rows], dtype=np.float32)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9)).astype(np.float32)
    quantiles = np.maximum.accumulate(quantiles)

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
        durations = np.asarray([r["duration"] for r in rows], dtype=np.float32)
        x[:, 0] = np.asarray([user_map.get(r["user"], 0) for r in rows], dtype=np.int64)
        x[:, 1] = np.asarray([video_map.get(r["video"], 0) for r in rows], dtype=np.int64)
        x[:, 2] = np.asarray([author_map.get(r["author"], 0) for r in rows], dtype=np.int64)
        x[:, 3] = np.asarray([tab_map.get(r["tab"], 0) for r in rows], dtype=np.int64)
        x[:, 4] = np.searchsorted(quantiles, durations, side="right").astype(np.int64)
        x += offsets.reshape(1, -1)
        return x, durations

    train_x, train_duration = encode(train_rows)
    val_x, val_duration = encode(val_rows)
    return {
        "train_x": train_x,
        "train_y": np.asarray([r["label"] for r in train_rows], dtype=np.float32),
        "train_users": np.asarray([r["user"] for r in train_rows], dtype=object),
        "train_duration": train_duration,
        "val_x": val_x,
        "val_y": np.asarray([r["label"] for r in val_rows], dtype=np.float32),
        "val_users": np.asarray([r["user"] for r in val_rows], dtype=object),
        "val_duration": val_duration,
        "val_videos": np.asarray([r["video"] for r in val_rows], dtype=object),
        "field_dims": field_dims,
        "fast": False,
    }


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        pos_parts.append(pos)
        neg_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class DurationRegimeDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.15):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = int(len(field_dims) * embed_dim)
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.cross_layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(2)])
        self.cross_bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        trunk_dim = input_dim + hidden_dim
        self.shared_head = nn.Linear(trunk_dim, 1)
        self.short_residual = nn.Linear(trunk_dim, 1)
        self.long_residual = nn.Linear(trunk_dim, 1)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, short_regime):
        x0 = self.embedding(x).flatten(1)
        crossed = x0
        for layer, bias in zip(self.cross_layers, self.cross_bias):
            crossed = x0 * layer(crossed) + bias + crossed
        deep = self.mlp(x0)
        trunk = torch.cat([crossed, deep], dim=1)
        shared = self.shared_head(trunk).squeeze(1)
        short_delta = self.short_residual(trunk).squeeze(1)
        long_delta = self.long_residual(trunk).squeeze(1)
        return shared + torch.where(short_regime, short_delta, long_delta)

    def regime_penalty(self):
        return (
            self.short_residual.weight.square().sum()
            + self.short_residual.bias.square().sum()
            + self.long_residual.weight.square().sum()
            + self.long_residual.bias.square().sum()
        )


def predict(model, x, duration, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            rb = torch.as_tensor(duration[start:end] <= 18000.0, dtype=torch.bool, device=device)
            result[start:end] = torch.sigmoid(model(xb, rb)).cpu().numpy()
    return result


def official_metrics(fast, users, labels, scores):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    raw = evaluate(users, labels, scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw.get("primary")),
    }


def train_model(data, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    pos_idx, neg_idx = make_pairs(data["train_users"], data["train_y"], seed)
    rng = np.random.default_rng(seed)
    n = len(data["train_y"])
    batch_size = 8192
    pair_batch_size = 4096
    best_gauc = -np.inf
    best_state = None

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        pair_order = rng.permutation(len(pos_idx)) if len(pos_idx) else np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, n, batch_size):
            batch = order[start:min(start + batch_size, n)]
            xb = torch.as_tensor(data["train_x"][batch], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["train_y"][batch], dtype=torch.float32, device=device)
            rb = torch.as_tensor(data["train_duration"][batch] <= 18000.0, dtype=torch.bool, device=device)
            logits = model(xb, rb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_order):
                if pair_cursor + pair_batch_size > len(pair_order):
                    pair_order = rng.permutation(len(pos_idx))
                    pair_cursor = 0
                chosen = pair_order[pair_cursor:pair_cursor + pair_batch_size]
                pair_cursor += len(chosen)
                pi = pos_idx[chosen]
                ni = neg_idx[chosen]
                px = torch.as_tensor(data["train_x"][pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(data["train_x"][ni], dtype=torch.long, device=device)
                pr = torch.as_tensor(data["train_duration"][pi] <= 18000.0, dtype=torch.bool, device=device)
                nr = torch.as_tensor(data["train_duration"][ni] <= 18000.0, dtype=torch.bool, device=device)
                pair_loss = F.softplus(-(model(px, pr) - model(nx, nr))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            loss = loss + 1.0e-3 * model.regime_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, data["val_x"], data["val_duration"], device)
        metrics = official_metrics(data["fast"], data["val_users"], data["val_y"], scores)
        if metrics["gauc"] > best_gauc:
            best_gauc = metrics["gauc"]
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, data, scores, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_users"], data["val_videos"], scores)):
            writer.writerow([i, user, video, format(float(score), ".10g")])
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if (data_dir / "train.npz").is_file() and (data_dir / "val.npz").is_file():
        data = load_npz(data_dir)
    else:
        data = load_csv(data_dir)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    model, device = train_model(data, args.seed, epochs)
    scores = predict(model, data["val_x"], data["val_duration"], device)
    metrics = official_metrics(data["fast"], data["val_users"], data["val_y"], scores)
    write_outputs(out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
