import argparse
import csv
import json
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


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
    train_file = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val_file = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    x_train = np.asarray(train_file["X"], dtype=np.int64)
    y_train = np.asarray(train_file["y"], dtype=np.float32).reshape(-1)
    x_val = np.asarray(val_file["X"], dtype=np.int64)
    y_val = np.asarray(val_file["y"], dtype=np.float32).reshape(-1)
    user_val = np.asarray(val_file["user"]).reshape(-1)
    field_dims = np.asarray(train_file["field_dims"], dtype=np.int64).reshape(-1)
    if "video" in val_file.files:
        video_val = np.asarray(val_file["video"]).reshape(-1)
    else:
        video_val = x_val[:, 1] - int(field_dims[0])
    return x_train, y_train, x_val, y_val, user_val, video_val, field_dims, True


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
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    user_map = make_mapping(row["user_id"] for row in train_rows)
    video_map = make_mapping(row["video_id"] for row in train_rows)
    tab_map = make_mapping(row["tab"] for row in train_rows)
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    if len(durations):
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            local = np.asarray([
                user_map.get(row["user_id"], 0),
                video_map.get(row["video_id"], 0),
                0,
                tab_map.get(row["tab"], 0),
                int(np.searchsorted(edges, row["duration_ms"], side="right")),
            ], dtype=np.int64)
            x[i] = local + offsets
            y[i] = row["long_view"]
        return x, y

    x_train, y_train = encode(train_rows)
    x_val, y_val = encode(val_rows)
    user_val = np.asarray([row["user_id"] for row in val_rows], dtype=object)
    video_val = np.asarray([row["video_id"] for row in val_rows], dtype=object)
    return x_train, y_train, x_val, y_val, user_val, video_val, field_dims, False


def build_top5_groups(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    groups = []
    for k in range(len(boundaries) - 1):
        member = order[boundaries[k]:boundaries[k + 1]].copy()
        if len(member) < 5:
            continue
        rng.shuffle(member)
        usable = (len(member) // 5) * 5
        member = member[:usable].reshape(-1, 5)
        group_labels = labels[member]
        mixed = (group_labels.sum(axis=1) > 0) & (group_labels.sum(axis=1) < 5)
        if np.any(mixed):
            groups.append(member[mixed])
    if not groups:
        return np.empty((0, 5), dtype=np.int64)
    return np.concatenate(groups, axis=0).astype(np.int64, copy=False)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.20):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross_w = nn.Parameter(torch.zeros(width))
        self.cross_b = nn.Parameter(torch.zeros(width))
        self.cross_out = nn.Linear(width, 1, bias=False)
        self.deep = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.deep[0].weight)
        nn.init.zeros_(self.deep[0].bias)
        nn.init.xavier_uniform_(self.deep[3].weight)
        nn.init.zeros_(self.deep[3].bias)
        nn.init.zeros_(self.cross_out.weight)

    def forward(self, x):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross = x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b + x0
        first_order = self.linear(x).sum(dim=1).squeeze(-1)
        return first_order + self.cross_out(cross).squeeze(-1) + self.deep(x0).squeeze(-1) + self.bias


def delta_ndcg_bpr(scores, labels):
    batch_size, group_size = scores.shape
    order = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    positions = torch.arange(group_size, device=scores.device).view(1, -1).expand(batch_size, -1)
    ranks.scatter_(1, order, positions)
    discounts = 1.0 / torch.log2(torch.arange(group_size, device=scores.device, dtype=scores.dtype) + 2.0)
    item_discount = discounts[ranks]
    positive_count = labels.sum(dim=1).long().clamp(min=1, max=group_size)
    cumulative = torch.cumsum(discounts, dim=0)
    idcg = cumulative[positive_count - 1]
    pair_mask = (labels.unsqueeze(2) > 0.5) & (labels.unsqueeze(1) < 0.5)
    delta = torch.abs(item_discount.unsqueeze(2) - item_discount.unsqueeze(1)) / idcg.view(-1, 1, 1)
    pair_loss = F.softplus(-(scores.unsqueeze(2) - scores.unsqueeze(1)))
    weights = delta * pair_mask.to(delta.dtype)
    return torch.sum(weights * pair_loss) / torch.sum(weights).clamp_min(1e-8)


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def metric_values(use_npz, users, labels, scores):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5"))
    primary = result.get("primary")
    return float(gauc), float(ndcg), float(primary)


def train_model(x_train, y_train, x_val, y_val, val_users, field_dims, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    groups = build_top5_groups(x_train[:, 0], y_train, seed)
    rng = np.random.default_rng(seed)
    batch_size = 8192
    group_batch_size = 512
    best_gauc = -math.inf
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        row_order = rng.permutation(len(x_train))
        if len(groups):
            group_order = rng.permutation(len(groups))
        else:
            group_order = np.empty(0, dtype=np.int64)
        group_cursor = 0

        for start in range(0, len(row_order), batch_size):
            idx = row_order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(groups):
                if group_cursor + group_batch_size > len(group_order):
                    group_order = rng.permutation(len(groups))
                    group_cursor = 0
                selected = group_order[group_cursor:group_cursor + group_batch_size]
                group_cursor += len(selected)
                group_idx = groups[selected]
                gx = torch.as_tensor(x_train[group_idx.reshape(-1)], dtype=torch.long, device=device)
                gy = torch.as_tensor(y_train[group_idx], dtype=torch.float32, device=device)
                group_scores = model(gx).reshape(len(selected), 5)
                rank_loss = delta_ndcg_bpr(group_scores, gy)
                loss = 0.5 * point_loss + 0.5 * rank_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        val_scores = predict(model, x_val, device)
        try:
            from data.official.evaluate import evaluate as official_evaluate
            result = official_evaluate(val_users, y_val, val_scores)
        except (ImportError, ModuleNotFoundError):
            from harness.evaluate_provisional import evaluate as provisional_evaluate
            result = provisional_evaluate(val_users, y_val, val_scores)
        gauc = float(result.get("GAUC", result.get("gauc")))
        if gauc > best_gauc:
            best_gauc = gauc
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def scalar_text(value):
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_npz = (data_dir / "train.npz").is_file() and (data_dir / "val.npz").is_file()
    if use_npz:
        x_train, y_train, x_val, y_val, user_val, video_val, field_dims, npz_mode = load_npz(data_dir)
    else:
        x_train, y_train, x_val, y_val, user_val, video_val, field_dims, npz_mode = load_csv(data_dir)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    model, device = train_model(
        x_train, y_train, x_val, y_val, user_val, field_dims, args.seed, epochs
    )
    scores = predict(model, x_val, device)
    gauc, ndcg5, primary = metric_values(npz_mode, user_val, y_val, scores)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i in range(len(scores)):
            writer.writerow([i, scalar_text(user_val[i]), scalar_text(video_val[i]), format(float(scores[i]), ".9g")])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
