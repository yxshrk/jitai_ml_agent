import argparse
import csv
import json
import os
import random
from pathlib import Path

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
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def encode_csv(train_rows, val_rows):
    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = sorted({r[name] for r in train_rows})
        mappings.append({v: i for i, v in enumerate(values)})

    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = [len(m) + 1 for m in mappings] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, (name, mapping) in enumerate(zip(field_names, mappings)):
            unknown = len(mapping)
            x[:, j] = np.asarray([mapping.get(r[name], unknown) for r in rows], dtype=np.int64) + offsets[j]
        durations = np.asarray([r["duration_ms"] for r in rows], dtype=np.float64)
        x[:, 4] = np.searchsorted(quantiles, durations, side="right").astype(np.int64) + offsets[4]
        y = np.asarray([r["long_view"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user_id"] for r in rows])
        videos = np.asarray([r["video_id"] for r in rows])
        return x, y, users, videos

    train = transform(train_rows)
    val = transform(val_rows)
    return train, val, np.asarray(field_dims, dtype=np.int64)


def load_data(data_dir):
    train_npz = data_dir / "train.npz"
    val_npz = data_dir / "val.npz"
    if train_npz.exists() and val_npz.exists():
        with np.load(train_npz, allow_pickle=False) as tr:
            train_x = np.asarray(tr["X"], dtype=np.int64)
            train_y = np.asarray(tr["y"], dtype=np.float32)
            train_user = np.asarray(tr["user"])
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
        with np.load(val_npz, allow_pickle=False) as va:
            val_x = np.asarray(va["X"], dtype=np.int64)
            val_y = np.asarray(va["y"], dtype=np.float32)
            val_user = np.asarray(va["user"])
            if "field_dims" in va:
                field_dims = np.maximum(field_dims, np.asarray(va["field_dims"], dtype=np.int64))
        video_offset = int(field_dims[0])
        val_video = val_x[:, 1].astype(np.int64) - video_offset
        return train_x, train_y, train_user, val_x, val_y, val_user, val_video, field_dims, True

    train_rows = read_csv_rows(data_dir / "train.csv", True)
    val_rows = read_csv_rows(data_dir / "val.csv", False)
    train, val, field_dims = encode_csv(train_rows, val_rows)
    train_x, train_y, train_user, _ = train
    val_x, val_y, val_user, val_video = val
    return train_x, train_y, train_user, val_x, val_y, val_user, val_video, field_dims, False


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.20):
        super().__init__()
        total = int(np.sum(field_dims))
        fields = len(field_dims)
        flat_dim = fields * embed_dim
        self.fields = fields
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.cross1 = CrossLayer(flat_dim)
        self.cross2 = CrossLayer(flat_dim)
        self.deep = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(flat_dim + 64, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        emb = self.embedding_dropout(self.embedding(x))
        summed = torch.sum(emb, dim=1)
        fm = 0.5 * torch.sum(summed * summed - torch.sum(emb * emb, dim=1), dim=1)
        linear = torch.sum(self.linear_embedding(x), dim=1).squeeze(1)
        x0 = emb.reshape(emb.shape[0], -1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        dcn = self.output(torch.cat([cross, deep], dim=1)).squeeze(1)
        return self.bias + linear + fm + dcn


def make_groups(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        idx = order[start:end].copy()
        if idx.size < 5:
            continue
        rng.shuffle(idx)
        usable = (idx.size // 5) * 5
        group = idx[:usable].reshape(-1, 5)
        sums = labels[group].sum(axis=1)
        group = group[(sums > 0.0) & (sums < 5.0)]
        if group.size:
            chunks.append(group)
    if not chunks:
        return np.empty((0, 5), dtype=np.int64)
    groups = np.concatenate(chunks, axis=0).astype(np.int64, copy=False)
    rng.shuffle(groups)
    return groups


def lambda_ndcg_loss(logits, labels):
    batch, width = logits.shape
    sorted_idx = torch.argsort(logits.detach(), dim=1, descending=True)
    ranks = torch.empty_like(sorted_idx)
    positions = torch.arange(width, device=logits.device).view(1, -1).expand(batch, -1)
    ranks.scatter_(1, sorted_idx, positions)
    discounts = 1.0 / torch.log2(ranks.to(torch.float32) + 2.0)

    positives = labels.sum(dim=1).to(torch.long).clamp(min=1, max=width)
    base_discounts = 1.0 / torch.log2(torch.arange(width, device=logits.device, dtype=torch.float32) + 2.0)
    prefix = torch.cumsum(base_discounts, dim=0)
    idcg = prefix[positives - 1].view(-1, 1, 1)

    pair_mask = (labels.unsqueeze(2) > 0.5) & (labels.unsqueeze(1) < 0.5)
    delta = torch.abs(discounts.unsqueeze(2) - discounts.unsqueeze(1)) / idcg
    pair_weight = delta * pair_mask.to(delta.dtype)
    score_diff = logits.unsqueeze(2) - logits.unsqueeze(1)
    losses = F.softplus(-score_diff) * pair_weight
    return losses.sum() / pair_weight.sum().clamp_min(1e-8)


def predict(model, x, device, batch_size=8192):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            output[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return output


def official_metrics(users, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


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

    train_x, train_y, train_user, val_x, val_y, val_user, val_video, field_dims, fast_path = load_data(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims, embed_dim=16, dropout=0.20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.80)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    rng = np.random.default_rng(args.seed)
    group_batch = 256
    point_batch = group_batch * 5
    best_state = None
    best_gauc = -float("inf")
    stale = 0

    for _ in range(epochs):
        groups = make_groups(train_user, train_y, rng)
        point_order = rng.permutation(len(train_x))
        group_steps = (len(groups) + group_batch - 1) // group_batch if len(groups) else 0
        point_steps = (len(point_order) + point_batch - 1) // point_batch
        steps = max(group_steps, point_steps)
        model.train()

        for step in range(steps):
            pstart = (step * point_batch) % len(point_order)
            point_idx = point_order[pstart:pstart + point_batch]
            if len(point_idx) < point_batch:
                point_idx = np.concatenate([point_idx, point_order[:point_batch - len(point_idx)]])
            px = torch.as_tensor(train_x[point_idx], dtype=torch.long, device=device)
            py = torch.as_tensor(train_y[point_idx], dtype=torch.float32, device=device)
            point_loss = F.binary_cross_entropy_with_logits(model(px), py)

            if len(groups):
                gstart = (step * group_batch) % len(groups)
                group_idx = groups[gstart:gstart + group_batch]
                if len(group_idx) < group_batch:
                    group_idx = np.concatenate([group_idx, groups[:group_batch - len(group_idx)]], axis=0)
                gx = torch.as_tensor(train_x[group_idx.reshape(-1)], dtype=torch.long, device=device)
                gy = torch.as_tensor(train_y[group_idx], dtype=torch.float32, device=device)
                group_logits = model(gx).reshape(-1, 5)
                rank_loss = lambda_ndcg_loss(group_logits, gy)
                loss = 0.5 * point_loss + 0.5 * rank_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        val_scores = predict(model, val_x, device)
        metrics = official_metrics(val_user, val_y, val_scores, fast_path)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores = predict(model, val_x, device)
    metrics = official_metrics(val_user, val_y, val_scores, fast_path)

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(val_user, val_video, val_scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user, video.item() if isinstance(video, np.generic) else video, float(score)])

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
