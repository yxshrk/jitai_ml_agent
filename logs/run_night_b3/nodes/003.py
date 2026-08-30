import argparse
import csv
import json
import os
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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_npz(data_dir):
    train_file = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val_file = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    train = {
        "X": np.asarray(train_file["X"], dtype=np.int64),
        "y": np.asarray(train_file["y"], dtype=np.float32),
        "user": np.asarray(train_file["user"]),
        "duration": np.asarray(train_file["duration_ms"], dtype=np.float32),
    }
    val = {
        "X": np.asarray(val_file["X"], dtype=np.int64),
        "y": np.asarray(val_file["y"], dtype=np.float32),
        "user": np.asarray(val_file["user"]),
        "duration": np.asarray(val_file["duration_ms"], dtype=np.float32),
    }
    dims = np.asarray(train_file["field_dims"], dtype=np.int64).reshape(-1)
    if dims.size != 5:
        raise ValueError("Expected exactly five offset-encoded fields")
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)
    local_video = val["X"][:, 1] - offsets[1]
    val["video"] = local_video.astype(str)
    val["row_id"] = np.arange(len(val["y"]), dtype=np.int64)
    return train, val, dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            item = {
                "row_id": row.get("row_id", str(i)),
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "long_view": float(row["long_view"]),
            }
            if training:
                item["author_id"] = row.get("author_id", "__missing_author__")
            else:
                item["author_id"] = row.get("author_id", "__missing_author__")
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    train_durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    def raw_fields(rows):
        result = []
        for r in rows:
            bucket = int(np.searchsorted(quantiles, r["duration_ms"], side="right"))
            result.append((r["user_id"], r["video_id"], r["author_id"], r["tab"], str(bucket)))
        return result

    train_fields = raw_fields(train_rows)
    val_fields = raw_fields(val_rows)
    mappings = []
    dims = []
    for field in range(5):
        values = sorted({x[field] for x in train_fields})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        mappings.append(mapping)
        dims.append(len(mapping) + 1)
    dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)

    def encode(fields):
        X = np.empty((len(fields), 5), dtype=np.int64)
        for i, values in enumerate(fields):
            for f in range(5):
                X[i, f] = mappings[f].get(values[f], 0) + offsets[f]
        return X

    train = {
        "X": encode(train_fields),
        "y": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in train_rows]),
        "duration": np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32),
    }
    val = {
        "X": encode(val_fields),
        "y": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "duration": np.asarray([r["duration_ms"] for r in val_rows], dtype=np.float32),
        "video": np.asarray([r["video_id"] for r in val_rows]),
        "row_id": np.asarray([r["row_id"] for r in val_rows]),
    }
    return train, val, dims, False


class DurationRegimeDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.20):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross_weight = nn.Linear(width, 1, bias=False)
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.shared_head = nn.Linear(width + hidden_dim, 1)
        self.short_residual = nn.Linear(width + hidden_dim, 1)
        self.long_residual = nn.Linear(width + hidden_dim, 1)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.cross_weight.weight)
        nn.init.zeros_(self.short_residual.weight)
        nn.init.zeros_(self.short_residual.bias)
        nn.init.zeros_(self.long_residual.weight)
        nn.init.zeros_(self.long_residual.bias)

    def forward(self, x, duration_ms):
        embedded = self.embedding(x).flatten(1)
        crossed = embedded + embedded * self.cross_weight(embedded) + self.cross_bias
        deep = self.mlp(embedded)
        representation = torch.cat((crossed, deep), dim=1)
        shared = self.shared_head(representation).squeeze(1)
        short_delta = self.short_residual(representation).squeeze(1)
        long_delta = self.long_residual(representation).squeeze(1)
        routed_delta = torch.where(duration_ms <= 18000.0, short_delta, long_delta)
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return first_order + shared + routed_delta

    def regime_penalty(self):
        return (
            self.short_residual.weight.square().mean()
            + self.short_residual.bias.square().mean()
            + self.long_residual.weight.square().mean()
            + self.long_residual.bias.square().mean()
        )


def build_pair_sources(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg)
    if not positives:
        return np.empty(0, dtype=np.int64), []
    return np.concatenate(positives).astype(np.int64), negatives


def sample_pair_negatives(rng, users, positive_indices, negative_groups):
    negative_lookup = {}
    for group in negative_groups:
        negative_lookup[users[group[0]]] = group
    sampled = np.empty(len(positive_indices), dtype=np.int64)
    for i, pos in enumerate(positive_indices):
        candidates = negative_lookup[users[pos]]
        sampled[i] = candidates[rng.integers(0, len(candidates))]
    return sampled


def predict(model, X, durations, device, batch_size=8192):
    model.eval()
    output = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.as_tensor(X[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(durations[start:end], dtype=torch.float32, device=device)
            output[start:end] = torch.sigmoid(model(xb, db)).cpu().numpy()
    return output


def metric_dict(raw):
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw.get("primary")),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists()
    if fast:
        train, val, field_dims, used_npz = load_npz(data_dir)
        from data.official.evaluate import evaluate
    else:
        train, val, field_dims, used_npz = load_csv(data_dir)
        from harness.evaluate_provisional import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    max_epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))

    rng = np.random.default_rng(args.seed)
    positive_indices, negative_groups = build_pair_sources(train["user"], train["y"])
    batch_size = 4096
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(len(train["y"]))
        if len(positive_indices):
            pair_pos = positive_indices[rng.permutation(len(positive_indices))]
            pair_neg = sample_pair_negatives(rng, train["user"], pair_pos, negative_groups)
        else:
            pair_pos = np.empty(0, dtype=np.int64)
            pair_neg = np.empty(0, dtype=np.int64)

        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(train["X"][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train["y"][idx], dtype=torch.float32, device=device)
            db = torch.as_tensor(train["duration"][idx], dtype=torch.float32, device=device)
            point_logits = model(xb, db)
            point_loss = F.binary_cross_entropy_with_logits(point_logits, yb)

            if len(pair_pos):
                take = rng.integers(0, len(pair_pos), size=len(idx))
                pi = pair_pos[take]
                ni = pair_neg[take]
                px = torch.as_tensor(train["X"][pi], dtype=torch.long, device=device)
                nx = torch.as_tensor(train["X"][ni], dtype=torch.long, device=device)
                pd = torch.as_tensor(train["duration"][pi], dtype=torch.float32, device=device)
                nd = torch.as_tensor(train["duration"][ni], dtype=torch.float32, device=device)
                pair_loss = F.softplus(-(model(px, pd) - model(nx, nd))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss = loss + 1.0e-3 * model.regime_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, val["X"], val["duration"], device)
        raw_metrics = evaluate(val["user"], val["y"], val_scores)
        current_gauc = float(raw_metrics.get("GAUC", raw_metrics.get("gauc")))
        if current_gauc > best_gauc + 1.0e-7:
            best_gauc = current_gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, val["X"], val["duration"], device)
    metrics = metric_dict(evaluate(val["user"], val["y"], scores))

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user_id, video_id, score in zip(val["row_id"], val["user"], val["video"], scores):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".9g")])
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
