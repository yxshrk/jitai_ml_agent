import argparse
import csv
import json
import os
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))


def load_npz(data_dir):
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(train["X"], dtype=np.int64)
    y_train = np.asarray(train["y"], dtype=np.float32)
    u_train = np.asarray(train["user"])
    d_train = np.asarray(train["duration_ms"], dtype=np.float32)
    x_val = np.asarray(val["X"], dtype=np.int64)
    y_val = np.asarray(val["y"], dtype=np.float32)
    u_val = np.asarray(val["user"])
    d_val = np.asarray(val["duration_ms"], dtype=np.float32)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64).reshape(-1)[:5]
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_val = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "u_train": u_train,
        "d_train": d_train,
        "x_val": x_val,
        "y_val": y_val,
        "u_val": u_val,
        "d_val": d_val,
        "field_dims": field_dims,
        "video_val": video_val,
        "npz": True,
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
            if training:
                item["long_view"] = float(row["long_view"])
            else:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    def raw_fields(row):
        bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
        return (row["user_id"], row["video_id"], "__constant_author__", row["tab"], str(bucket))

    maps = []
    train_raw = [raw_fields(r) for r in train_rows]
    val_raw = [raw_fields(r) for r in val_rows]
    for j in range(5):
        values = sorted({r[j] for r in train_raw})
        maps.append({v: i for i, v in enumerate(values)})
    field_dims = np.asarray([len(m) + 1 for m in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(raw):
        x = np.empty((len(raw), 5), dtype=np.int64)
        for i, row in enumerate(raw):
            for j, value in enumerate(row):
                x[i, j] = maps[j].get(value, len(maps[j])) + offsets[j]
        return x

    return {
        "x_train": encode(train_raw),
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "u_train": np.asarray([r["user_id"] for r in train_rows]),
        "d_train": np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float32),
        "x_val": encode(val_raw),
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "u_val": np.asarray([r["user_id"] for r in val_rows]),
        "d_val": np.asarray([r["duration_ms"] for r in val_rows], dtype=np.float32),
        "field_dims": field_dims,
        "video_val": np.asarray([r["video_id"] for r in val_rows]),
        "npz": False,
    }


class DurationRegimeDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        representation_width = width + 64
        self.shared_head = nn.Linear(representation_width, 1)
        self.short_residual_head = nn.Linear(representation_width, 1, bias=True)
        self.long_residual_head = nn.Linear(representation_width, 1, bias=True)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight)
        nn.init.xavier_uniform_(self.cross.weight)
        nn.init.zeros_(self.cross.bias)
        nn.init.zeros_(self.short_residual_head.weight)
        nn.init.zeros_(self.short_residual_head.bias)
        nn.init.zeros_(self.long_residual_head.weight)
        nn.init.zeros_(self.long_residual_head.bias)

    def forward(self, x, duration_ms):
        emb = self.embedding(x)
        flat = emb.flatten(1)
        cross = flat * self.cross(flat) + flat
        deep = self.mlp(flat)
        representation = torch.cat((cross, deep), dim=1)
        linear_term = self.linear(x).sum(dim=1).squeeze(1)
        summed = emb.sum(dim=1)
        fm_term = 0.5 * ((summed * summed) - (emb * emb).sum(dim=1)).sum(dim=1)
        shared = self.shared_head(representation).squeeze(1) + linear_term + fm_term
        short_delta = self.short_residual_head(representation).squeeze(1)
        long_delta = self.long_residual_head(representation).squeeze(1)
        short_mask = (duration_ms <= 18000.0).to(shared.dtype)
        return shared + short_mask * short_delta + (1.0 - short_mask) * long_delta

    def regime_penalty(self):
        return (
            self.short_residual_head.weight.square().mean()
            + self.short_residual_head.bias.square().mean()
            + self.long_residual_head.weight.square().mean()
            + self.long_residual_head.bias.square().mean()
        )


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for k in range(len(boundaries) - 1):
        idx = order[boundaries[k]:boundaries[k + 1]]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if pos.size and neg.size:
            chosen = neg[rng.integers(0, neg.size, size=pos.size)]
            positives.append(pos)
            negatives.append(chosen)
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, x, duration, device, batch_size=16384):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            db = torch.as_tensor(duration[start:end], dtype=torch.float32, device=device)
            output[start:end] = torch.sigmoid(model(xb, db)).cpu().numpy()
    return output


def official_metrics(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def train_model(data, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationRegimeDCN(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    max_epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    batch_size = 4096
    pair_pos, pair_neg = build_pairs(data["u_train"], data["y_train"], seed)
    rng = np.random.default_rng(seed)
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        point_order = rng.permutation(len(data["x_train"]))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(point_order), batch_size):
            point_idx = point_order[start:start + batch_size]
            xb = torch.as_tensor(data["x_train"][point_idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["y_train"][point_idx], dtype=torch.float32, device=device)
            db = torch.as_tensor(data["d_train"][point_idx], dtype=torch.float32, device=device)
            logits = model(xb, db)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_order):
                count = len(point_idx)
                if pair_cursor + count <= len(pair_order):
                    selected = pair_order[pair_cursor:pair_cursor + count]
                    pair_cursor += count
                else:
                    first = pair_order[pair_cursor:]
                    pair_order = rng.permutation(len(pair_pos))
                    needed = count - len(first)
                    selected = np.concatenate((first, pair_order[:needed]))
                    pair_cursor = needed
                pi = pair_pos[selected]
                ni = pair_neg[selected]
                px = torch.as_tensor(data["x_train"][pi], dtype=torch.long, device=device)
                pd = torch.as_tensor(data["d_train"][pi], dtype=torch.float32, device=device)
                nx = torch.as_tensor(data["x_train"][ni], dtype=torch.long, device=device)
                nd = torch.as_tensor(data["d_train"][ni], dtype=torch.float32, device=device)
                pair_loss = F.softplus(-(model(px, pd) - model(nx, nd))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss = loss + 1e-3 * model.regime_penalty()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, data["x_val"], data["d_val"], device)
        metrics = official_metrics(data["npz"], data["u_val"], data["y_val"], val_scores)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 2 and epoch >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, data, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["u_val"], data["video_val"], scores)):
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, "%.10g" % float(score)])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


def main():
    args = parse_args()
    seed_everything(args.seed)
    npz_path = os.path.join(args.data_dir, "train.npz")
    if os.path.exists(npz_path) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    model, device = train_model(data, args.seed)
    scores = predict(model, data["x_val"], data["d_val"], device)
    metrics = official_metrics(data["npz"], data["u_val"], data["y_val"], scores)
    write_outputs(args.out_dir, data, scores, metrics)


if __name__ == "__main__":
    main()
