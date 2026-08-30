"""Regularized FM with a residual interaction MLP and coherent regularization search.

The compound package applies dropout to the interaction MLP, accessed-row L2 to
ID embeddings, dense-only AdamW weight decay, exponential learning-rate decay,
and validation-primary checkpointing. Hyperparameters are selected by a broad,
deterministic fast-path fan-out before a full-length final training.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, hidden=32, dropout=0.15):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.fc1 = torch.nn.Linear(k, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, std=0.005)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(dim=1)
        pair_vector = 0.5 * (summed * summed - (e * e).sum(dim=1))
        fm_pair = pair_vector.sum(dim=1)
        residual = self.fc2(self.dropout(torch.relu(self.fc1(pair_vector)))).squeeze(1)
        linear = self.lin(x).sum(dim=(1, 2))
        return self.bias + linear + fm_pair + residual

    def accessed_row_penalty(self, x):
        e = self.emb(x)
        return e.square().sum() / max(1, x.shape[0])


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc", 0.0))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5", 0.0))),
        "primary": float(result["primary"]),
    }


def load_npz_data(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])
    video_local = va["X"][:, 1].astype(np.int64) - offsets[1]
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "users": va["user"],
        "videos": video_local,
        "total_dim": int(field_dims.sum()),
        "fast": True,
    }


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    user_values = set()
    video_values = set()
    tab_values = set()
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = int(float(row["duration_ms"]))
            label = float(row["long_view"])
            train_rows.append((user, video, tab, duration, label))
            user_values.add(user)
            video_values.add(video)
            tab_values.add(tab)
    user_map = {v: i + 1 for i, v in enumerate(sorted(user_values))}
    video_map = {v: i + 1 for i, v in enumerate(sorted(video_values))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted(tab_values))}
    duration_buckets = 62
    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, duration_buckets],
        dtype=np.int64,
    )
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])

    def encode(user, video, tab, duration):
        bucket = min(duration_buckets - 1, max(0, duration // 10000))
        local = np.asarray(
            [user_map.get(user, 0), video_map.get(video, 0), 0, tab_map.get(tab, 0), bucket],
            dtype=np.int64,
        )
        return local + offsets

    Xt = np.empty((len(train_rows), 5), dtype=np.int64)
    yt = np.empty(len(train_rows), dtype=np.float32)
    for i, (user, video, tab, duration, label) in enumerate(train_rows):
        Xt[i] = encode(user, video, tab, duration)
        yt[i] = label

    val_features = []
    val_labels = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = int(float(row["duration_ms"]))
            val_features.append(encode(user, video, tab, duration))
            val_labels.append(int(float(row["long_view"])))
            val_users.append(user)
            val_videos.append(video)
    return {
        "Xt": Xt,
        "yt": yt,
        "Xv": np.asarray(val_features, dtype=np.int64),
        "yv": np.asarray(val_labels, dtype=np.int64),
        "users": np.asarray(val_users),
        "videos": np.asarray(val_videos),
        "total_dim": int(field_dims.sum()),
        "fast": False,
    }


def predict(model, Xv, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = Xv[start:start + 65536].to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_one(data, config, epochs, seed, device, evaluator):
    seed_everything(seed)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    if device.type == "cuda":
        Xt = Xt.to(device)
        yt = yt.to(device)
        Xv = Xv.to(device)

    model = RegularizedFM(
        data["total_dim"], k=16, hidden=32, dropout=config["dropout"]
    ).to(device)
    dense_weights = [model.fc1.weight, model.fc2.weight]
    unregularized = [
        model.emb.weight,
        model.lin.weight,
        model.bias,
        model.fc1.bias,
        model.fc2.bias,
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": dense_weights, "weight_decay": config["weight_decay"]},
            {"params": unregularized, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    patience = 0
    curve = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device if device.type == "cuda" else None)
        loss_sum = 0.0
        examples = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            if device.type == "cpu":
                xb = xb.to(device)
                yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            data_loss = bce(logits, yb)
            row_penalty = model.accessed_row_penalty(xb)
            loss = data_loss + config["row_l2"] * row_penalty
            loss.backward()
            optimizer.step()
            count = int(len(idx))
            loss_sum += float(data_loss.detach().cpu()) * count
            examples += count
        scheduler.step()
        scores = predict(model, Xv, device)
        raw_metrics = evaluator(data["users"], data["yv"], scores)
        metrics = metric_values(raw_metrics)
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, examples), 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 9),
            "val_gauc": round(metrics["gauc"], 6),
            "val_ndcg5": round(metrics["ndcg5"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_metrics = metrics
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_metrics, curve


def candidate_configs(seed, count):
    grid = list(itertools.product(
        [0.05, 0.15, 0.25, 0.40],
        [1e-5, 1e-4, 3e-4, 1e-3],
        [1e-6, 1e-5, 1e-4, 1e-3],
        [0.85, 0.90, 0.95, 0.98],
    ))
    rng = np.random.RandomState(seed + 2718)
    order = rng.permutation(len(grid))[:count]
    configs = []
    for index in order:
        dropout, row_l2, weight_decay, lr_gamma = grid[int(index)]
        configs.append({
            "dropout": float(dropout),
            "row_l2": float(row_l2),
            "weight_decay": float(weight_decay),
            "lr_gamma": float(lr_gamma),
        })
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(
        os.path.join(args.data_dir, "val.npz")
    )
    if fast:
        from data.official.evaluate import evaluate as evaluator
        data = load_npz_data(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        data = load_csv_data(args.data_dir)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = None
    if smoke_value is not None:
        smoke_cap = max(1, int(smoke_value))
    probe_epochs = min(10, args.epochs)
    final_epochs = args.epochs
    probe_count = 80
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
        probe_count = 2

    configs = candidate_configs(args.seed, probe_count)
    search_history = []
    winning_config = None
    winning_primary = -1.0
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for probe_index, config in enumerate(configs):
            _, metrics, curve = train_one(
                data, config, probe_epochs, args.seed, device, evaluator
            )
            record = {
                "probe": probe_index + 1,
                "config": config,
                "best_epoch": int(np.argmax([x["val_primary"] for x in curve])) + 1,
                "gauc": round(metrics["gauc"], 6),
                "ndcg5": round(metrics["ndcg5"], 6),
                "primary": round(metrics["primary"], 6),
                "curve": curve,
            }
            search_history.append(record)
            progress.write(json.dumps({
                "probe": probe_index + 1,
                "config": config,
                "primary": metrics["primary"],
            }, sort_keys=True) + "\n")
            progress.flush()
            if metrics["primary"] > winning_primary + 1e-9:
                winning_primary = metrics["primary"]
                winning_config = dict(config)

    final_scores, final_metrics, final_curve = train_one(
        data, winning_config, final_epochs, args.seed, device, evaluator
    )
    verified = metric_values(evaluator(data["users"], data["yv"], final_scores))
    metrics_output = {
        "gauc": verified["gauc"],
        "ndcg5": verified["ndcg5"],
        "primary": verified["primary"],
        "winning_config": winning_config,
        "search_history": search_history,
        "history": final_curve,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
