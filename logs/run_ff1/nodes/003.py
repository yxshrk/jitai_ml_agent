"""Regularized residual neural FM with validation-screened compound regularization."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, hidden=32, dropout=0.2):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.fc1 = torch.nn.Linear(k, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.lin.weight, std=0.01)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, std=0.01)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        interaction = 0.5 * (s * s - (e * e).sum(1))
        fm_pair = interaction.sum(1)
        residual = self.fc2(self.drop(self.act(self.fc1(interaction)))).squeeze(1)
        return self.bias + self.lin(x).sum((1, 2)) + fm_pair + residual


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv_data(data_dir):
    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                duration = float(row.get("duration_ms", 0) or 0)
                bucket = int(min(31, max(0, np.floor(np.log2(1.0 + duration / 1000.0)))))
                item = {
                    "user": row["user_id"],
                    "video": row["video_id"],
                    "author": "__unknown_author__",
                    "tab": row.get("tab", ""),
                    "dur": str(bucket),
                }
                item["y"] = float(row["long_view"])
                rows.append(item)
        return rows

    train_rows = read_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_rows(os.path.join(data_dir, "val.csv"), False)
    keys = ["user", "video", "author", "tab", "dur"]
    maps = []
    dims = []
    for key in keys:
        values = sorted({r[key] for r in train_rows})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, key in enumerate(keys):
            mapping = maps[j]
            x[:, j] = np.asarray([mapping.get(r[key], 0) for r in rows], dtype=np.int64) + offsets[j]
        y = np.asarray([r["y"] for r in rows], dtype=np.float32)
        return x, y

    xt, yt = encode(train_rows)
    xv, yv = encode(val_rows)
    users = np.asarray([r["user"] for r in val_rows])
    videos = np.asarray([r["video"] for r in val_rows])
    return xt, yt, xv, yv, users, videos, np.asarray(dims, dtype=np.int64), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr = np.load(train_path)
        va = np.load(val_path)
        dims = tr["field_dims"].astype(np.int64)
        xt = tr["X"].astype(np.int64)
        yt = tr["y"].astype(np.float32)
        xv = va["X"].astype(np.int64)
        yv = va["y"].astype(np.float32)
        users = va["user"]
        video_offset = int(dims[0])
        videos = xv[:, 1] - video_offset
        return xt, yt, xv, yv, users, videos, dims, True
    return load_csv_data(data_dir)


def make_configs(seed, count):
    anchors = [
        {"dropout": 0.05, "row_l2": 1e-6, "weight_decay": 1e-5, "gamma": 0.99, "lr": 2e-3},
        {"dropout": 0.10, "row_l2": 3e-6, "weight_decay": 3e-5, "gamma": 0.98, "lr": 2e-3},
        {"dropout": 0.15, "row_l2": 1e-5, "weight_decay": 1e-4, "gamma": 0.97, "lr": 1.5e-3},
        {"dropout": 0.20, "row_l2": 1e-5, "weight_decay": 3e-4, "gamma": 0.96, "lr": 1e-3},
        {"dropout": 0.25, "row_l2": 3e-5, "weight_decay": 1e-4, "gamma": 0.95, "lr": 1e-3},
        {"dropout": 0.30, "row_l2": 3e-5, "weight_decay": 3e-4, "gamma": 0.94, "lr": 8e-4},
        {"dropout": 0.35, "row_l2": 1e-4, "weight_decay": 5e-4, "gamma": 0.92, "lr": 8e-4},
    ]
    rng = np.random.RandomState(seed + 731)
    drops = [0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    l2s = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
    decays = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    gammas = [0.92, 0.94, 0.96, 0.98, 0.99]
    lrs = [8e-4, 1e-3, 1.2e-3, 1.5e-3, 2e-3]
    seen = {tuple(sorted(c.items())) for c in anchors}
    while len(anchors) < count:
        c = {
            "dropout": float(rng.choice(drops)),
            "row_l2": float(rng.choice(l2s)),
            "weight_decay": float(rng.choice(decays)),
            "gamma": float(rng.choice(gammas)),
            "lr": float(rng.choice(lrs)),
        }
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            anchors.append(c)
    return anchors[:count]


def predict(model, xv, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = torch.from_numpy(xv[start:start + 65536]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_model(xt, yt, xv, yv, users, total_dim, config, epochs, seed, device, evaluate_fn):
    set_seed(seed)
    model = RegularizedFM(total_dim, k=16, hidden=32, dropout=config["dropout"]).to(device)
    pos_rate = float(np.clip(np.mean(yt), 1e-4, 1 - 1e-4))
    model.bias.data.fill_(float(np.log(pos_rate / (1.0 - pos_rate))))
    dense_decay = [model.fc1.weight, model.fc2.weight]
    other = [model.emb.weight, model.lin.weight, model.bias, model.fc1.bias, model.fc2.bias]
    optimizer = torch.optim.AdamW(
        [
            {"params": dense_decay, "weight_decay": config["weight_decay"]},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=config["lr"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 2048
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size].numpy()
            xb = torch.from_numpy(xt[idx]).to(device)
            yb = torch.from_numpy(yt[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            unique_rows = torch.unique(xb.reshape(-1))
            row_penalty = model.emb.weight[unique_rows].pow(2).sum(1).mean()
            row_penalty = row_penalty + model.lin.weight[unique_rows].pow(2).sum(1).mean()
            loss = bce(logits, yb) + config["row_l2"] * row_penalty
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        scheduler.step()
        scores = torch.sigmoid(torch.from_numpy(predict(model, xv, device))).cpu().numpy()
        metrics = evaluate_fn(users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        epoch_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 9),
                "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
                "val_primary": round(primary, 6),
            }
        )
        if primary > best_primary + 1e-7:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
        if epochs > 3 and patience >= 3:
            break
    return best_primary, best_scores, epoch_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt, yt, xv, yv, users, videos, dims, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_value) if smoke_value is not None else None
    if smoke_epochs is not None:
        probe_epochs = max(1, min(6, smoke_epochs))
        final_epochs = max(1, min(args.epochs, smoke_epochs))
        probe_count = 4 if smoke_epochs == 1 else (8 if device.type == "cpu" else 12)
    else:
        probe_epochs = 6 if device.type == "cpu" else 8
        final_epochs = args.epochs if device.type == "cpu" else max(args.epochs, 18)
        probe_count = 10 if device.type == "cpu" else 24

    configs = make_configs(args.seed, probe_count)
    progress_path = os.path.join(args.out_dir, "progress.log")
    probe_history = []
    winning_index = -1
    winning_primary = -1.0
    total_dim = int(dims.sum())
    for index, config in enumerate(configs):
        primary, _, epochs = train_model(
            xt, yt, xv, yv, users, total_dim, config, probe_epochs,
            args.seed, device, evaluate_fn
        )
        record = {
            "probe": index,
            "config": config,
            "epochs_requested": probe_epochs,
            "best_primary": round(float(primary), 7),
            "curve": epochs,
        }
        probe_history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"probe": index, "config": config, "primary": primary}, sort_keys=True) + "\n")
        if primary > winning_primary:
            winning_primary = primary
            winning_index = index

    winning_config = configs[winning_index]
    final_primary, best_scores, final_history = train_model(
        xt, yt, xv, yv, users, total_dim, winning_config, final_epochs,
        args.seed, device, evaluate_fn
    )
    final_metrics = evaluate_fn(users, yv.astype(int), best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])
    metrics_payload = {
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "history": {
            "probes": probe_history,
            "selected_probe": winning_index,
            "selected_config": winning_config,
            "probe_best_primary": winning_primary,
            "final_best_primary": final_primary,
            "final_epochs": final_history,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, users[i], videos[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
