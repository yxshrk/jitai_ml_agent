import argparse
import csv
import json
import os
import sys

import numpy as np
import torch


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(dim=1)
        pair = 0.5 * (s * s - (e * e).sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


def centered_logits(raw_logits, user_ids, global_bias):
    _, inverse, counts = torch.unique_consecutive(
        user_ids, return_inverse=True, return_counts=True
    )
    sums = torch.zeros(
        counts.numel(), device=raw_logits.device, dtype=raw_logits.dtype
    )
    sums.scatter_add_(0, inverse, raw_logits)
    means = sums / counts.to(raw_logits.dtype)
    return raw_logits - means[inverse] + global_bias


def make_complete_user_batches(users, target_size):
    users = np.asarray(users)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    groups = np.split(order, boundaries)
    batches = []
    pending = []
    pending_size = 0
    for group in groups:
        group_size = len(group)
        if pending and pending_size + group_size > target_size:
            batches.append(np.concatenate(pending).astype(np.int64, copy=False))
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += group_size
        if pending_size >= target_size:
            batches.append(np.concatenate(pending).astype(np.int64, copy=False))
            pending = []
            pending_size = 0
    if pending:
        batches.append(np.concatenate(pending).astype(np.int64, copy=False))
    return batches


def center_numpy_scores(raw_scores, users, global_bias):
    users = np.asarray(users)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=raw_scores)
    means = sums / np.maximum(counts, 1.0)
    return (raw_scores - means[inverse] + float(global_bias)).astype(np.float32)


def read_csv_rows(path, need_label):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def categorical_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), True)

    user_map = categorical_mapping([r["user_id"] for r in train_rows])
    video_map = categorical_mapping([r["video_id"] for r in train_rows])
    tab_map = categorical_mapping([r["tab"] for r in train_rows])

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    if len(durations):
        thresholds = np.quantile(durations, np.arange(1, 10) / 10.0)
    else:
        thresholds = np.zeros(9, dtype=np.float64)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        users = []
        videos = []
        labels = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            u = user_map.get(row["user_id"], len(user_map))
            v = video_map.get(row["video_id"], len(video_map))
            t = tab_map.get(row["tab"], len(tab_map))
            d = int(np.searchsorted(thresholds, row["duration_ms"], side="right"))
            x[i] = np.asarray([u, v, 0, t, d], dtype=np.int64) + offsets
            users.append(row["user_id"])
            videos.append(row["video_id"])
            labels[i] = row["long_view"]
        return x, labels, np.asarray(users, dtype=object), np.asarray(videos, dtype=object)

    xt, yt, train_users, _ = encode(train_rows)
    xv, yv, val_users, val_videos = encode(val_rows)
    return {
        "X_train": xt,
        "y_train": yt,
        "train_users": train_users,
        "X_val": xv,
        "y_val": yv,
        "val_users": val_users,
        "val_videos": val_videos,
        "field_dims": field_dims,
        "fast_path": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = va["X"][:, 1].astype(np.int64) - video_offset
        return {
            "X_train": tr["X"].astype(np.int64),
            "y_train": tr["y"].astype(np.float32),
            "train_users": np.asarray(tr["user"]),
            "X_val": va["X"].astype(np.int64),
            "y_val": va["y"].astype(np.float32),
            "val_users": np.asarray(va["user"]),
            "val_videos": val_videos,
            "field_dims": field_dims,
            "fast_path": True,
        }
    return load_csv_data(data_dir)


def get_evaluator(fast_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def predict_centered(model, x_val, val_users, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x_val), 65536):
            xb = torch.as_tensor(
                x_val[start:start + 65536], dtype=torch.long, device=device
            )
            parts.append(model(xb).detach().cpu().numpy())
    raw = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return center_numpy_scores(raw, val_users, model.bias.detach().cpu().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast_path"])

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    batches = make_complete_user_batches(data["train_users"], 8192)
    rng = np.random.RandomState(args.seed)

    best_primary = -np.inf
    best_scores = None
    best_state = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        batch_order = rng.permutation(len(batches))
        loss_sum = 0.0
        example_count = 0
        for batch_number in batch_order:
            indices = batches[int(batch_number)]
            xb = torch.as_tensor(data["X_train"][indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["y_train"][indices], dtype=torch.float32, device=device)
            ub_np = np.asarray(data["train_users"])[indices]
            _, ub_inverse = np.unique(ub_np, return_inverse=True)
            ub = torch.as_tensor(ub_inverse, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            raw_logits = model(xb)
            logits = centered_logits(raw_logits, ub, model.bias)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            count = len(indices)
            loss_sum += float(loss.detach().cpu().item()) * count
            example_count += count

        scores = predict_centered(model, data["X_val"], data["val_users"], device)
        metrics = evaluate(data["val_users"], data["y_val"].astype(int), scores)
        values = metric_values(metrics)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(example_count, 1), 6),
            "val_gauc": round(values["gauc"], 6),
            "val_ndcg5": round(values["ndcg5"], 6),
            "val_primary": round(values["primary"], 6),
        })

        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_scores is None:
        best_scores = predict_centered(model, data["X_val"], data["val_users"], device)

    final_metrics = metric_values(
        evaluate(data["val_users"], data["y_val"].astype(int), best_scores)
    )
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "config": {
            "method": "gauge-fixed-bce",
            "embedding_dim": 16,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_target_size": 8192,
            "complete_user_slates": True,
            "best_epoch": int(np.argmax([h["val_primary"] for h in history]) + 1),
            "seed": args.seed,
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([
                i,
                data["val_users"][i],
                data["val_videos"][i],
                format(float(score), ".8g"),
            ])


if __name__ == "__main__":
    main()
