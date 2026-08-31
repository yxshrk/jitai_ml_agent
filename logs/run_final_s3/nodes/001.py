"""Gauge-fixed FM: complete-user-slate centered BCE on the official five fields."""
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def scalar_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    durations = []
    with open(train_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            label = float(row["long_view"])
            train_rows.append((user, video, tab, duration, label))
            durations.append(duration)

    durations_np = np.asarray(durations, dtype=np.float64)
    if len(durations_np):
        edges = np.quantile(durations_np, np.linspace(0.1, 0.9, 9))
        edges = np.maximum.accumulate(edges)
    else:
        edges = np.zeros(9, dtype=np.float64)

    user_values = sorted({r[0] for r in train_rows})
    video_values = sorted({r[1] for r in train_rows})
    tab_values = sorted({r[2] for r in train_rows})
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(user, video, tab, duration):
        values = np.asarray([
            user_map.get(user, 0),
            video_map.get(video, 0),
            0,
            tab_map.get(tab, 0),
            int(np.searchsorted(edges, duration, side="right")),
        ], dtype=np.int64)
        return values + offsets

    Xt = np.empty((len(train_rows), 5), dtype=np.int64)
    yt = np.empty(len(train_rows), dtype=np.float32)
    train_users = []
    for i, (user, video, tab, duration, label) in enumerate(train_rows):
        Xt[i] = encode(user, video, tab, duration)
        yt[i] = label
        train_users.append(scalar_id(user))

    val_features = []
    val_labels = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            val_features.append(encode(user, video, tab, duration))
            val_labels.append(float(row["long_view"]))
            val_users.append(scalar_id(user))
            val_videos.append(scalar_id(video))

    Xv = np.asarray(val_features, dtype=np.int64).reshape(-1, 5)
    return {
        "Xt": Xt,
        "yt": yt,
        "train_users": np.asarray(train_users),
        "Xv": Xv,
        "yv": np.asarray(val_labels, dtype=np.float32),
        "val_users": np.asarray(val_users),
        "val_videos": np.asarray(val_videos),
        "field_dims": field_dims,
        "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        if "video" in va.files:
            val_videos = va["video"]
        else:
            val_videos = va["X"][:, 1].astype(np.int64) - int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_users": tr["user"],
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_users": va["user"],
            "val_videos": val_videos,
            "field_dims": field_dims,
            "npz": True,
        }
    return load_csv_data(data_dir)


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [x.astype(np.int64, copy=False) for x in np.split(order, boundaries)]


def centered_logits(logits, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, group_ids, logits)
    counts = torch.bincount(group_ids, minlength=group_count).to(logits.dtype)
    means = sums / counts.clamp_min(1.0)
    return logits - means[group_ids] + global_bias


def complete_slate_batches(groups, rng, target_size):
    group_order = rng.permutation(len(groups))
    pending = []
    pending_size = 0
    for group_number in group_order:
        group = groups[int(group_number)]
        if pending and pending_size + len(group) > target_size:
            yield pending
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            yield pending
            pending = []
            pending_size = 0
    if pending:
        yield pending


def center_numpy_by_user(raw_scores, users, global_bias):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    result = np.empty_like(raw_scores)
    if len(order) == 0:
        return result
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for j in range(len(boundaries) - 1):
        idx = order[boundaries[j]:boundaries[j + 1]]
        result[idx] = raw_scores[idx] - raw_scores[idx].mean() + global_bias
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs if smoke is None else min(args.epochs, max(1, int(smoke)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data_dir)
    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt_cpu = torch.from_numpy(data["Xt"])
    yt_cpu = torch.from_numpy(data["yt"])
    Xv_cpu = torch.from_numpy(data["Xv"])
    groups = make_user_groups(data["train_users"])

    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    best = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for batch_groups in complete_slate_batches(groups, rng, 8192):
            idx_np = np.concatenate(batch_groups)
            gid_np = np.repeat(
                np.arange(len(batch_groups), dtype=np.int64),
                [len(group) for group in batch_groups],
            )
            idx = torch.from_numpy(idx_np).to(device)
            group_ids = torch.from_numpy(gid_np).to(device)
            xb = Xt_cpu[idx_np].to(device)
            yb = yt_cpu[idx_np].to(device)

            opt.zero_grad(set_to_none=True)
            raw = model(xb)
            fixed = centered_logits(raw, group_ids, len(batch_groups), model.bias)
            loss = bce(fixed, yb)
            loss.backward()
            opt.step()

            loss_sum += float(loss.detach().cpu()) * len(idx_np)
            example_count += len(idx_np)

        model.eval()
        raw_parts = []
        with torch.no_grad():
            for start in range(0, len(Xv_cpu), 65536):
                xb = Xv_cpu[start:start + 65536].to(device)
                raw_parts.append(model(xb).detach().cpu().numpy())
        raw_scores = np.concatenate(raw_parts) if raw_parts else np.empty(0, dtype=np.float32)
        global_bias = float(model.bias.detach().cpu().item())
        scores = center_numpy_by_user(raw_scores, data["val_users"], global_bias)
        metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, example_count), 5),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    final_metrics = evaluate(data["val_users"], data["yv"].astype(int), best_scores)
    result = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_users"][i], data["val_videos"][i], f"{float(score):.9g}"])


if __name__ == "__main__":
    main()
