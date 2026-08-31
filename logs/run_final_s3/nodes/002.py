"""User-centered FM with cosine cyclic-LR snapshot score ensembling."""
import argparse
import csv
import itertools
import json
import math
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


def rank_by_user(scores, users):
    result = np.empty(len(scores), dtype=np.float64)
    for idx in make_user_groups(users):
        n = len(idx)
        if n <= 1:
            result[idx] = 0.5
            continue
        local = np.asarray(scores[idx], dtype=np.float64)
        order = np.argsort(local, kind="stable")
        sorted_values = local[order]
        ranks = np.empty(n, dtype=np.float64)
        start = 0
        while start < n:
            end = start + 1
            while end < n and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        result[idx] = ranks / float(n - 1)
    return result


def predict(model, Xv_cpu, users):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(Xv_cpu), 65536):
            xb = Xv_cpu[start:start + 65536].to(next(model.parameters()).device)
            parts.append(model(xb).detach().cpu().numpy())
    raw = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    bias = float(model.bias.detach().cpu().item())
    centered = center_numpy_by_user(raw, users, bias)
    return centered, rank_by_user(centered, users)


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


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
    max_lr = 1e-3
    min_lr = 1e-5
    opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    cycle_count = min(4, epochs)
    cycle_ends = np.asarray([
        int(math.ceil((i + 1) * epochs / float(cycle_count)))
        for i in range(cycle_count)
    ], dtype=np.int64)
    cycle_starts = np.r_[1, cycle_ends[:-1] + 1]

    history = []
    snapshots = []
    snapshot_ranks = []

    for epoch in range(1, epochs + 1):
        cycle_id = int(np.searchsorted(cycle_ends, epoch, side="left"))
        cycle_start = int(cycle_starts[cycle_id])
        cycle_end = int(cycle_ends[cycle_id])
        cycle_epochs = cycle_end - cycle_start + 1
        batches = list(complete_slate_batches(groups, rng, 8192))

        model.train()
        loss_sum = 0.0
        example_count = 0
        for batch_number, batch_groups in enumerate(batches):
            within_epoch = (batch_number + 1) / float(max(1, len(batches)))
            cycle_progress = ((epoch - cycle_start) + within_epoch) / float(cycle_epochs)
            cycle_progress = min(1.0, max(0.0, cycle_progress))
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * cycle_progress))
            for param_group in opt.param_groups:
                param_group["lr"] = lr

            idx_np = np.concatenate(batch_groups)
            gid_np = np.repeat(
                np.arange(len(batch_groups), dtype=np.int64),
                [len(group) for group in batch_groups],
            )
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

        scores, ranked_scores = predict(model, Xv_cpu, data["val_users"])
        metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
        gauc, ndcg5, primary = metric_values(metrics)
        is_snapshot = epoch == cycle_end
        history.append({
            "epoch": epoch,
            "cycle": cycle_id + 1,
            "end_lr": round(float(opt.param_groups[0]["lr"]), 9),
            "train_loss": round(loss_sum / max(1, example_count), 5),
            "val_gauc": round(gauc, 6),
            "val_ndcg5": round(ndcg5, 6),
            "val_primary": round(primary, 6),
            "snapshot": is_snapshot,
        })
        if is_snapshot:
            snapshots.append({
                "snapshot": len(snapshots),
                "epoch": epoch,
                "gauc": gauc,
                "ndcg5": ndcg5,
                "primary": primary,
            })
            snapshot_ranks.append(ranked_scores)

    if not snapshot_ranks:
        scores, ranked_scores = predict(model, Xv_cpu, data["val_users"])
        metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
        gauc, ndcg5, primary = metric_values(metrics)
        snapshots.append({
            "snapshot": 0,
            "epoch": epochs,
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
        })
        snapshot_ranks.append(ranked_scores)

    ensemble_history = []
    best_primary = -1.0
    best_subset = None
    best_scores = None
    snapshot_count = len(snapshot_ranks)
    for subset_size in range(1, snapshot_count + 1):
        for subset in itertools.combinations(range(snapshot_count), subset_size):
            ensemble_scores = np.mean(
                np.stack([snapshot_ranks[i] for i in subset], axis=0), axis=0
            )
            metrics = evaluate(
                data["val_users"], data["yv"].astype(int), ensemble_scores
            )
            gauc, ndcg5, primary = metric_values(metrics)
            ensemble_history.append({
                "snapshots": list(subset),
                "epochs": [int(snapshots[i]["epoch"]) for i in subset],
                "gauc": gauc,
                "ndcg5": ndcg5,
                "primary": primary,
            })
            if primary > best_primary + 1e-12 or (
                abs(primary - best_primary) <= 1e-12
                and (best_subset is None or len(subset) > len(best_subset))
            ):
                best_primary = primary
                best_subset = subset
                best_scores = ensemble_scores.copy()

    os.makedirs(args.out_dir, exist_ok=True)
    final_metrics = evaluate(data["val_users"], data["yv"].astype(int), best_scores)
    final_gauc, final_ndcg5, final_primary = metric_values(final_metrics)
    result = {
        "gauc": final_gauc,
        "ndcg5": final_ndcg5,
        "primary": final_primary,
        "selected_snapshots": list(best_subset),
        "selected_epochs": [int(snapshots[i]["epoch"]) for i in best_subset],
        "history": history,
        "snapshots": snapshots,
        "ensemble_history": ensemble_history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([
                i,
                data["val_users"][i],
                data["val_videos"][i],
                f"{float(score):.9g}",
            ])


if __name__ == "__main__":
    main()
