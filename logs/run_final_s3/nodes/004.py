"""Two-stage dial search for a regularized DCN-lite centered-BCE/BPR package."""
import argparse
import csv
import datetime
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, cross_layers=2, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        input_dim = fields * k
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(input_dim)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)
        ])
        self.deep1 = torch.nn.Linear(input_dim, hidden)
        self.deep2 = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(input_dim + hidden, 1)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        torch.nn.init.xavier_uniform_(self.deep1.weight)
        torch.nn.init.zeros_(self.deep1.bias)
        torch.nn.init.xavier_uniform_(self.deep2.weight)
        torch.nn.init.zeros_(self.deep2.bias)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        deep = self.dropout(torch.relu(self.deep1(x0)))
        deep = self.dropout(torch.relu(self.deep2(deep)))
        residual = self.head(torch.cat((cross, deep), dim=1)).squeeze(1)
        return self.bias + linear + pair + residual


def scalar_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def date_ordinal(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_csv_data(data_dir):
    train_rows = []
    durations = []
    with open(os.path.join(data_dir, "train.csv"), "r", newline="") as fh:
        for row in csv.DictReader(fh):
            duration = float(row["duration_ms"])
            train_rows.append((row["user_id"], row["video_id"], row["tab"], duration,
                               float(row["long_view"]), date_ordinal(row["date"])))
            durations.append(duration)

    duration_array = np.asarray(durations, dtype=np.float64)
    edges = (np.maximum.accumulate(np.quantile(duration_array, np.linspace(0.1, 0.9, 9)))
             if len(duration_array) else np.zeros(9, dtype=np.float64))
    user_values = sorted({r[0] for r in train_rows})
    video_values = sorted({r[1] for r in train_rows})
    tab_values = sorted({r[2] for r in train_rows})
    user_map = {v: i + 1 for i, v in enumerate(user_values)}
    video_map = {v: i + 1 for i, v in enumerate(video_values)}
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                             len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(user, video, tab, duration):
        values = np.asarray([user_map.get(user, 0), video_map.get(video, 0), 0,
                             tab_map.get(tab, 0),
                             int(np.searchsorted(edges, duration, side="right"))], dtype=np.int64)
        return values + offsets

    xt = np.empty((len(train_rows), 5), dtype=np.int64)
    yt = np.empty(len(train_rows), dtype=np.float32)
    train_users = []
    train_dates = np.empty(len(train_rows), dtype=np.int64)
    for i, (user, video, tab, duration, label, date_value) in enumerate(train_rows):
        xt[i] = encode(user, video, tab, duration)
        yt[i] = label
        train_users.append(scalar_id(user))
        train_dates[i] = date_value

    val_features = []
    val_labels = []
    val_users = []
    val_videos = []
    with open(os.path.join(data_dir, "val.csv"), "r", newline="") as fh:
        for row in csv.DictReader(fh):
            duration = float(row["duration_ms"])
            val_features.append(encode(row["user_id"], row["video_id"], row["tab"], duration))
            val_labels.append(float(row["long_view"]))
            val_users.append(scalar_id(row["user_id"]))
            val_videos.append(scalar_id(row["video_id"]))

    return {
        "Xt": xt,
        "yt": yt,
        "train_users": np.asarray(train_users),
        "train_dates": train_dates,
        "Xv": np.asarray(val_features, dtype=np.int64).reshape(-1, 5),
        "yv": np.asarray(val_labels, dtype=np.float32),
        "val_users": np.asarray(val_users),
        "val_videos": np.asarray(val_videos),
        "field_dims": field_dims,
        "npz": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr = np.load(train_path)
        va = np.load(val_path)
        field_dims = tr["field_dims"].astype(np.int64)
        val_videos = va["video"] if "video" in va.files else va["X"][:, 1].astype(np.int64) - int(field_dims[0])
        dates = tr["date"] if "date" in tr.files else np.zeros(len(tr["y"]), dtype=np.int64)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_users": tr["user"],
            "train_dates": np.asarray([date_ordinal(x) for x in dates], dtype=np.int64),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_users": va["user"],
            "val_videos": val_videos,
            "field_dims": field_dims,
            "npz": True,
        }
    return load_csv_data(data_dir)


def make_group_records(users, labels):
    order = np.argsort(users, kind="stable")
    if len(order) == 0:
        return []
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    records = []
    for indices in np.split(order, boundaries):
        indices = indices.astype(np.int64, copy=False)
        local_labels = labels[indices]
        pos = np.flatnonzero(local_labels > 0.5).astype(np.int64)
        neg = np.flatnonzero(local_labels <= 0.5).astype(np.int64)
        records.append((indices, pos, neg))
    return records


def complete_slate_batches(groups, rng, target_size):
    pending = []
    pending_size = 0
    for group_number in rng.permutation(len(groups)):
        group = groups[int(group_number)]
        size = len(group[0])
        if pending and pending_size + size > target_size:
            yield pending
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += size
        if pending_size >= target_size:
            yield pending
            pending = []
            pending_size = 0
    if pending:
        yield pending


def centered_logits(logits, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=logits.dtype, device=logits.device)
    sums.scatter_add_(0, group_ids, logits)
    counts = torch.bincount(group_ids, minlength=group_count).to(logits.dtype)
    return logits - (sums / counts.clamp_min(1.0))[group_ids] + global_bias


def center_numpy_by_user(raw_scores, users, global_bias):
    order = np.argsort(users, kind="stable")
    result = np.empty_like(raw_scores)
    if len(order) == 0:
        return result
    sorted_users = users[order]
    bounds = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        result[idx] = raw_scores[idx] - raw_scores[idx].mean() + global_bias
    return result


def rank_by_user(scores, users):
    result = np.empty(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    if len(order) == 0:
        return result
    sorted_users = users[order]
    bounds = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        local_order = np.argsort(scores[idx], kind="stable")
        ranks = np.empty(len(idx), dtype=np.float64)
        ranks[local_order] = np.arange(len(idx), dtype=np.float64)
        if len(idx) > 1:
            ranks /= float(len(idx) - 1)
        result[idx] = ranks
    return result


def recency_weights(train_dates, half_life):
    if len(train_dates) == 0 or np.max(train_dates) <= 0:
        return np.ones(len(train_dates), dtype=np.float32)
    age = np.maximum(0, int(np.max(train_dates)) - train_dates.astype(np.int64))
    weights = np.exp2(-age.astype(np.float64) / float(half_life))
    weights /= max(weights.mean(), 1e-12)
    return weights.astype(np.float32)


def predict(model, xv_cpu, users, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(xv_cpu), 65536):
            parts.append(model(xv_cpu[start:start + 65536].to(device)).detach().cpu().numpy())
    raw = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return center_numpy_by_user(raw, users, float(model.bias.detach().cpu().item()))


def train_one(config, seed, epochs, data, tensors, groups, evaluate, device, final_run=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(int(data["field_dims"].sum()), dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                                  weight_decay=float(config["weight_decay"]))
    xt_cpu, yt_cpu, xv_cpu = tensors
    weights_cpu = torch.from_numpy(recency_weights(data["train_dates"], config["half_life"]))
    rng = np.random.default_rng(seed)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    checkpoint_history = []
    stale = 0

    for epoch in range(epochs):
        decay_power = epoch // int(config["decay_every"])
        current_lr = float(config["lr"]) * (float(config["decay_gamma"]) ** decay_power)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        model.train()
        batches = list(complete_slate_batches(groups, rng, 8192))
        halfway = max(1, int(math.ceil(len(batches) / 2.0)))
        running_loss = 0.0
        running_count = 0
        stop_training = False
        for batch_number, batch_groups in enumerate(batches, 1):
            idx_np = np.concatenate([g[0] for g in batch_groups])
            sizes = [len(g[0]) for g in batch_groups]
            gid_np = np.repeat(np.arange(len(batch_groups), dtype=np.int64), sizes)
            xb = xt_cpu[idx_np].to(device)
            yb = yt_cpu[idx_np].to(device)
            wb = weights_cpu[idx_np].to(device)
            group_ids = torch.from_numpy(gid_np).to(device)

            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            fixed = centered_logits(raw, group_ids, len(batch_groups), model.bias)
            bce_values = torch.nn.functional.binary_cross_entropy_with_logits(fixed, yb, reduction="none")
            bce_loss = (bce_values * wb).sum() / wb.sum().clamp_min(1e-8)

            pos_local = []
            neg_local = []
            offset = 0
            for indices, pos, neg in batch_groups:
                if len(pos) and len(neg):
                    pair_count = min(max(len(pos), len(neg)), 8)
                    pos_local.extend((offset + rng.choice(pos, size=pair_count, replace=True)).tolist())
                    neg_local.extend((offset + rng.choice(neg, size=pair_count, replace=True)).tolist())
                offset += len(indices)
            if pos_local:
                pos_tensor = torch.as_tensor(pos_local, dtype=torch.long, device=device)
                neg_tensor = torch.as_tensor(neg_local, dtype=torch.long, device=device)
                pair_weights = 0.5 * (wb[pos_tensor] + wb[neg_tensor])
                pair_values = torch.nn.functional.softplus(-(raw[pos_tensor] - raw[neg_tensor]))
                bpr_loss = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
            else:
                bpr_loss = raw.sum() * 0.0
            loss = 0.5 * bce_loss + 0.5 * bpr_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(idx_np)
            running_count += len(idx_np)

            if batch_number == halfway or batch_number == len(batches):
                scores = predict(model, xv_cpu, data["val_users"], device)
                metrics = evaluate(data["val_users"], data["yv"].astype(int), scores)
                primary = float(metrics["primary"])
                checkpoint_history.append({
                    "epoch": epoch + (0.5 if batch_number == halfway and halfway < len(batches) else 1.0),
                    "lr": current_lr,
                    "train_loss": running_loss / max(1, running_count),
                    "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
                    "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
                    "primary": primary,
                })
                if primary > best_primary + 1e-7:
                    best_primary = primary
                    best_scores = scores.copy()
                    best_metrics = metrics
                    stale = 0
                else:
                    stale += 1
                model.train()
                if final_run and stale >= 6 and epoch >= 3:
                    stop_training = True
                    break
        if stop_training:
            break
    return best_primary, best_scores, best_metrics, checkpoint_history


def coarse_configs(rng, count):
    configs = []
    decay_choices = [(0.45, 1), (0.58, 1), (0.70, 1), (0.55, 2), (0.72, 2)]
    half_choices = [3.5, 5.0, 7.0, 10.0, 14.0]
    for _ in range(count):
        gamma, every = decay_choices[int(rng.integers(len(decay_choices)))]
        configs.append({
            "dropout": float(rng.uniform(0.15, 0.40)),
            "weight_decay": float(10.0 ** rng.uniform(math.log10(3e-5), math.log10(3e-3))),
            "lr": float(10.0 ** rng.uniform(math.log10(3.5e-4), math.log10(1.6e-3))),
            "decay_gamma": gamma,
            "decay_every": every,
            "half_life": half_choices[int(rng.integers(len(half_choices)))],
        })
    return configs


def refined_configs(rng, winner, count):
    configs = [dict(winner)]
    for _ in range(max(0, count - 1)):
        configs.append({
            "dropout": float(np.clip(rng.normal(winner["dropout"], 0.035), 0.10, 0.45)),
            "weight_decay": float(np.clip(winner["weight_decay"] * math.exp(rng.normal(0.0, 0.45)), 1e-5, 5e-3)),
            "lr": float(np.clip(winner["lr"] * math.exp(rng.normal(0.0, 0.25)), 2e-4, 2e-3)),
            "decay_gamma": float(np.clip(rng.normal(winner["decay_gamma"], 0.07), 0.35, 0.85)),
            "decay_every": int(winner["decay_every"] if rng.random() < 0.75 else (1 if winner["decay_every"] == 2 else 2)),
            "half_life": float(np.clip(winner["half_life"] * math.exp(rng.normal(0.0, 0.25)), 2.5, 18.0)),
        })
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke = smoke_value is not None
    smoke_epochs = max(1, int(smoke_value)) if smoke else None
    coarse_epochs = min(3, smoke_epochs) if smoke else 3
    refine_epochs = min(5, smoke_epochs) if smoke else 5
    final_epochs = min(args.epochs, smoke_epochs) if smoke else args.epochs
    coarse_count = 2 if smoke else 40
    refine_count = 1 if smoke else 24
    final_seed_count = 1 if smoke else 5

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

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    xt_cpu = torch.from_numpy(data["Xt"])
    yt_cpu = torch.from_numpy(data["yt"])
    xv_cpu = torch.from_numpy(data["Xv"])
    tensors = (xt_cpu, yt_cpu, xv_cpu)
    groups = make_group_records(data["train_users"], data["yt"])
    search_rng = np.random.default_rng(args.seed + 991)
    history = []

    best_config = None
    best_probe_primary = -1.0
    coarse = coarse_configs(search_rng, coarse_count)
    for probe_index, config in enumerate(coarse):
        primary, _, metrics, checkpoints = train_one(
            config, args.seed + 1000 + probe_index, coarse_epochs, data, tensors,
            groups, evaluate, device, final_run=False)
        record = {"stage": "coarse", "probe": probe_index, "config": config,
                  "primary": primary,
                  "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
                  "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
                  "checkpoints": checkpoints}
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "coarse", "probe": probe_index,
                                 "config": config, "primary": primary}) + "\n")
        if primary > best_probe_primary:
            best_probe_primary = primary
            best_config = dict(config)

    refined = refined_configs(search_rng, best_config, refine_count)
    for probe_index, config in enumerate(refined):
        primary, _, metrics, checkpoints = train_one(
            config, args.seed + 5000 + probe_index, refine_epochs, data, tensors,
            groups, evaluate, device, final_run=False)
        record = {"stage": "refine", "probe": probe_index, "config": config,
                  "primary": primary,
                  "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
                  "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
                  "checkpoints": checkpoints}
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "refine", "probe": probe_index,
                                 "config": config, "primary": primary}) + "\n")
        if primary > best_probe_primary:
            best_probe_primary = primary
            best_config = dict(config)

    final_rank_scores = []
    final_histories = []
    for seed_offset in range(final_seed_count):
        final_seed = args.seed + seed_offset
        primary, scores, metrics, checkpoints = train_one(
            best_config, final_seed, final_epochs, data, tensors, groups,
            evaluate, device, final_run=True)
        final_rank_scores.append(rank_by_user(scores, data["val_users"]))
        final_histories.append({
            "seed": final_seed,
            "primary": primary,
            "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
            "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
            "checkpoints": checkpoints,
        })
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({"stage": "final", "seed": final_seed,
                                 "config": best_config, "primary": primary}) + "\n")

    best_scores = np.mean(np.stack(final_rank_scores, axis=0), axis=0)
    final_metrics = evaluate(data["val_users"], data["yv"].astype(int), best_scores)
    result = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "selected_config": best_config,
        "probe_best_primary": best_probe_primary,
        "history": history,
        "final_runs": final_histories,
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
