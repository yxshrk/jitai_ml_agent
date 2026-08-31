import argparse
import csv
import datetime
import itertools
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
    def __init__(self, total_dim, k=16, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        s = e.sum(dim=1)
        pair = 0.5 * (s * s - (e * e).sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, n_fields=5, k=16, dropout=0.0):
        super().__init__()
        width = n_fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.input_dropout = torch.nn.Dropout(dropout)
        self.cross_weight = torch.nn.Parameter(torch.empty(width))
        self.cross_bias = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1, bias=False),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_weight, std=0.01)
        torch.nn.init.normal_(self.cross_out.weight, std=0.01)
        torch.nn.init.normal_(self.mlp[-1].weight, std=0.01)

    def forward(self, x):
        z0 = self.input_dropout(self.emb(x)).flatten(1)
        scale = (z0 * self.cross_weight).sum(dim=1, keepdim=True)
        z1 = z0 * scale + self.cross_bias + z0
        linear = self.lin(x).sum(dim=(1, 2))
        return self.bias + linear + self.cross_out(z1).squeeze(1) + self.mlp(z0).squeeze(1)


def centered_logits(raw_logits, user_ids, global_bias):
    _, inverse, counts = torch.unique_consecutive(
        user_ids, return_inverse=True, return_counts=True
    )
    sums = torch.zeros(counts.numel(), device=raw_logits.device, dtype=raw_logits.dtype)
    sums.scatter_add_(0, inverse, raw_logits)
    means = sums / counts.to(raw_logits.dtype)
    return raw_logits - means[inverse] + global_bias


def center_numpy_scores(raw_scores, users, global_bias):
    users = np.asarray(users)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    _, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=raw_scores)
    means = sums / np.maximum(counts, 1.0)
    return (raw_scores - means[inverse] + float(global_bias)).astype(np.float32)


def make_complete_user_batches(users, target_size):
    users = np.asarray(users)
    order = np.argsort(users, kind="stable")
    if len(order) == 0:
        return []
    sorted_users = users[order]
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


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            if training:
                item["date"] = row.get("date", "")
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
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = categorical_mapping([r["user_id"] for r in train_rows])
    video_map = categorical_mapping([r["video_id"] for r in train_rows])
    tab_map = categorical_mapping([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    thresholds = (
        np.quantile(durations, np.arange(1, 10) / 10.0)
        if len(durations)
        else np.zeros(9, dtype=np.float64)
    )
    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows, training):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        labels = np.zeros(len(rows), dtype=np.float32)
        users = []
        videos = []
        dates = []
        for i, row in enumerate(rows):
            u = user_map.get(row["user_id"], len(user_map))
            v = video_map.get(row["video_id"], len(video_map))
            t = tab_map.get(row["tab"], len(tab_map))
            d = int(np.searchsorted(thresholds, row["duration_ms"], side="right"))
            x[i] = np.asarray([u, v, 0, t, d], dtype=np.int64) + offsets
            labels[i] = row["long_view"]
            users.append(row["user_id"])
            videos.append(row["video_id"])
            if training:
                dates.append(row.get("date", ""))
        return (
            x,
            labels,
            np.asarray(users, dtype=object),
            np.asarray(videos, dtype=object),
            np.asarray(dates, dtype=object),
        )

    xt, yt, train_users, _, train_dates = encode(train_rows, True)
    xv, yv, val_users, val_videos, _ = encode(val_rows, False)
    return {
        "X_train": xt,
        "y_train": yt,
        "train_users": train_users,
        "train_dates": train_dates,
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
        return {
            "X_train": tr["X"].astype(np.int64),
            "y_train": tr["y"].astype(np.float32),
            "train_users": np.asarray(tr["user"]),
            "train_dates": np.asarray(tr["date"]) if "date" in tr.files else np.zeros(len(tr["y"])),
            "X_val": va["X"].astype(np.int64),
            "y_val": va["y"].astype(np.float32),
            "val_users": np.asarray(va["user"]),
            "val_videos": va["X"][:, 1].astype(np.int64) - video_offset,
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


def date_to_ordinal(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.datetime.strptime(digits[:8], "%Y%m%d").date().toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def make_recency_weights(dates):
    ordinals = np.asarray([date_to_ordinal(v) for v in dates], dtype=np.int64)
    valid = ordinals > 0
    weights = np.ones(len(ordinals), dtype=np.float32)
    if np.any(valid):
        latest = int(ordinals[valid].max())
        age = np.maximum(latest - ordinals, 0)
        weights[valid] = np.power(0.5, age[valid] / 7.0).astype(np.float32)
        weights /= max(float(weights.mean()), 1e-6)
    return weights


def build_model(config, total_dim, device):
    dropout = 0.30 if config["regularization"] == "strong" else 0.05
    if config["architecture"] == "fm":
        model = FM(total_dim, k=16, dropout=dropout)
    else:
        model = DCNLite(total_dim, n_fields=5, k=16, dropout=dropout)
    return model.to(device)


def predict_centered(model, x_val, val_users, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x_val), 65536):
            xb = torch.as_tensor(x_val[start:start + 65536], dtype=torch.long, device=device)
            parts.append(model(xb).detach().cpu().numpy())
    raw = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return center_numpy_scores(raw, val_users, model.bias.detach().cpu().item())


def make_pair_positions(user_codes, labels, rng):
    positive_positions = []
    negative_positions = []
    if len(labels) == 0:
        return positive_positions, negative_positions
    boundaries = np.flatnonzero(user_codes[1:] != user_codes[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(labels)]))
    for start, end in zip(starts, ends):
        local = np.arange(start, end)
        pos = local[labels[start:end] > 0.5]
        neg = local[labels[start:end] <= 0.5]
        if len(pos) and len(neg):
            positive_positions.extend(pos.tolist())
            negative_positions.extend(rng.choice(neg, size=len(pos), replace=True).tolist())
    return positive_positions, negative_positions


def train_candidate(config, data, batches, recency_weights, evaluate, device, seed, epochs):
    seed_everything(seed)
    model = build_model(config, int(data["field_dims"].sum()), device)
    if config["regularization"] == "strong":
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = None
    rng = np.random.RandomState(seed + 913)
    best_primary = -np.inf
    best_scores = None
    best_state = None
    curve = []
    train_users = np.asarray(data["train_users"])
    use_recency = config["weighting"] == "recency-7d"
    use_hybrid = config["loss"] == "bpr-hybrid"

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(batches))
        halves = np.array_split(order, 2)
        epoch_loss = 0.0
        epoch_count = 0
        for half_number, half in enumerate(halves):
            for batch_number in half:
                indices = batches[int(batch_number)]
                xb = torch.as_tensor(data["X_train"][indices], dtype=torch.long, device=device)
                y_np = data["y_train"][indices]
                yb = torch.as_tensor(y_np, dtype=torch.float32, device=device)
                ub_np = train_users[indices]
                _, ub_inverse = np.unique(ub_np, return_inverse=True)
                ub = torch.as_tensor(ub_inverse, dtype=torch.long, device=device)
                if use_recency:
                    wb = torch.as_tensor(recency_weights[indices], dtype=torch.float32, device=device)
                else:
                    wb = torch.ones(len(indices), dtype=torch.float32, device=device)

                optimizer.zero_grad(set_to_none=True)
                raw = model(xb)
                logits = centered_logits(raw, ub, model.bias)
                point_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yb, reduction="none"
                )
                point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1.0)
                loss = point_loss
                if use_hybrid:
                    pos_idx, neg_idx = make_pair_positions(ub_inverse, y_np, rng)
                    if pos_idx:
                        pt = torch.as_tensor(pos_idx, dtype=torch.long, device=device)
                        nt = torch.as_tensor(neg_idx, dtype=torch.long, device=device)
                        pair_losses = torch.nn.functional.softplus(-(raw[pt] - raw[nt]))
                        pair_weights = 0.5 * (wb[pt] + wb[nt])
                        pair_loss = (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1.0)
                        loss = 0.5 * point_loss + 0.5 * pair_loss
                loss.backward()
                optimizer.step()
                count = len(indices)
                epoch_loss += float(loss.detach().cpu().item()) * count
                epoch_count += count

            scores = predict_centered(model, data["X_val"], data["val_users"], device)
            values = metric_values(evaluate(data["val_users"], data["y_val"].astype(int), scores))
            curve.append({
                "epoch_fraction": float(epoch + (half_number + 1) / 2.0),
                "train_loss": round(epoch_loss / max(epoch_count, 1), 6),
                "val_gauc": round(values["gauc"], 6),
                "val_ndcg5": round(values["ndcg5"], 6),
                "val_primary": round(values["primary"], 6),
            })
            if values["primary"] > best_primary + 1e-7:
                best_primary = values["primary"]
                best_scores = scores.copy()
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            model.train()
        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_scores is None:
        best_scores = predict_centered(model, data["X_val"], data["val_users"], device)
        best_primary = metric_values(
            evaluate(data["val_users"], data["y_val"].astype(int), best_scores)
        )["primary"]
    return model, best_scores, float(best_primary), curve


def config_dict(values):
    return {
        "architecture": values[0],
        "loss": values[1],
        "weighting": values[2],
        "regularization": values[3],
    }


def config_key(config):
    return "|".join([
        config["architecture"],
        config["loss"],
        config["weighting"],
        config["regularization"],
    ])


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w"):
        pass

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast_path"])
    batches = make_complete_user_batches(data["train_users"], 8192)
    recency_weights = make_recency_weights(data["train_dates"])

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = None if smoke_value is None else max(1, int(smoke_value))
    probe_epochs = 2 if smoke_cap is None else min(2, smoke_cap)
    refine_epochs = 5 if smoke_cap is None else min(5, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)

    all_configs = [
        config_dict(values)
        for values in itertools.product(
            ["fm", "dcn-lite"],
            ["logloss", "bpr-hybrid"],
            ["uniform", "recency-7d"],
            ["mild", "strong"],
        )
    ]
    if smoke_cap is not None:
        selected_indices = [0, 5, 10, 15]
        matrix_configs = [all_configs[i] for i in selected_indices]
        repetitions = 1
    else:
        matrix_configs = all_configs
        repetitions = 3

    history = []
    grouped_scores = {config_key(config): [] for config in matrix_configs}
    config_lookup = {config_key(config): config for config in matrix_configs}

    for repetition in range(repetitions):
        probe_seed = args.seed + repetition * 1009
        for config in matrix_configs:
            model, scores, primary, curve = train_candidate(
                config, data, batches, recency_weights, evaluate, device,
                probe_seed, probe_epochs
            )
            record = {
                "phase": "matrix_probe",
                "repetition": repetition,
                "seed": probe_seed,
                "epochs": probe_epochs,
                "config": dict(config),
                "best_primary": primary,
                "curve": curve,
            }
            history.append(record)
            grouped_scores[config_key(config)].append(primary)
            append_progress(progress_path, {
                "phase": "matrix_probe",
                "config": config,
                "repetition": repetition,
                "primary": primary,
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summaries = []
    for key, scores in grouped_scores.items():
        summaries.append({
            "config": dict(config_lookup[key]),
            "mean_primary": float(np.mean(scores)),
            "std_primary": float(np.std(scores)),
            "replicate_scores": [float(v) for v in scores],
        })
    summaries.sort(key=lambda item: item["mean_primary"], reverse=True)

    if smoke_cap is None:
        refinement = []
        for rank, summary in enumerate(summaries[:4]):
            config = summary["config"]
            refine_seed = args.seed + 5003 + rank * 271
            model, scores, primary, curve = train_candidate(
                config, data, batches, recency_weights, evaluate, device,
                refine_seed, refine_epochs
            )
            record = {
                "phase": "refinement",
                "rank_from_matrix": rank + 1,
                "seed": refine_seed,
                "epochs": refine_epochs,
                "config": dict(config),
                "best_primary": primary,
                "curve": curve,
            }
            history.append(record)
            refinement.append((primary, config))
            append_progress(progress_path, {
                "phase": "refinement",
                "config": config,
                "primary": primary,
            })
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        refinement.sort(key=lambda item: item[0], reverse=True)
        winning_config = dict(refinement[0][1])
    else:
        winning_config = dict(summaries[0]["config"])

    final_model, final_scores, _, final_curve = train_candidate(
        winning_config, data, batches, recency_weights, evaluate, device,
        args.seed, final_epochs
    )
    final_metrics = metric_values(
        evaluate(data["val_users"], data["y_val"].astype(int), final_scores)
    )
    append_progress(progress_path, {
        "phase": "final",
        "config": winning_config,
        "primary": final_metrics["primary"],
    })

    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "matrix_summary": summaries,
        "final_history": final_curve,
        "config": {
            "method": "cross-stage-factorial-matrix",
            "winning_config": winning_config,
            "matrix_cells": len(matrix_configs),
            "matrix_repetitions": repetitions,
            "probe_epochs": probe_epochs,
            "refinement_epochs": refine_epochs,
            "final_epochs": final_epochs,
            "half_epoch_checkpointing": True,
            "embedding_dim": 16,
            "batch_target_size": 8192,
            "seed": args.seed,
        },
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([
                i,
                data["val_users"][i],
                data["val_videos"][i],
                format(float(score), ".8g"),
            ])


if __name__ == "__main__":
    main()
