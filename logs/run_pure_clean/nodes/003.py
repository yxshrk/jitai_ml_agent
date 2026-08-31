"""Regularized DeepFM trained with a hybrid pointwise BCE and within-user BPR objective."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch


class RegularizedDeepFM(torch.nn.Module):
    def __init__(self, total_dim, n_fields, dropout, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_fields * k, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def forward(self, x, return_rows=False):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        deep = self.mlp(e.flatten(1)).squeeze(1)
        logits = self.bias + linear + pair + deep
        if return_rows:
            return logits, e
        return logits


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc"))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5"))),
        "primary": float(metric["primary"]),
    }


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    feature_names = ["user_id", "video_id", "tab", "dur_bucket"]
    train_rows = []
    train_y = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = str(min(63, max(0, int(float(row["duration_ms"])) // 5000)))
            train_rows.append([row["user_id"], row["video_id"], row["tab"], duration])
            train_y.append(float(row["long_view"]))
    val_rows = []
    val_y = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = str(min(63, max(0, int(float(row["duration_ms"])) // 5000)))
            val_rows.append([row["user_id"], row["video_id"], row["tab"], duration])
            val_y.append(float(row["long_view"]))
            val_users.append(row["user_id"])
            val_videos.append(row["video_id"])
    mappings = []
    field_dims = []
    for field_index in range(len(feature_names)):
        values = sorted({row[field_index] for row in train_rows})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        result = np.empty((len(rows), len(feature_names)), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                result[i, j] = mappings[j].get(value, 0) + offsets[j]
        return result

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray(train_y, dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray(val_y, dtype=np.int64),
        "users": np.asarray(val_users),
        "videos": np.asarray(val_videos),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            video_offset = int(field_dims[0])
            return {
                "Xt": tr["X"].astype(np.int64),
                "yt": tr["y"].astype(np.float32),
                "Xv": va["X"].astype(np.int64),
                "yv": va["y"].astype(np.int64),
                "users": va["user"].copy(),
                "videos": va["X"][:, 1].astype(np.int64) - video_offset,
                "field_dims": field_dims,
                "fast": True,
            }
    return load_csv_data(data_dir)


def make_evaluator(fast):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def score_model(model, Xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = torch.from_numpy(Xv[start:start + 65536]).to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def build_user_groups(Xt):
    users = Xt[:, 0]
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [part.astype(np.int64, copy=False) for part in np.split(order, boundaries)]


def make_group_batches(groups, labels, batch_size, pair_cap, rng):
    group_order = rng.permutation(len(groups))
    batches = []
    rows = []
    row_count = 0
    for group_number in group_order:
        group = groups[int(group_number)]
        if rows and row_count + len(group) > batch_size:
            batches.append(rows)
            rows = []
            row_count = 0
        rows.append(group)
        row_count += len(group)
        if row_count >= batch_size:
            batches.append(rows)
            rows = []
            row_count = 0
    if rows:
        batches.append(rows)

    for batch_groups in batches:
        indices = np.concatenate(batch_groups)
        positive_local = []
        negative_local = []
        offset = 0
        for group in batch_groups:
            group_labels = labels[group]
            positives = np.flatnonzero(group_labels > 0.5)
            negatives = np.flatnonzero(group_labels <= 0.5)
            if len(positives) and len(negatives):
                pair_count = min(pair_cap, max(len(positives), len(negatives)))
                pos_choice = rng.choice(positives, size=pair_count, replace=len(positives) < pair_count)
                neg_choice = rng.choice(negatives, size=pair_count, replace=len(negatives) < pair_count)
                positive_local.append(pos_choice.astype(np.int64) + offset)
                negative_local.append(neg_choice.astype(np.int64) + offset)
            offset += len(group)
        if positive_local:
            pos = np.concatenate(positive_local)
            neg = np.concatenate(negative_local)
        else:
            pos = np.empty(0, dtype=np.int64)
            neg = np.empty(0, dtype=np.int64)
        yield indices, pos, neg


def train_model(data, groups, config, seed, epochs, device, evaluate_fn, track_epochs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RegularizedDeepFM(
        int(data["field_dims"].sum()),
        int(data["Xt"].shape[1]),
        float(config["dropout"]),
    ).to(device)
    dense_params = list(model.mlp.parameters())
    sparse_params = [model.emb.weight, model.lin.weight, model.bias]
    optimizer = torch.optim.AdamW(
        [
            {"params": sparse_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": float(config["weight_decay"])},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config["lr_gamma"])
    )
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = data["Xt"]
    yt = data["yt"]
    batch_size = 8192
    rng = np.random.RandomState(seed + 1709)
    best_primary = -1.0
    best_scores = None
    best_metric = None
    patience = 0
    epoch_history = []
    last_pointwise = 0.0
    last_pairwise = 0.0
    last_total = 0.0
    epochs_ran = 0
    for epoch in range(epochs):
        model.train()
        pointwise_sum = 0.0
        pairwise_sum = 0.0
        total_sum = 0.0
        batches = 0
        paired_batches = 0
        for indices, pos_local, neg_local in make_group_batches(
            groups, yt, batch_size, int(config["pair_cap"]), rng
        ):
            xb = torch.from_numpy(Xt[indices]).to(device)
            yb = torch.from_numpy(yt[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, accessed_rows = model(xb, return_rows=True)
            pointwise_loss = bce(logits, yb)
            if len(pos_local):
                pos_tensor = torch.from_numpy(pos_local).to(device)
                neg_tensor = torch.from_numpy(neg_local).to(device)
                pairwise_loss = torch.nn.functional.softplus(
                    -(logits[pos_tensor] - logits[neg_tensor])
                ).mean()
                paired_batches += 1
            else:
                pairwise_loss = logits.sum() * 0.0
            row_penalty = accessed_rows.square().sum(dim=(1, 2)).mean()
            loss = (
                pointwise_loss
                + float(config["bpr_weight"]) * pairwise_loss
                + float(config["row_l2"]) * row_penalty
            )
            loss.backward()
            optimizer.step()
            pointwise_sum += float(pointwise_loss.detach().cpu())
            if len(pos_local):
                pairwise_sum += float(pairwise_loss.detach().cpu())
            total_sum += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        epochs_ran = epoch + 1
        last_pointwise = pointwise_sum / max(1, batches)
        last_pairwise = pairwise_sum / max(1, paired_batches)
        last_total = total_sum / max(1, batches)
        if track_epochs:
            scores = score_model(model, data["Xv"], device)
            metric = metric_values(evaluate_fn(data["users"], data["yv"], scores))
            epoch_history.append({
                "epoch": epoch + 1,
                "pointwise_loss": round(last_pointwise, 6),
                "pairwise_loss": round(last_pairwise, 6),
                "total_loss": round(last_total, 6),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 9),
                "val_gauc": round(metric["gauc"], 6),
                "val_ndcg5": round(metric["ndcg5"], 6),
                "val_primary": round(metric["primary"], 6),
            })
            if metric["primary"] > best_primary + 1e-7:
                best_primary = metric["primary"]
                best_scores = scores.copy()
                best_metric = metric
                patience = 0
            else:
                patience += 1
                if patience >= 3 and epoch + 1 >= 6:
                    break
    if not track_epochs:
        scores = score_model(model, data["Xv"], device)
        best_metric = metric_values(evaluate_fn(data["users"], data["yv"], scores))
        best_primary = best_metric["primary"]
        best_scores = scores
    return {
        "scores": best_scores,
        "metric": best_metric,
        "primary": best_primary,
        "pointwise_loss": last_pointwise,
        "pairwise_loss": last_pairwise,
        "total_loss": last_total,
        "epochs_ran": epochs_ran,
        "epoch_history": epoch_history,
    }


def base_candidate_configs(seed, count):
    hand = [
        (0.10, 1e-5, 1e-5, 0.97),
        (0.15, 3e-5, 3e-5, 0.96),
        (0.20, 1e-4, 1e-4, 0.95),
        (0.25, 3e-4, 1e-4, 0.94),
        (0.30, 1e-4, 3e-4, 0.92),
        (0.35, 3e-4, 3e-4, 0.90),
        (0.20, 3e-5, 1e-3, 0.96),
        (0.30, 1e-5, 1e-4, 0.97),
        (0.40, 1e-4, 1e-4, 0.95),
        (0.15, 1e-3, 3e-5, 0.93),
        (0.25, 3e-4, 1e-3, 0.97),
        (0.35, 1e-5, 3e-4, 0.95),
    ]
    configs = []
    for dropout, row_l2, weight_decay, gamma in hand:
        configs.append({
            "dropout": dropout,
            "row_l2": row_l2,
            "weight_decay": weight_decay,
            "lr_gamma": gamma,
        })
    rng = np.random.RandomState(seed + 913)
    while len(configs) < count:
        configs.append({
            "dropout": float(rng.uniform(0.06, 0.46)),
            "row_l2": float(10.0 ** rng.uniform(-6.0, -2.5)),
            "weight_decay": float(10.0 ** rng.uniform(-6.0, -2.3)),
            "lr_gamma": float(rng.uniform(0.86, 0.99)),
        })
    return configs[:count]


def candidate_configs(seed, count):
    weights = [0.04, 0.10, 0.25, 0.60]
    base_count = (count + len(weights) - 1) // len(weights)
    bases = base_candidate_configs(seed, base_count)
    configs = []
    for base in bases:
        for weight in weights:
            config = dict(base)
            config["bpr_weight"] = weight
            config["pair_cap"] = 24
            configs.append(config)
    return configs[:count]


def clean_config(config):
    result = {}
    for key, value in config.items():
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        else:
            result[key] = round(float(value), 10)
    return result


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    data = load_data(args.data_dir)
    evaluate_fn = make_evaluator(data["fast"])
    groups = build_user_groups(data["Xt"])

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_value) if smoke_value is not None else None
    if smoke_epochs is not None:
        probe_count = 2
        final_count = 1
        probe_epochs = max(1, min(1, smoke_epochs))
        final_epochs = max(1, min(args.epochs, smoke_epochs))
    else:
        probe_count = 200 if device.type == "cuda" else 80
        final_count = 6 if device.type == "cuda" else 4
        probe_epochs = 10 if device.type == "cuda" else 8
        final_epochs = args.epochs

    configs = candidate_configs(args.seed, probe_count)
    probe_history = []
    winning_config = None
    winning_probe_primary = -1.0
    for index, config in enumerate(configs):
        result = train_model(
            data, groups, config, args.seed, probe_epochs, device, evaluate_fn, False
        )
        record = {
            "phase": "probe",
            "probe": index + 1,
            "seed": args.seed,
            "epochs": probe_epochs,
            "config": clean_config(config),
            "gauc": round(result["metric"]["gauc"], 6),
            "ndcg5": round(result["metric"]["ndcg5"], 6),
            "primary": round(result["primary"], 6),
            "pointwise_loss": round(result["pointwise_loss"], 6),
            "pairwise_loss": round(result["pairwise_loss"], 6),
            "total_loss": round(result["total_loss"], 6),
        }
        probe_history.append(record)
        append_progress(progress_path, record)
        if result["primary"] > winning_probe_primary:
            winning_probe_primary = result["primary"]
            winning_config = dict(config)

    final_history = []
    best_final = None
    for member in range(final_count):
        final_seed = args.seed + member * 1009
        result = train_model(
            data, groups, winning_config, final_seed, final_epochs,
            device, evaluate_fn, True
        )
        record = {
            "phase": "final",
            "run": member + 1,
            "seed": final_seed,
            "epochs_ran": result["epochs_ran"],
            "config": clean_config(winning_config),
            "gauc": round(result["metric"]["gauc"], 6),
            "ndcg5": round(result["metric"]["ndcg5"], 6),
            "primary": round(result["primary"], 6),
            "epochs": result["epoch_history"],
        }
        final_history.append(record)
        append_progress(progress_path, {
            "phase": "final",
            "run": member + 1,
            "seed": final_seed,
            "primary": round(result["primary"], 6),
        })
        if best_final is None or result["primary"] > best_final["primary"]:
            best_final = result

    final_metric = best_final["metric"]
    metrics = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "winning_config": clean_config(winning_config),
        "winning_probe_primary": winning_probe_primary,
        "history": {
            "probes": probe_history,
            "final_runs": final_history,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_final["scores"]):
            writer.writerow([
                i,
                data["users"][i],
                data["videos"][i],
                format(float(score), ".9g"),
            ])


if __name__ == "__main__":
    main()
