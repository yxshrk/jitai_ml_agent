"""Regularized DeepFM with normalized exponential training recency weighting."""
import argparse
import csv
import datetime
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


def date_numbers(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").astype(np.int64).astype(np.float64)
    result = np.empty(len(arr), dtype=np.float64)
    failed = False
    for i, value in enumerate(arr):
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        else:
            text = str(value)
        text = text.strip()
        try:
            compact = text.replace("-", "").replace("/", "")
            if len(compact) >= 8 and compact[:8].isdigit():
                d = datetime.datetime.strptime(compact[:8], "%Y%m%d").date()
                result[i] = float(d.toordinal())
            else:
                result[i] = float(text)
        except (ValueError, TypeError, OverflowError):
            failed = True
            break
    if not failed and np.all(np.isfinite(result)):
        return result
    strings = np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in arr
    ])
    unique = sorted(set(strings.tolist()))
    mapping = {value: i for i, value in enumerate(unique)}
    return np.asarray([mapping[value] for value in strings], dtype=np.float64)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    train_y = []
    train_dates = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            duration = str(min(63, max(0, int(float(row["duration_ms"])) // 5000)))
            train_rows.append([row["user_id"], row["video_id"], row["tab"], duration])
            train_y.append(float(row["long_view"]))
            train_dates.append(row["date"])
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
    for field_index in range(4):
        values = sorted({row[field_index] for row in train_rows})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        result = np.empty((len(rows), 4), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                result[i, j] = mappings[j].get(value, 0) + offsets[j]
        return result

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray(train_y, dtype=np.float32),
        "train_dates": date_numbers(train_dates),
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
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_dates": date_numbers(tr["date"]),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.int64),
            "users": va["user"],
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


def recency_weights(train_dates, half_life):
    if half_life is None or float(half_life) <= 0.0:
        weights = np.ones(len(train_dates), dtype=np.float32)
    else:
        dates = np.asarray(train_dates, dtype=np.float64)
        age = np.maximum(0.0, np.nanmax(dates) - dates)
        exponent = np.maximum(-30.0, -np.log(2.0) * age / float(half_life))
        weights = np.exp(exponent).astype(np.float64)
        weights /= max(float(weights.mean()), 1e-12)
        weights = weights.astype(np.float32)
    total = float(np.sum(weights, dtype=np.float64))
    squared = float(np.sum(weights.astype(np.float64) ** 2))
    ess = total * total / max(squared, 1e-12)
    return weights, ess / max(1, len(weights))


def train_model(data, config, seed, epochs, device, evaluate_fn, track_epochs):
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
    Xt = data["Xt"]
    yt = data["yt"]
    row_weights, ess_ratio = recency_weights(data["train_dates"], config.get("half_life"))
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1709)
    best_primary = -1.0
    best_scores = None
    best_metric = None
    patience = 0
    epoch_history = []
    last_loss = 0.0
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator).numpy()
        loss_sum = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(Xt[indices]).to(device)
            yb = torch.from_numpy(yt[indices]).to(device)
            wb = torch.from_numpy(row_weights[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, accessed_rows = model(xb, return_rows=True)
            per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            prediction_loss = (per_row * wb).sum() / wb.sum().clamp_min(1e-8)
            row_penalty = accessed_rows.square().sum(dim=(1, 2)).mean()
            loss = prediction_loss + float(config["row_l2"]) * row_penalty
            loss.backward()
            optimizer.step()
            loss_sum += float(prediction_loss.detach().cpu())
            batches += 1
        scheduler.step()
        last_loss = loss_sum / max(1, batches)
        if track_epochs:
            scores = score_model(model, data["Xv"], device)
            metric = metric_values(evaluate_fn(data["users"], data["yv"], scores))
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 6),
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
        "train_loss": last_loss,
        "epochs_ran": len(epoch_history) if track_epochs else epochs,
        "epoch_history": epoch_history,
        "ess_ratio": ess_ratio,
    }


def candidate_configs(seed, count):
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
            "half_life": 0.0,
        })
    rng = np.random.RandomState(seed + 913)
    while len(configs) < count:
        configs.append({
            "dropout": float(rng.uniform(0.06, 0.46)),
            "row_l2": float(10.0 ** rng.uniform(-6.0, -2.5)),
            "weight_decay": float(10.0 ** rng.uniform(-6.0, -2.3)),
            "lr_gamma": float(rng.uniform(0.86, 0.99)),
            "half_life": 0.0,
        })
    return configs[:count]


def clean_config(config):
    return {key: round(float(value), 10) for key, value in config.items()}


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
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_value) if smoke_value is not None else None
    if smoke_epochs is not None:
        base_probe_count = 2
        top_count = 1
        half_lives = [7.0, 30.0]
        final_count = 1
        probe_epochs = max(1, min(1, smoke_epochs))
        final_epochs = max(1, min(args.epochs, smoke_epochs))
    else:
        base_probe_count = 100 if device.type == "cuda" else 40
        top_count = 12 if device.type == "cuda" else 6
        half_lives = [2.0, 3.0, 5.0, 7.0, 10.0, 14.0, 21.0, 30.0, 45.0, 60.0, 90.0]
        final_count = 5 if device.type == "cuda" else 3
        probe_epochs = 11 if device.type == "cuda" else 8
        final_epochs = args.epochs

    base_history = []
    base_results = []
    for index, config in enumerate(candidate_configs(args.seed, base_probe_count)):
        result = train_model(data, config, args.seed, probe_epochs, device, evaluate_fn, False)
        record = {
            "phase": "parent_config_probe",
            "probe": index + 1,
            "seed": args.seed,
            "epochs": probe_epochs,
            "config": clean_config(config),
            "gauc": round(result["metric"]["gauc"], 6),
            "ndcg5": round(result["metric"]["ndcg5"], 6),
            "primary": round(result["primary"], 6),
            "train_loss": round(result["train_loss"], 6),
            "ess_ratio": round(result["ess_ratio"], 6),
        }
        base_history.append(record)
        base_results.append((result["primary"], dict(config)))
        append_progress(progress_path, record)

    base_results.sort(key=lambda item: item[0], reverse=True)
    top_configs = [item[1] for item in base_results[:top_count]]
    valid_half_lives = []
    for half_life in half_lives:
        _, ess_ratio = recency_weights(data["train_dates"], half_life)
        if ess_ratio >= 0.15:
            valid_half_lives.append(half_life)
    if not valid_half_lives:
        valid_half_lives = [max(half_lives)]

    recency_history = []
    winning_config = None
    winning_probe_primary = -1.0
    recency_index = 0
    for rank, base_config in enumerate(top_configs):
        for half_life in valid_half_lives:
            recency_index += 1
            config = dict(base_config)
            config["half_life"] = float(half_life)
            result = train_model(data, config, args.seed, probe_epochs, device, evaluate_fn, False)
            record = {
                "phase": "recency_probe",
                "probe": recency_index,
                "parent_config_rank": rank + 1,
                "seed": args.seed,
                "epochs": probe_epochs,
                "config": clean_config(config),
                "gauc": round(result["metric"]["gauc"], 6),
                "ndcg5": round(result["metric"]["ndcg5"], 6),
                "primary": round(result["primary"], 6),
                "train_loss": round(result["train_loss"], 6),
                "ess_ratio": round(result["ess_ratio"], 6),
            }
            recency_history.append(record)
            append_progress(progress_path, record)
            if result["primary"] > winning_probe_primary:
                winning_probe_primary = result["primary"]
                winning_config = dict(config)

    final_history = []
    best_final = None
    for member in range(final_count):
        final_seed = args.seed + member * 1009
        result = train_model(
            data, winning_config, final_seed, final_epochs, device, evaluate_fn, True
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
            "ess_ratio": round(result["ess_ratio"], 6),
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
        "winning_config": winning_config,
        "winning_probe_primary": winning_probe_primary,
        "history": {
            "parent_config_probes": base_history,
            "recency_probes": recency_history,
            "final_runs": final_history,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_final["scores"]):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
