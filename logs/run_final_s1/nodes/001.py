"""Frequency-adaptive embedding regularization for the official five-field FM.

Sweeps the strength and frequency exponent using matched seeds, selects by mean
validation primary, then performs a full final training with the requested seed.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate
from harness.evaluate_provisional import evaluate as provisional_evaluate


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

    def adaptive_embedding_penalty(self, x, row_weights):
        e = self.emb(x)
        w = row_weights[x]
        return (w * e.square().sum(dim=2)).mean()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def parse_number(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            train_rows.append({
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__NO_AUTHOR__"),
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
                "long_view": float(r["long_view"]),
            })
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    field_names = ("user_id", "video_id", "author_id", "tab")
    maps = []
    for name in field_names:
        values = sorted({r[name] for r in train_rows})
        maps.append({v: i for i, v in enumerate(values)})
    field_dims = [len(m) + 1 for m in maps] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            for j, name in enumerate(field_names):
                local = maps[j].get(r[name], len(maps[j]))
                X[i, j] = local + offsets[j]
            bucket = int(np.searchsorted(quantiles, r["duration_ms"], side="right"))
            X[i, 4] = bucket + offsets[4]
        return X

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            val_rows.append({
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__NO_AUTHOR__"),
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
                "long_view": float(r["long_view"]),
            })
    return {
        "X_train": encode(train_rows),
        "y_train": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "X_val": encode(val_rows),
        "y_val": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "users_val": np.asarray([parse_number(r["user_id"]) for r in val_rows]),
        "videos_val": np.asarray([parse_number(r["video_id"]) for r in val_rows]),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "evaluator": provisional_evaluate,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            X_train = tr["X"].astype(np.int64)
            X_val = va["X"].astype(np.int64)
            video_offset = int(field_dims[0])
            videos = X_val[:, 1].astype(np.int64) - video_offset
            return {
                "X_train": X_train,
                "y_train": tr["y"].astype(np.float32),
                "X_val": X_val,
                "y_val": va["y"].astype(np.float32),
                "users_val": va["user"].copy(),
                "videos_val": videos,
                "field_dims": field_dims,
                "evaluator": official_evaluate,
            }
    return load_csv_data(data_dir)


def make_frequency_weights(X, field_dims, alpha):
    total_dim = int(field_dims.sum())
    counts = np.bincount(X.reshape(-1), minlength=total_dim).astype(np.float64)
    weights = np.ones(total_dim, dtype=np.float32)
    start = 0
    for dim in field_dims:
        end = start + int(dim)
        c = counts[start:end]
        observed = c > 0
        if np.any(observed):
            raw = np.ones_like(c)
            raw[observed] = np.power(c[observed], -alpha)
            occurrence_mean = float(np.sum(raw[observed] * c[observed]) / np.sum(c[observed]))
            raw[observed] /= max(occurrence_mean, 1e-12)
            raw[~observed] = 1.0
            weights[start:end] = np.clip(raw, 0.1, 20.0).astype(np.float32)
        start = end
    return weights


def train_once(data, device, seed, epochs, alpha, reg_lambda):
    seed_everything(seed)
    Xt = torch.from_numpy(data["X_train"])
    yt = torch.from_numpy(data["y_train"])
    Xv = torch.from_numpy(data["X_val"])
    row_weights = torch.from_numpy(
        make_frequency_weights(data["X_train"], data["field_dims"], alpha)
    ).to(device)
    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 100003)
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        batches = 0
        for begin in range(0, n, batch_size):
            idx = perm[begin:begin + batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            data_loss = bce(logits, yb)
            reg_loss = model.adaptive_embedding_penalty(xb, row_weights)
            loss = data_loss + reg_lambda * reg_loss
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        model.eval()
        score_parts = []
        with torch.no_grad():
            for begin in range(0, len(Xv), 65536):
                xb = Xv[begin:begin + 65536].to(device, non_blocking=True)
                score_parts.append(model(xb).detach().cpu().numpy())
        scores = np.concatenate(score_parts).astype(np.float64, copy=False)
        metrics = metric_values(data["evaluator"](data["users_val"], data["y_val"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(batches, 1), 6),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    final_metrics = metric_values(
        data["evaluator"](data["users_val"], data["y_val"].astype(int), best_scores)
    )
    del model, opt, row_weights
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, final_metrics, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    data = load_data(args.data_dir)

    alphas = [0.25, 0.5, 0.75, 1.0]
    lambdas = [0.001, 0.003, 0.01, 0.03]
    seed_count = 6 if device.type == "cuda" else 3
    if smoke is not None:
        alphas = [0.5]
        lambdas = [0.003]
        seed_count = 1

    history = []
    summaries = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for alpha in alphas:
            for reg_lambda in lambdas:
                probe_scores = []
                for seed_index in range(seed_count):
                    probe_seed = args.seed + seed_index * 1009
                    _, metrics, curve = train_once(
                        data, device, probe_seed, epochs, alpha, reg_lambda
                    )
                    record = {
                        "type": "probe",
                        "alpha": alpha,
                        "reg_lambda": reg_lambda,
                        "seed": probe_seed,
                        "gauc": metrics["gauc"],
                        "ndcg5": metrics["ndcg5"],
                        "primary": metrics["primary"],
                        "epochs_run": len(curve),
                    }
                    history.append(record)
                    probe_scores.append(metrics["primary"])
                    progress.write(json.dumps(record, sort_keys=True) + "\n")
                    progress.flush()
                summary = {
                    "alpha": alpha,
                    "reg_lambda": reg_lambda,
                    "mean_primary": float(np.mean(probe_scores)),
                    "std_primary": float(np.std(probe_scores)),
                    "seeds": seed_count,
                }
                summaries.append(summary)
        winner = max(summaries, key=lambda x: (x["mean_primary"], -x["std_primary"]))
        final_scores, final_metrics, final_curve = train_once(
            data,
            device,
            args.seed,
            epochs,
            winner["alpha"],
            winner["reg_lambda"],
        )
        final_record = {
            "type": "final",
            "alpha": winner["alpha"],
            "reg_lambda": winner["reg_lambda"],
            "seed": args.seed,
            "gauc": final_metrics["gauc"],
            "ndcg5": final_metrics["ndcg5"],
            "primary": final_metrics["primary"],
            "epochs_run": len(final_curve),
        }
        progress.write(json.dumps(final_record, sort_keys=True) + "\n")
        progress.flush()

    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected": {
            "alpha": winner["alpha"],
            "reg_lambda": winner["reg_lambda"],
            "probe_mean_primary": winner["mean_primary"],
            "probe_std_primary": winner["std_primary"],
        },
        "history": history,
        "config_summaries": summaries,
        "final_curve": final_curve,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, data["users_val"][i], data["videos_val"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
