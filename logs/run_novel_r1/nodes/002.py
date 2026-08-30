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


class ShiftClassifier(torch.nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        return self.bias + self.lin(x).sum((1, 2))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "__none__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
                row["date"],
            ))

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append((
                row["user_id"],
                row["video_id"],
                row.get("author_id", "__none__"),
                row["tab"],
                float(row["duration_ms"]),
                float(row["long_view"]),
                row["date"],
            ))

    durations = np.asarray([r[4] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    quantiles = np.asarray(quantiles, dtype=np.float64)

    raw_train_fields = [
        [r[0] for r in train_rows],
        [r[1] for r in train_rows],
        [r[2] for r in train_rows],
        [r[3] for r in train_rows],
        [str(int(np.searchsorted(quantiles, r[4], side="right"))) for r in train_rows],
    ]
    raw_val_fields = [
        [r[0] for r in val_rows],
        [r[1] for r in val_rows],
        [r[2] for r in val_rows],
        [r[3] for r in val_rows],
        [str(int(np.searchsorted(quantiles, r[4], side="right"))) for r in val_rows],
    ]

    field_dims = []
    mappings = []
    for values in raw_train_fields:
        unique = sorted(set(values))
        mapping = {value: i + 1 for i, value in enumerate(unique)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)

    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)
    Xt = np.empty((len(train_rows), 5), dtype=np.int64)
    Xv = np.empty((len(val_rows), 5), dtype=np.int64)
    for field in range(5):
        mapping = mappings[field]
        Xt[:, field] = np.asarray(
            [mapping.get(v, 0) for v in raw_train_fields[field]], dtype=np.int64
        ) + offsets[field]
        Xv[:, field] = np.asarray(
            [mapping.get(v, 0) for v in raw_val_fields[field]], dtype=np.int64
        ) + offsets[field]

    return {
        "Xt": Xt,
        "yt": np.asarray([r[5] for r in train_rows], dtype=np.float32),
        "dates": np.asarray([r[6] for r in train_rows]),
        "Xv": Xv,
        "yv": np.asarray([r[5] for r in val_rows], dtype=np.float32),
        "val_user": np.asarray([r[0] for r in val_rows]),
        "val_video": np.asarray([r[1] for r in val_rows]),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "fast_path": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=False) as tr:
            Xt = tr["X"].astype(np.int64)
            yt = tr["y"].astype(np.float32)
            dates = tr["date"].copy()
            field_dims = tr["field_dims"].astype(np.int64)
        with np.load(val_npz, allow_pickle=False) as va:
            Xv = va["X"].astype(np.int64)
            yv = va["y"].astype(np.float32)
            val_user = va["user"].copy()
        video_offset = int(field_dims[0])
        val_video = Xv[:, 1] - video_offset
        return {
            "Xt": Xt,
            "yt": yt,
            "dates": dates,
            "Xv": Xv,
            "yv": yv,
            "val_user": val_user,
            "val_video": val_video,
            "field_dims": field_dims,
            "fast_path": True,
        }
    return load_csv_data(data_dir)


def get_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
        return evaluate
    from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def temporal_labels(dates):
    normalized = np.asarray([str(x) for x in dates])
    unique_days = np.unique(normalized)
    if len(unique_days) < 2:
        return np.zeros(len(normalized), dtype=np.float32)
    cutoff = unique_days[len(unique_days) // 2]
    return (normalized >= cutoff).astype(np.float32)


def train_shift_classifier(Xt, shift_y, total_dim, device, seed, epochs):
    seed_all(seed)
    model = ShiftClassifier(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
    X_cpu = torch.from_numpy(Xt)
    y_cpu = torch.from_numpy(shift_y)
    n = len(X_cpu)
    bs = 16384
    history = []

    if shift_y.min() == shift_y.max():
        return np.ones(n, dtype=np.float32), history

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + 7919 * (epoch + 1))
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = X_cpu[idx].to(device, non_blocking=True)
            yb = y_cpu[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
            count = len(idx)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count
        history.append({"epoch": epoch + 1, "loss": round(loss_sum / max(seen, 1), 6)})

    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, n, 65536):
            xb = X_cpu[start:start + 65536].to(device, non_blocking=True)
            pieces.append(torch.sigmoid(model(xb)).cpu().numpy())
    probabilities = np.concatenate(pieces).astype(np.float32)
    return probabilities, history


def make_weights(probabilities, prior_late, config):
    if config["alpha"] == 0.0:
        return np.ones(len(probabilities), dtype=np.float32)
    ratio = probabilities.astype(np.float64) / max(float(prior_late), 1e-6)
    ratio = np.maximum(ratio, 1e-6) ** float(config["alpha"])
    ratio = np.clip(ratio, float(config["floor"]), float(config["cap"]))
    ratio /= max(float(ratio.mean()), 1e-12)
    return ratio.astype(np.float32)


def predict(model, X_cpu, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X_cpu), 65536):
            xb = X_cpu[start:start + 65536].to(device, non_blocking=True)
            output.append(model(xb).detach().cpu().numpy())
    return np.concatenate(output)


def train_fm(Xt, yt, Xv, weights, total_dim, evaluator, val_user, val_y,
             device, seed, epochs):
    seed_all(seed)
    model = FM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_train = torch.from_numpy(Xt)
    y_train = torch.from_numpy(yt)
    w_train = torch.from_numpy(weights)
    X_val = torch.from_numpy(Xv)
    n = len(y_train)
    bs = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        generator = torch.Generator()
        generator.manual_seed(seed + 104729 * (epoch + 1))
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            xb = X_train[idx].to(device, non_blocking=True)
            yb = y_train[idx].to(device, non_blocking=True)
            wb = w_train[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xb), yb, reduction="none"
            )
            loss = (raw * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
            loss_sum += float((raw.detach() * wb).sum().cpu())
            weight_sum += float(wb.sum().detach().cpu())

        scores = predict(model, X_val, device)
        metrics = evaluator(val_user, val_y.astype(int), scores)
        values = metric_values(metrics)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(weight_sum, 1e-12), 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    return best_scores, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluator = get_evaluator(data["fast_path"])
    total_dim = int(data["field_dims"].sum())

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_value) if smoke_value is not None else None
    fm_epochs = max(1, args.epochs)
    shift_epochs = 5
    if smoke_epochs is not None:
        fm_epochs = min(fm_epochs, max(1, smoke_epochs))
        shift_epochs = min(shift_epochs, max(1, smoke_epochs))

    shift_y = temporal_labels(data["dates"])
    prior_late = float(shift_y.mean())
    probabilities, shift_history = train_shift_classifier(
        data["Xt"], shift_y, total_dim, device, args.seed + 17, shift_epochs
    )

    if smoke_epochs is not None:
        configs = [{"alpha": 1.0, "floor": 0.25, "cap": 2.0}]
    else:
        configs = [{"alpha": 0.0, "floor": 1.0, "cap": 1.0}]
        alphas = np.linspace(0.1, 1.2, 12)
        for alpha in alphas:
            for floor in (0.10, 0.25):
                for cap in (1.5, 2.0, 3.0):
                    configs.append({
                        "alpha": round(float(alpha), 4),
                        "floor": float(floor),
                        "cap": float(cap),
                    })

    progress_path = os.path.join(args.out_dir, "progress.log")
    probe_history = []
    best_primary = -1.0
    best_config = None

    for probe_index, config in enumerate(configs):
        weights = make_weights(probabilities, prior_late, config)
        scores, curve = train_fm(
            data["Xt"], data["yt"], data["Xv"], weights, total_dim,
            evaluator, data["val_user"], data["yv"], device, args.seed,
            fm_epochs
        )
        metrics = metric_values(evaluator(
            data["val_user"], data["yv"].astype(int), scores
        ))
        record = {
            "probe": probe_index + 1,
            "config": config,
            "weight_min": round(float(weights.min()), 6),
            "weight_max": round(float(weights.max()), 6),
            "weight_std": round(float(weights.std()), 6),
            "gauc": metrics["gauc"],
            "ndcg5": metrics["ndcg5"],
            "primary": metrics["primary"],
            "curve": curve,
        }
        probe_history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({
                "probe": probe_index + 1,
                "config": config,
                "primary": metrics["primary"],
            }, sort_keys=True) + "\n")
        if metrics["primary"] > best_primary + 1e-12:
            best_primary = metrics["primary"]
            best_config = dict(config)

    final_weights = make_weights(probabilities, prior_late, best_config)
    final_scores, final_curve = train_fm(
        data["Xt"], data["yt"], data["Xv"], final_weights, total_dim,
        evaluator, data["val_user"], data["yv"], device, args.seed,
        fm_epochs
    )
    final_metrics = metric_values(evaluator(
        data["val_user"], data["yv"].astype(int), final_scores
    ))

    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": {
            "shift_classifier": shift_history,
            "shift_prior_late": prior_late,
            "shift_probability_mean": float(probabilities.mean()),
            "shift_probability_std": float(probabilities.std()),
            "probes": probe_history,
            "selected_config": best_config,
            "final_curve": final_curve,
        },
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
