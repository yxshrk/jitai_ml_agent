"""FM baseline with validation-selected exponential training recency weighting."""
import argparse
import csv
import datetime
import json
import math
import os
import sys

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


def parse_day(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    s = str(value).strip()
    try:
        if "-" in s:
            return datetime.date.fromisoformat(s[:10]).toordinal()
        n = int(float(s))
        digits = str(abs(n))
        if len(digits) >= 8:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        return n
    except (ValueError, TypeError, OverflowError):
        return 0


def day_array(values):
    values = np.asarray(values)
    unique, inv = np.unique(values, return_inverse=True)
    parsed = np.asarray([parse_day(x) for x in unique], dtype=np.int64)
    return parsed[inv]


def read_csv_rows(path, include_label):
    users, videos, tabs, durations, dates, labels = [], [], [], [], [], []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            users.append(row["user_id"])
            videos.append(row["video_id"])
            tabs.append(row["tab"])
            durations.append(float(row["duration_ms"]))
            dates.append(row["date"])
            if include_label:
                labels.append(float(row["long_view"]))
    result = {
        "user": np.asarray(users),
        "video": np.asarray(videos),
        "tab": np.asarray(tabs),
        "duration": np.asarray(durations, dtype=np.float64),
        "date": np.asarray(dates),
    }
    if include_label:
        result["y"] = np.asarray(labels, dtype=np.float32)
    return result


def make_mapping(values):
    unique = sorted(set(values.tolist()))
    return {v: i + 1 for i, v in enumerate(unique)}


def apply_mapping(values, mapping):
    return np.fromiter((mapping.get(v, 0) for v in values), dtype=np.int64, count=len(values))


def load_csv_data(data_dir):
    tr = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    va = read_csv_rows(os.path.join(data_dir, "val.csv"), True)
    user_map = make_mapping(tr["user"])
    video_map = make_mapping(tr["video"])
    tab_map = make_mapping(tr["tab"])

    positive_duration = tr["duration"][tr["duration"] > 0]
    if len(positive_duration):
        cuts = np.unique(np.quantile(positive_duration, np.linspace(0.0, 1.0, 17)[1:-1]))
    else:
        cuts = np.asarray([], dtype=np.float64)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(cuts) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(split):
        local = np.column_stack([
            apply_mapping(split["user"], user_map),
            apply_mapping(split["video"], video_map),
            np.zeros(len(split["user"]), dtype=np.int64),
            apply_mapping(split["tab"], tab_map),
            np.searchsorted(cuts, split["duration"], side="right") + 1,
        ])
        return local + offsets[None, :]

    return {
        "Xt": encode(tr),
        "yt": tr["y"],
        "Xv": encode(va),
        "yv": va["y"],
        "train_date": tr["date"],
        "val_date": va["date"],
        "val_user": va["user"],
        "val_video": va["video"],
        "field_dims": field_dims,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz, allow_pickle=False)
        va = np.load(val_npz, allow_pickle=False)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_video = va["X"][:, 1].astype(np.int64) - video_offset
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "train_date": tr["date"],
            "val_date": va["date"],
            "val_user": va["user"],
            "val_video": val_video,
            "field_dims": field_dims,
            "fast": True,
        }
    data = load_csv_data(data_dir)
    data["fast"] = False
    return data


def recency_weights(train_dates, val_dates, half_life):
    train_days = day_array(train_dates)
    val_days = day_array(val_dates)
    valid_val = val_days[val_days != 0]
    if len(valid_val):
        boundary = int(valid_val.min())
    else:
        boundary = int(train_days.max()) + 1
    ages = np.maximum(0, boundary - train_days).astype(np.float64)
    weights = np.exp(-math.log(2.0) * ages / float(half_life))
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        weights = np.ones_like(weights)
    else:
        weights /= mean_weight
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    return weights.astype(np.float32), ess / max(1, len(weights)), boundary


def metric_values(evaluator, users, labels, scores):
    raw = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(raw["GAUC"] if "GAUC" in raw else raw["gauc"]),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw["primary"]),
    }


def train_one(total_dim, Xt, yt, Xv, yv, users, weights, seed, epochs, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = FM(total_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n = len(yt)
    bs = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_epoch = 0
    patience = 0
    curve = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        last_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            losses = bce(model(Xt[idx]), yt[idx])
            loss = (losses * weights[idx]).mean()
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu().item())

        model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(Xv), 65536):
                chunks.append(model(Xv[i:i + 65536]).detach().cpu().numpy())
        scores = np.concatenate(chunks)
        metrics = metric_values(evaluator, users, yv, scores)
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_metrics = metrics
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, best_metrics, best_epoch, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    data = load_data(a.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = a.epochs if smoke is None else min(a.epochs, int(smoke))
    if epochs < 1:
        epochs = 1

    Xt = torch.from_numpy(data["Xt"]).to(device)
    yt = torch.from_numpy(data["yt"]).to(device)
    Xv = torch.from_numpy(data["Xv"]).to(device)
    total_dim = int(data["field_dims"].sum())

    if smoke is not None:
        half_lives = np.asarray([2.0, 8.0, 32.0, 128.0])
        probe_seeds = [a.seed]
    else:
        half_lives = np.geomspace(1.0, 160.0, 48)
        seed_count = 8 if device.type == "cuda" else 3
        probe_seeds = [a.seed + 1009 * i for i in range(seed_count)]

    weight_cache = {}
    eligible = []
    for half_life in half_lives:
        weights_np, ess_ratio, boundary = recency_weights(
            data["train_date"], data["val_date"], float(half_life)
        )
        if ess_ratio >= 0.20:
            key = float(half_life)
            weight_cache[key] = (weights_np, ess_ratio, boundary)
            eligible.append(key)
    if not eligible:
        key = float(half_lives[-1])
        weight_cache[key] = recency_weights(data["train_date"], data["val_date"], key)
        eligible = [key]

    progress_path = os.path.join(a.out_dir, "progress.log")
    probe_history = []
    grouped = {key: [] for key in eligible}
    with open(progress_path, "a") as progress:
        for half_life in eligible:
            weights_np, ess_ratio, boundary = weight_cache[half_life]
            weights = torch.from_numpy(weights_np).to(device)
            for probe_seed in probe_seeds:
                _, metrics, best_epoch, _ = train_one(
                    total_dim, Xt, yt, Xv, data["yv"], data["val_user"],
                    weights, probe_seed, epochs, device, evaluator
                )
                record = {
                    "half_life_days": round(half_life, 8),
                    "seed": int(probe_seed),
                    "ess_ratio": round(float(ess_ratio), 6),
                    "validation_boundary": int(boundary),
                    "best_epoch": int(best_epoch),
                    "gauc": round(metrics["gauc"], 6),
                    "ndcg5": round(metrics["ndcg5"], 6),
                    "primary": round(metrics["primary"], 6),
                }
                probe_history.append(record)
                grouped[half_life].append(metrics["primary"])
                progress.write(json.dumps(record, sort_keys=True) + "\n")
                progress.flush()
            del weights

    selected_half_life = max(
        eligible,
        key=lambda h: (float(np.mean(grouped[h])), -h)
    )
    selected_weights_np, selected_ess, boundary = weight_cache[selected_half_life]
    selected_weights = torch.from_numpy(selected_weights_np).to(device)
    best_scores, final_metrics, best_epoch, final_curve = train_one(
        total_dim, Xt, yt, Xv, data["yv"], data["val_user"], selected_weights,
        a.seed, epochs, device, evaluator
    )

    metrics_out = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_half_life_days": float(selected_half_life),
        "selected_mean_probe_primary": float(np.mean(grouped[selected_half_life])),
        "selected_ess_ratio": float(selected_ess),
        "validation_boundary": int(boundary),
        "best_epoch": int(best_epoch),
        "history": probe_history,
        "final_learning_curve": final_curve,
    }
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.6g}\n")


if __name__ == "__main__":
    main()
