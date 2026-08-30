"""FM baseline with normalized seven-day exponential training recency weights."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch


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


def date_ordinal(value):
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def ordinal_array(values):
    cache = {}
    result = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        key = str(value)
        if key not in cache:
            cache[key] = date_ordinal(value)
        result[i] = cache[key]
    return result


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"] or 0.0),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })

    def make_map(key):
        values = sorted({row[key] for row in train_rows})
        return {value: i + 1 for i, value in enumerate(values)}

    user_map = make_map("user_id")
    video_map = make_map("video_id")
    tab_map = make_map("tab")
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9)) if len(train_duration) else np.zeros(9)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
        x += offsets[None, :]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "train_date": np.asarray([r["date"] for r in train_rows]),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "val_user": np.asarray([r["user_id"] for r in val_rows]),
        "val_video": np.asarray([r["video_id"] for r in val_rows]),
        "val_date": np.asarray([r["date"] for r in val_rows]),
        "field_dims": field_dims,
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
            "train_date": tr["date"],
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_user": va["user"],
            "val_video": va["X"][:, 1].astype(np.int64) - video_offset,
            "val_date": va["date"],
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    data = load_data(args.data_dir)
    if data["fast"]:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    train_ord = ordinal_array(data["train_date"])
    val_ord = ordinal_array(data["val_date"])
    valid_boundary = float(np.min(val_ord)) if len(val_ord) else float(np.max(train_ord) + 1.0)
    age_days = np.maximum(0.0, valid_boundary - train_ord)
    weights_np = np.exp(-np.log(2.0) * age_days / 7.0).astype(np.float32)
    weights_np /= max(float(weights_np.mean()), 1e-12)
    effective_n = float(weights_np.sum() ** 2 / np.maximum(np.square(weights_np).sum(), 1e-12))

    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    wt = torch.from_numpy(weights_np)
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        last_loss = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad()
            row_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(Xt[idx]), yt[idx], reduction="none"
            )
            loss = (row_loss * wt[idx]).mean()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[start:start + 65536]).cpu().numpy()
                for start in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(data["val_user"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
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

    final_metrics = evaluate(data["val_user"], data["yv"].astype(int), best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "history": history,
            "config": {
                "recency_half_life_days": 7.0,
                "weight_normalization": "mean_one",
                "effective_sample_size": effective_n,
                "validation_boundary_ordinal": valid_boundary,
            },
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_user"][i], data["val_video"][i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
