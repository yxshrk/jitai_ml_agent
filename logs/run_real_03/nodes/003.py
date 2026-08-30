"""DCN-lite model augmented with hour-of-day and day-of-week context fields."""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch


class FMDCNLite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=7, k=16, hidden=128):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        width = num_fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.zeros_(self.mlp[2].weight)
        torch.nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        x0 = e.flatten(1)
        cross = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True)
        cross = cross + self.cross_b
        return fm + self.mlp(cross).squeeze(1)


def temporal_fields(hourmin, dates):
    hourmin = np.asarray(hourmin, dtype=np.int64)
    dates = np.asarray(dates, dtype=np.int64)
    hours = np.clip((hourmin // 100) % 24, 0, 23).astype(np.int64)
    weekdays = np.empty(len(dates), dtype=np.int64)
    cache = {}
    for value in np.unique(dates):
        ivalue = int(value)
        try:
            year = ivalue // 10000
            month = (ivalue // 100) % 100
            day = ivalue % 100
            weekday = datetime.date(year, month, day).weekday()
        except (ValueError, OverflowError):
            weekday = ivalue % 7
        cache[ivalue] = weekday
    for i, value in enumerate(dates):
        weekdays[i] = cache[int(value)]
    return hours, weekdays


def append_temporal(X, hourmin, dates, base_dim):
    hours, weekdays = temporal_fields(hourmin, dates)
    hour_field = base_dim + hours
    weekday_field = base_dim + 24 + weekdays
    return np.column_stack((X.astype(np.int64), hour_field, weekday_field))


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user_id": row["user_id"],
                    "video_id": row["video_id"],
                    "author_id": row.get("author_id", "0"),
                    "tab": row["tab"],
                    "duration_ms": float(row["duration_ms"]),
                    "hourmin": int(float(row["hourmin"])),
                    "date": int(float(row["date"])),
                    "long_view": float(row["long_view"]),
                }
                rows.append(item)
        return rows

    train_rows = read_rows(train_path, True)
    val_rows = read_rows(val_path, False)
    fields = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    dims = []
    for field in fields:
        values = sorted({row[field] for row in train_rows})
        mapping = {value: i + 1 for i, value in enumerate(values)}
        mappings.append(mapping)
        dims.append(len(values) + 1)

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    dims.append(10)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                X[i, j] = offsets[j] + mappings[j].get(row[field], 0)
            bucket = int(np.searchsorted(edges, row["duration_ms"], side="right"))
            X[i, 4] = offsets[4] + bucket
        return X

    Xt = encode(train_rows)
    Xv = encode(val_rows)
    return {
        "Xt": Xt,
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "Xv": Xv,
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "train_hourmin": np.asarray([r["hourmin"] for r in train_rows]),
        "val_hourmin": np.asarray([r["hourmin"] for r in val_rows]),
        "train_date": np.asarray([r["date"] for r in train_rows]),
        "val_date": np.asarray([r["date"] for r in val_rows]),
        "val_user": np.asarray([r["user_id"] for r in val_rows]),
        "val_video": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    train_npz = os.path.join(a.data_dir, "train.npz")
    val_npz = os.path.join(a.data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data.official.evaluate import evaluate
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        base_dim = int(field_dims.sum())
        Xt_np = append_temporal(tr["X"], tr["hourmin"], tr["date"], base_dim)
        Xv_np = append_temporal(va["X"], va["hourmin"], va["date"], base_dim)
        yt_np = tr["y"].astype(np.float32)
        yv = va["y"].astype(np.int64)
        val_users = va["user"]
        video_offset = int(field_dims[0])
        val_videos = va["X"][:, 1].astype(np.int64) - video_offset
    else:
        from harness.evaluate_provisional import evaluate
        data = read_csv_data(a.data_dir)
        field_dims = data["field_dims"]
        base_dim = int(field_dims.sum())
        Xt_np = append_temporal(data["Xt"], data["train_hourmin"], data["train_date"], base_dim)
        Xv_np = append_temporal(data["Xv"], data["val_hourmin"], data["val_date"], base_dim)
        yt_np = data["yt"]
        yv = data["yv"].astype(np.int64)
        val_users = data["val_user"]
        val_videos = data["val_video"]

    total_dim = base_dim + 31
    Xt = torch.from_numpy(Xt_np.astype(np.int64))
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(Xv_np.astype(np.int64))

    model = FMDCNLite(total_dim, num_fields=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(val_users, yv, scores)
        primary = metrics["primary"]
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(a.out_dir, exist_ok=True)
    metrics = evaluate(val_users, yv, best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"],
            "ndcg5": metrics.get("nDCG@5", metrics.get("ndcg5")),
            "primary": metrics["primary"],
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{val_users[i]},{val_videos[i]},{score:.6g}\n")


if __name__ == "__main__":
    main()
