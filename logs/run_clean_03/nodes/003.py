"""FM baseline with normalized exponential training recency weighting."""
import argparse
import csv
import datetime
import json
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


def date_ordinals(values):
    values = np.asarray(values).reshape(-1)
    out = np.empty(len(values), dtype=np.float64)
    valid = True
    for i, value in enumerate(values):
        try:
            v = int(value)
            year = v // 10000
            month = (v // 100) % 100
            day = v % 100
            out[i] = datetime.date(year, month, day).toordinal()
        except (TypeError, ValueError, OverflowError):
            valid = False
            break
    if valid:
        return out
    text = values.astype(str)
    unique = np.unique(text)
    lookup = {value: i for i, value in enumerate(unique)}
    return np.asarray([lookup[value] for value in text], dtype=np.float64)


def recency_weights(dates, half_life_days=7.0):
    ordinals = date_ordinals(dates)
    ages = np.maximum(0.0, np.max(ordinals) - ordinals)
    weights = np.exp(-np.log(2.0) * ages / half_life_days)
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        weights = np.ones_like(weights)
    else:
        weights /= mean_weight
    return weights.astype(np.float32)


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_with_mapping(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def duration_bucket(value):
    duration = float(value)
    return int(np.searchsorted(np.asarray([7000, 15000, 30000, 60000, 120000]), duration, side="right"))


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    train_users = [row["user_id"] for row in train_rows]
    train_videos = [row["video_id"] for row in train_rows]
    train_tabs = [row["tab"] for row in train_rows]
    user_map = make_mapping(train_users)
    video_map = make_mapping(train_videos)
    tab_map = make_mapping(train_tabs)

    train_fields = [
        encode_with_mapping(train_users, user_map),
        encode_with_mapping(train_videos, video_map),
        np.zeros(len(train_rows), dtype=np.int64),
        encode_with_mapping(train_tabs, tab_map),
        np.asarray([duration_bucket(row["duration_ms"]) for row in train_rows], dtype=np.int64),
    ]
    val_fields = [
        encode_with_mapping([row["user_id"] for row in val_rows], user_map),
        encode_with_mapping([row["video_id"] for row in val_rows], video_map),
        np.zeros(len(val_rows), dtype=np.int64),
        encode_with_mapping([row["tab"] for row in val_rows], tab_map),
        np.asarray([duration_bucket(row["duration_ms"]) for row in val_rows], dtype=np.int64),
    ]
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        6,
    ], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])
    xt = np.stack(train_fields, axis=1) + offsets
    xv = np.stack(val_fields, axis=1) + offsets
    return {
        "Xt": xt,
        "yt": np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32),
        "Xv": xv,
        "yv": np.asarray([int(float(row["long_view"])) for row in val_rows], dtype=np.int64),
        "train_date": np.asarray([row["date"] for row in train_rows]),
        "val_user": np.asarray([row["user_id"] for row in val_rows]),
        "val_video": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
        "official": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.int64),
            "train_date": tr["date"],
            "val_user": va["user"],
            "val_video": np.zeros(len(va["y"]), dtype=np.int64),
            "field_dims": tr["field_dims"],
            "official": True,
        }
    return load_csv_data(data_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))

    data = load_data(args.data_dir)
    if data["official"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    total_dim = int(np.asarray(data["field_dims"]).sum())
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    row_weights = torch.from_numpy(recency_weights(data["train_date"]))

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            losses = bce(model(xt[idx]), yt[idx])
            loss = (losses * row_weights[idx]).mean()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])
        metrics = evaluate(data["val_user"], data["yv"], scores)
        primary = metrics["primary"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(loss.item()), 5),
            "val_gauc": round(metrics.get("GAUC", metrics.get("gauc", 0.0)), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best + 1e-6:
            best = primary
            best_scores = scores
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        model.eval()
        with torch.no_grad():
            best_scores = np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(data["val_user"], data["yv"], best_scores)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"],
            "ndcg5": metrics.get("nDCG@5", metrics.get("ndcg5")),
            "primary": metrics["primary"],
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.6g}\n")


if __name__ == "__main__":
    main()
