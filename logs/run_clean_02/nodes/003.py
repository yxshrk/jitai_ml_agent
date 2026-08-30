"""FM baseline with normalized exponential training-recency weighting."""
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
    out = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        text = str(value.decode() if isinstance(value, bytes) else value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                out[i] = datetime.datetime.strptime(digits[:8], "%Y%m%d").date().toordinal()
            else:
                out[i] = int(float(text))
        except (ValueError, OverflowError):
            out[i] = i
    return out


def load_csv(data_dir):
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    def duration_bucket(row):
        duration = max(0, int(float(row["duration_ms"])))
        return str(min(duration // 10000, 60))

    field_getters = [
        lambda r: r["user_id"],
        lambda r: r["video_id"],
        lambda r: "0",
        lambda r: r["tab"],
        duration_bucket,
    ]
    mappings = []
    field_dims = []
    for getter in field_getters:
        values = sorted({getter(row) for row in train_rows})
        mapping = {value: i for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, (getter, mapping) in enumerate(zip(field_getters, mappings)):
            unknown = len(mapping)
            x[:, j] = [mapping.get(getter(row), unknown) + offsets[j] for row in rows]
        return x

    train = {
        "X": encode(train_rows),
        "y": np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32),
        "user": np.asarray([row["user_id"] for row in train_rows]),
        "date": np.asarray([row["date"] for row in train_rows]),
        "field_dims": field_dims,
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32),
        "user": np.asarray([row["user_id"] for row in val_rows]),
        "video": np.asarray([row["video_id"] for row in val_rows]),
        "date": np.asarray([row["date"] for row in val_rows]),
    }
    return train, val


def metric_value(metrics, upper, lower):
    return metrics[upper] if upper in metrics else metrics[lower]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    using_npz = os.path.exists(train_npz) and os.path.exists(val_npz)
    if using_npz:
        from data.official.evaluate import evaluate
        train_file = np.load(train_npz)
        val_file = np.load(val_npz)
        tr = {key: train_file[key] for key in train_file.files}
        va = {key: val_file[key] for key in val_file.files}
        video_offset = int(tr["field_dims"][0])
        val_videos = va["X"][:, 1].astype(np.int64) - video_offset
    else:
        from harness.evaluate_provisional import evaluate
        tr, va = load_csv(args.data_dir)
        val_videos = va["video"]

    total_dim = int(np.asarray(tr["field_dims"]).sum())
    xt = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    yt = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))

    train_days = date_ordinals(np.asarray(tr["date"]))
    val_days = date_ordinals(np.asarray(va["date"]))
    boundary = int(val_days.min())
    ages = np.maximum(0, boundary - train_days).astype(np.float64)
    recency = np.exp(-np.log(2.0) * ages / 30.0)
    recency /= recency.mean()
    weights = torch.from_numpy(recency.astype(np.float32))

    model = FM(total_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    n = len(yt)
    batch_size = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            row_losses = bce(model(xt[idx]), yt[idx])
            loss = (row_losses * weights[idx]).sum() / weights[idx].sum()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[start:start + 65536]).numpy()
                for start in range(0, len(xv), 65536)
            ])
        metrics = evaluate(np.asarray(va["user"]), np.asarray(va["y"]).astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(
        np.asarray(va["user"]), np.asarray(va["y"]).astype(int), best_scores
    )
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(metric_value(final_metrics, "GAUC", "gauc")),
            "ndcg5": float(metric_value(final_metrics, "nDCG@5", "ndcg5")),
            "primary": float(final_metrics["primary"]),
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, va["user"][i], val_videos[i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
