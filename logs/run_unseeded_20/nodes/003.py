"""FM baseline with an aggressive compound anti-overfitting package."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RegularizedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.3):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    video_offset = int(field_dims[0])
    val_video = va["X"][:, 1].astype(np.int64) - video_offset
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "val_user": va["user"],
        "val_video": val_video,
        "field_dims": field_dims,
        "official": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", "__constant_author__"),
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "label": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {"__UNK__": 0}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = make_mapping(row["user"] for row in train_rows)
    video_map = make_mapping(row["video"] for row in train_rows)
    author_map = make_mapping(row["author"] for row in train_rows)
    tab_map = make_mapping(row["tab"] for row in train_rows)

    train_duration = np.asarray([row["duration"] for row in train_rows], dtype=np.float64)
    boundaries = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    boundaries = np.unique(boundaries)
    duration_dim = int(len(boundaries) + 1)
    field_dims = np.asarray(
        [len(user_map), len(video_map), len(author_map), len(tab_map), duration_dim],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video"], 0) + offsets[1]
            x[i, 2] = author_map.get(row["author"], 0) + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = np.searchsorted(boundaries, row["duration"], side="right") + offsets[4]
            y[i] = row["label"]
        return x, y

    Xt, yt = encode(train_rows)
    Xv, yv = encode(val_rows)
    return {
        "Xt": Xt,
        "yt": yt,
        "Xv": Xv,
        "yv": yv.astype(np.int64),
        "val_user": np.asarray([row["user"] for row in val_rows]),
        "val_video": np.asarray([row["video"] for row in val_rows]),
        "field_dims": field_dims,
        "official": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz"))
        and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    data = load_npz(args.data_dir) if fast_path else load_csv(args.data_dir)

    if data["official"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())

    model = RegularizedFM(total_dim, k=16, dropout=0.3)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.emb.parameters(), "weight_decay": 0.0},
            {"params": model.lin.parameters(), "weight_decay": 1e-3},
            {"params": [model.bias], "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    best_gauc = -1.0
    best_scores = None
    patience = 0

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            prediction_loss = bce(logits, yt[idx])
            accessed_embeddings = model.emb(xb)
            row_l2 = accessed_embeddings.square().sum(dim=(1, 2)).mean()
            loss = prediction_loss + 1e-4 * row_l2
            loss.backward()
            optimizer.step()

        scheduler.step()
        model.eval()
        with torch.no_grad():
            scores = np.concatenate(
                [
                    model(Xv[start:start + 65536]).cpu().numpy()
                    for start in range(0, len(Xv), 65536)
                ]
            )
        metrics = evaluate(data["val_user"], data["yv"], scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc + 1e-6:
            best_gauc = gauc
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    final_metrics = evaluate(data["val_user"], data["yv"], best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_user"][i], data["val_video"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
