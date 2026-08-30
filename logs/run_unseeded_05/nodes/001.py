"""FM baseline with tightly bounded late-checkpoint EMA averaging."""
import argparse
import csv
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


def read_csv(path):
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def make_mapping(rows, column):
    values = sorted({row[column] for row in rows})
    return {value: i + 1 for i, value in enumerate(values)}


def load_csv_data(data_dir):
    train_rows = read_csv(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv(os.path.join(data_dir, "val.csv"))

    user_map = make_mapping(train_rows, "user_id")
    video_map = make_mapping(train_rows, "video_id")
    tab_map = make_mapping(train_rows, "tab")

    train_duration = np.asarray(
        [float(row["duration_ms"]) for row in train_rows], dtype=np.float64
    )
    boundaries = np.unique(
        np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    )

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            duration = float(row["duration_ms"])
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = min(int(np.searchsorted(boundaries, duration, side="right")), 9)
        return x + offsets[None, :]

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32),
        "users": np.asarray([row["user_id"] for row in val_rows]),
        "videos": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
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
                "yv": va["y"].astype(np.float32),
                "users": va["user"].copy(),
                "videos": (va["X"][:, 1].astype(np.int64) - video_offset),
                "field_dims": field_dims,
                "fast": True,
            }
    return load_csv_data(data_dir)


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def predict(model, x, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            parts.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(parts)


def copy_state(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def update_ema(ema_state, model, decay):
    current = model.state_dict()
    with torch.no_grad():
        for name in ema_state:
            ema_state[name].mul_(decay).add_(current[name], alpha=1.0 - decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast"])

    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    yv_int = data["yv"].astype(int)

    model = FM(int(data["field_dims"].sum()), k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    batch_size = 8192
    raw_best = -1.0
    patience = 0
    fallback_scores = None
    fallback_primary = -1.0
    ema_state = None
    ema_best_scores = None
    ema_best_primary = -1.0

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()

        raw_scores = predict(model, xv)
        raw_metrics = evaluate(data["users"], yv_int, raw_scores)
        raw_primary = float(raw_metrics["primary"])

        if raw_primary > fallback_primary:
            fallback_primary = raw_primary
            fallback_scores = raw_scores.copy()

        if epoch_number == 4:
            ema_state = copy_state(model)
        elif 5 <= epoch_number <= 8:
            if ema_state is None:
                ema_state = copy_state(model)
            else:
                update_ema(ema_state, model, decay=0.75)

        if 5 <= epoch_number <= 8 and ema_state is not None:
            ema_model = FM(int(data["field_dims"].sum()), k=16)
            ema_model.load_state_dict(ema_state)
            ema_scores = predict(ema_model, xv)
            ema_metrics = evaluate(data["users"], yv_int, ema_scores)
            ema_primary = float(ema_metrics["primary"])
            if ema_primary > ema_best_primary:
                ema_best_primary = ema_primary
                ema_best_scores = ema_scores.copy()

        if raw_primary > raw_best + 1e-6:
            raw_best = raw_primary
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    best_scores = ema_best_scores if ema_best_scores is not None else fallback_scores
    metrics = evaluate(data["users"], yv_int, best_scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(
            {"gauc": float(gauc), "ndcg5": float(ndcg5),
             "primary": float(metrics["primary"])},
            fh,
        )

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
