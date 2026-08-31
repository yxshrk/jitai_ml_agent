"""Matched-seed confirmation of user-centered (gauge-fixed) BCE for FM."""
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

    def relative_logit(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def forward(self, x):
        return self.relative_logit(x) + self.bias


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_rows = []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    val_rows = []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    duration_train = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(duration_train, np.arange(1, 10) / 10.0)

    field_names = ("user", "video", "author", "tab")
    mappings = {}
    dims = []
    for field in field_names:
        values = sorted({r[field] for r in train_rows})
        mappings[field] = {value: i + 1 for i, value in enumerate(values)}
        dims.append(len(values) + 1)
    dims.append(10)
    field_dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(field_names):
                x[i, j] = mappings[field].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right"))
            x[i, 4] = min(bucket, 9) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    return {
        "Xt": xt,
        "yt": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "train_user": xt[:, 0].copy(),
        "Xv": xv,
        "yv": np.asarray([r["y"] for r in val_rows], dtype=np.int64),
        "val_user": np.asarray([r["user"] for r in val_rows]),
        "val_video": np.asarray([r["video"] for r in val_rows]),
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
        xv = va["X"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_user": tr["X"][:, 0].astype(np.int64),
            "Xv": xv,
            "yv": va["y"].astype(np.int64),
            "val_user": np.asarray(va["user"]),
            "val_video": xv[:, 1] - video_offset,
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


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


def make_user_groups(user_ids):
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return order, starts, ends


def make_complete_user_batches(order, starts, ends, rng, batch_size):
    group_order = rng.permutation(len(starts))
    batches = []
    pieces = []
    count = 0
    for group_index in group_order:
        piece = order[starts[group_index]:ends[group_index]]
        size = len(piece)
        if pieces and count + size > batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
        pieces.append(piece)
        count += size
        if count >= batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
    if pieces:
        batches.append(np.concatenate(pieces))
    return batches


def centered_logits(model, x, users):
    relative = model.relative_logit(x)
    _, inverse, counts = torch.unique(users, sorted=False, return_inverse=True,
                                      return_counts=True)
    sums = torch.zeros(len(counts), dtype=relative.dtype, device=relative.device)
    sums.scatter_add_(0, inverse, relative)
    means = sums / counts.to(relative.dtype)
    return relative - means[inverse] + model.bias


def predict(model, xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_one(data, evaluate, device, seed, epochs, centered):
    seed_everything(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    train_user = torch.from_numpy(data["train_user"])
    xv = torch.from_numpy(data["Xv"])
    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    order, starts, ends = make_user_groups(data["train_user"])
    rng = np.random.RandomState(seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []

    for epoch in range(epochs):
        model.train()
        batches = make_complete_user_batches(order, starts, ends, rng, 8192)
        loss_value = 0.0
        for idx_np in batches:
            idx = torch.from_numpy(idx_np)
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            ub = train_user[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if centered:
                logits = centered_logits(model, xb, ub)
            else:
                logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        scores = predict(model, xv, device)
        metrics = evaluate(data["val_user"], data["yv"], scores)
        gauc, ndcg5, primary = metric_values(metrics)
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_value, 6),
            "val_gauc": round(gauc, 8),
            "val_ndcg5": round(ndcg5, 8),
            "val_primary": round(primary, 8),
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(data["val_user"], data["yv"], best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    return {
        "seed": int(seed),
        "objective": "user_centered_bce" if centered else "ordinary_bce_control",
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_epoch": int(max(epoch_history, key=lambda x: x["val_primary"])["epoch"]),
        "epochs": epoch_history,
    }, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast"])

    if smoke is not None:
        probe_count = 2
    elif device.type == "cuda":
        probe_count = 16
    else:
        probe_count = 10
    seeds = [args.seed + 1009 * i for i in range(probe_count)]

    history = []
    paired_deltas = []
    final_scores = None
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for i, seed in enumerate(seeds):
            control_result, _ = train_one(data, evaluate, device, seed, epochs, False)
            history.append(control_result)
            progress.write(json.dumps({
                "seed": seed,
                "objective": "ordinary_bce_control",
                "primary": control_result["primary"],
            }, sort_keys=True) + "\n")
            progress.flush()

            centered_result, centered_scores = train_one(
                data, evaluate, device, seed, epochs, True
            )
            history.append(centered_result)
            delta = centered_result["primary"] - control_result["primary"]
            paired_deltas.append(float(delta))
            progress.write(json.dumps({
                "seed": seed,
                "objective": "user_centered_bce",
                "primary": centered_result["primary"],
                "paired_delta": delta,
            }, sort_keys=True) + "\n")
            progress.flush()
            if i == 0:
                final_scores = centered_scores.copy()

    final_metrics = evaluate(data["val_user"], data["yv"], final_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    deltas = np.asarray(paired_deltas, dtype=np.float64)
    confirmation = {
        "probe_count": int(probe_count),
        "mean_paired_delta": float(deltas.mean()),
        "std_paired_delta": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        "positive_pairs": int((deltas > 0).sum()),
        "clears_0.002_pairs": int((deltas >= 0.002).sum()),
        "final_prediction_seed": int(args.seed),
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "confirmation": confirmation,
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
