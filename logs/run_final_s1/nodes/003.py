import argparse
import csv
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
        s = e.sum(dim=1)
        pair = 0.5 * (s * s - (e * e).sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


def encode_column(train_values, val_values):
    mapping = {}
    encoded_train = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        encoded_train[i] = mapping[value]
    encoded_val = np.fromiter((mapping.get(v, 0) for v in val_values), dtype=np.int64,
                              count=len(val_values))
    return encoded_train, encoded_val, len(mapping) + 1


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_cols = {"user_id": [], "video_id": [], "tab": [], "duration_ms": [],
                  "long_view": []}
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            train_cols["user_id"].append(row["user_id"])
            train_cols["video_id"].append(row["video_id"])
            train_cols["tab"].append(row["tab"])
            train_cols["duration_ms"].append(float(row["duration_ms"] or 0.0))
            train_cols["long_view"].append(float(row["long_view"]))
    val_cols = {"user_id": [], "video_id": [], "tab": [], "duration_ms": [],
                "long_view": []}
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_cols["user_id"].append(row["user_id"])
            val_cols["video_id"].append(row["video_id"])
            val_cols["tab"].append(row["tab"])
            val_cols["duration_ms"].append(float(row["duration_ms"] or 0.0))
            val_cols["long_view"].append(float(row["long_view"]))

    tu, vu, du = encode_column(train_cols["user_id"], val_cols["user_id"])
    tv, vv, dv = encode_column(train_cols["video_id"], val_cols["video_id"])
    tt, vt, dt = encode_column(train_cols["tab"], val_cols["tab"])
    train_duration = np.asarray(train_cols["duration_ms"], dtype=np.float64)
    val_duration = np.asarray(val_cols["duration_ms"], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    td = np.searchsorted(quantiles, train_duration, side="right").astype(np.int64)
    vd = np.searchsorted(quantiles, val_duration, side="right").astype(np.int64)
    dd = 10

    field_dims = np.asarray([du, dv, dv, dt, dd], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))
    Xt = np.column_stack([tu, tv, tv, tt, td]) + offsets
    Xv = np.column_stack([vu, vv, vv, vt, vd]) + offsets
    return {
        "Xt": Xt.astype(np.int64),
        "yt": np.asarray(train_cols["long_view"], dtype=np.float32),
        "train_user": np.asarray(train_cols["user_id"]),
        "Xv": Xv.astype(np.int64),
        "yv": np.asarray(val_cols["long_view"], dtype=np.float32),
        "val_user": np.asarray(val_cols["user_id"]),
        "val_video": np.asarray(val_cols["video_id"]),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            Xv = va["X"].astype(np.int64)
            video_offset = int(field_dims[0])
            val_video = Xv[:, 1] - video_offset
            return {
                "Xt": tr["X"].astype(np.int64),
                "yt": tr["y"].astype(np.float32),
                "train_user": np.asarray(tr["user"]),
                "Xv": Xv,
                "yv": va["y"].astype(np.float32),
                "val_user": np.asarray(va["user"]),
                "val_video": val_video,
                "field_dims": field_dims,
                "fast": True,
            }
    return load_csv_data(data_dir)


def predict(model, x, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def make_lambda_pairs(users, labels, scores, rng, samples_per_top=4):
    order = np.lexsort((-scores, users))
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    discounts = 1.0 / np.log2(np.arange(2, 7, dtype=np.float64))
    positive_indices = []
    negative_indices = []
    pair_weights = []

    for b in range(len(boundaries) - 1):
        group = order[boundaries[b]:boundaries[b + 1]]
        group_labels = labels[group] > 0.5
        positives = group[group_labels]
        negatives = group[~group_labels]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        idcg = float(discounts[:min(len(positives), 5)].sum())
        if idcg <= 0.0:
            continue
        ranks = {int(idx): rank for rank, idx in enumerate(group)}
        top = group[:min(5, len(group))]
        for idx in top:
            idx_int = int(idx)
            if labels[idx_int] > 0.5:
                pool = negatives
                pos_first = True
            else:
                pool = positives
                pos_first = False
            replace = len(pool) < samples_per_top
            chosen = rng.choice(pool, size=samples_per_top, replace=replace)
            rank_top = ranks[idx_int]
            discount_top = discounts[rank_top] if rank_top < 5 else 0.0
            for other in chosen:
                other_int = int(other)
                rank_other = ranks[other_int]
                discount_other = discounts[rank_other] if rank_other < 5 else 0.0
                delta = abs(discount_top - discount_other) / idcg
                if delta <= 0.0:
                    continue
                if pos_first:
                    positive_indices.append(idx_int)
                    negative_indices.append(other_int)
                else:
                    positive_indices.append(other_int)
                    negative_indices.append(idx_int)
                pair_weights.append(delta)

    if not pair_weights:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))
    weights = np.asarray(pair_weights, dtype=np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return (np.asarray(positive_indices, dtype=np.int64),
            np.asarray(negative_indices, dtype=np.int64), weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data_dir)
    Xt = data["Xt"]
    yt = data["yt"]
    Xv = data["Xv"]
    yv = data["yv"]
    train_users = data["train_user"]
    val_users = data["val_user"]
    total_dim = int(data["field_dims"].sum())

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    model = FM(total_dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    rng = np.random.default_rng(args.seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        last_bce = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(Xt[idx]).to(device)
            yb = torch.from_numpy(yt[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bce(model(xb), yb)
            loss.backward()
            optimizer.step()
            last_bce = float(loss.detach().cpu())

        train_scores = predict(model, Xt, device)
        pos_idx, neg_idx, lambda_weights = make_lambda_pairs(
            train_users, yt, train_scores, rng, samples_per_top=4)
        last_pair = 0.0
        if len(pos_idx) > 0:
            pair_order = rng.permutation(len(pos_idx))
            for start in range(0, len(pair_order), batch_size):
                take = pair_order[start:start + batch_size]
                xp = torch.from_numpy(Xt[pos_idx[take]]).to(device)
                xn = torch.from_numpy(Xt[neg_idx[take]]).to(device)
                wt = torch.from_numpy(lambda_weights[take]).to(device)
                optimizer.zero_grad(set_to_none=True)
                pair_loss = (torch.nn.functional.softplus(-(model(xp) - model(xn))) * wt).mean()
                loss = 0.5 * pair_loss
                loss.backward()
                optimizer.step()
                last_pair = float(pair_loss.detach().cpu())

        scores = predict(model, Xv, device)
        metrics = evaluate(val_users, yv.astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_bce": round(last_bce, 6),
            "train_lambda_loss": round(last_pair, 6),
            "lambda_pairs": int(len(pos_idx)),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_ndcg5": round(float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))), 6),
            "val_primary": round(primary, 6)
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        best_scores = predict(model, Xv, device)
    final_metrics = evaluate(val_users, yv.astype(int), best_scores)
    output = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "history": history,
        "config": {
            "method": "lambda_weighted_pairs",
            "pair_weight": 0.5,
            "samples_per_top5_item": 4,
            "embedding_dim": 16,
            "learning_rate": 0.001,
            "seed": args.seed
        }
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], data["val_video"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
