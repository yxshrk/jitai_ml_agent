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


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                row = {
                    "user": r["user_id"],
                    "video": r["video_id"],
                    "author": r.get("author_id", r["video_id"]),
                    "tab": r["tab"],
                    "duration": float(r["duration_ms"] or 0.0),
                    "y": float(r["long_view"]),
                }
                rows.append(row)
        return rows

    tr_rows = read_rows(train_path, True)
    va_rows = read_rows(val_path, False)
    durations = np.asarray([r["duration"] for r in tr_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    fields = ["user", "video", "author", "tab"]
    maps = []
    for field in fields:
        values = sorted({r[field] for r in tr_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})

    field_dims = np.asarray([len(m) + 1 for m in maps] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = maps[j].get(r[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, r["duration"], side="right"))
            x[i, 4] = bucket + offsets[4]
        y = np.asarray([r["y"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user"] for r in rows])
        videos = np.asarray([r["video"] for r in rows])
        return x, y, users, videos

    Xt, yt, ut, _ = encode(tr_rows)
    Xv, yv, uv, vv = encode(va_rows)
    return Xt, yt, ut, Xv, yv, uv, vv, field_dims, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        Xt = tr["X"].astype(np.int64)
        yt = tr["y"].astype(np.float32)
        ut = tr["user"]
        Xv = va["X"].astype(np.int64)
        yv = va["y"].astype(np.float32)
        uv = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        vv = Xv[:, 1] - video_offset
        return Xt, yt, ut, Xv, yv, uv, vv, field_dims, True
    return load_csv_data(data_dir)


def make_pair_sampler(users, labels):
    positive = np.flatnonzero(labels >= 0.5)
    negative = np.flatnonzero(labels < 0.5)
    if len(positive) == 0 or len(negative) == 0:
        return None

    neg_order = np.argsort(users[negative], kind="stable")
    neg_sorted = negative[neg_order]
    neg_users = users[neg_sorted]
    unique_users, starts, counts = np.unique(
        neg_users, return_index=True, return_counts=True
    )
    pos_users = users[positive]
    locations = np.searchsorted(unique_users, pos_users)
    valid = locations < len(unique_users)
    matched = np.zeros(len(positive), dtype=bool)
    matched[valid] = unique_users[locations[valid]] == pos_users[valid]
    positive = positive[matched]
    locations = locations[matched]
    if len(positive) == 0:
        return None

    return (
        torch.from_numpy(positive.astype(np.int64)),
        torch.from_numpy(neg_sorted.astype(np.int64)),
        torch.from_numpy(starts[locations].astype(np.int64)),
        torch.from_numpy(counts[locations].astype(np.int64)),
    )


def predict(model, x, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            parts.append(model(x[i:i + batch_size]).cpu().numpy())
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    Xt_np, yt_np, ut, Xv_np, yv, uv, vv, field_dims, fast_path = load_data(args.data_dir)
    Xt = torch.from_numpy(Xt_np)
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(Xv_np)

    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    model = FM(int(field_dims.sum()), k=16)
    averaged_model = FM(int(field_dims.sum()), k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    pair_sampler = make_pair_sampler(ut, yt_np)

    n = len(yt)
    batch_size = 8192
    best_raw = -1.0
    raw_patience = 0
    best_average = -1.0
    best_scores = None
    average_state = None
    average_count = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        last_loss = 0.0
        for begin in range(0, n, batch_size):
            idx = permutation[begin:begin + batch_size]
            optimizer.zero_grad()
            point_loss = bce(model(Xt[idx]), yt[idx])

            if pair_sampler is not None:
                pos_idx, neg_sorted, neg_starts, neg_counts = pair_sampler
                q = torch.randint(0, len(pos_idx), (len(idx),))
                selected_pos = pos_idx[q]
                offsets = torch.floor(torch.rand(len(idx)) * neg_counts[q].float()).long()
                selected_neg = neg_sorted[neg_starts[q] + offsets]
                difference = model(Xt[selected_pos]) - model(Xt[selected_neg])
                pair_loss = torch.nn.functional.softplus(-difference).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

        raw_scores = predict(model, Xv)
        raw_metrics = evaluate(uv, yv.astype(int), raw_scores)
        raw_primary = float(raw_metrics["primary"])
        if raw_primary > best_raw + 1e-6:
            best_raw = raw_primary
            raw_patience = 0
        else:
            raw_patience += 1

        averaged_primary = None
        if 4 <= epoch <= 7:
            state = model.state_dict()
            if average_state is None:
                average_state = {k: v.detach().clone() for k, v in state.items()}
                average_count = 1
            else:
                average_count += 1
                for key in average_state:
                    average_state[key].add_((state[key].detach() - average_state[key]) / average_count)

            if average_count >= 2:
                averaged_model.load_state_dict(average_state)
                averaged_scores = predict(averaged_model, Xv)
                averaged_metrics = evaluate(uv, yv.astype(int), averaged_scores)
                averaged_primary = float(averaged_metrics["primary"])
                if averaged_primary > best_average + 1e-6:
                    best_average = averaged_primary
                    best_scores = averaged_scores.copy()

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
            "val_gauc": round(float(raw_metrics.get("GAUC", raw_metrics.get("gauc", 0.0))), 6),
            "val_primary": round(raw_primary, 6),
            "swa_primary": None if averaged_primary is None else round(averaged_primary, 6),
        })

        if epoch >= 7 and raw_patience >= 2:
            break

    if best_scores is None:
        best_scores = raw_scores

    metrics = evaluate(uv, yv.astype(int), best_scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics.get("nDCG@5", metrics.get("ndcg5"))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(gauc),
            "ndcg5": float(ndcg5),
            "primary": float(metrics["primary"]),
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, uv[i], vv[i], format(float(score), ".7g")])


if __name__ == "__main__":
    main()
