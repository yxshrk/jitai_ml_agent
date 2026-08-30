import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2):
        super().__init__()
        self.fields = fields
        self.k = k
        dim = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        self.input_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim)) for _ in range(2)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(dim)) for _ in range(2)
        ])
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.out = torch.nn.Linear(dim + hidden // 2, 1)

    def forward(self, x):
        x0 = self.input_dropout(self.emb(x).reshape(x.shape[0], -1))
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(dim=1, keepdim=True) + b + cross
        deep = self.mlp(x0)
        return self.out(torch.cat((cross, deep), dim=1)).squeeze(1)


def date_ordinals(values):
    values = np.asarray(values)
    out = np.empty(len(values), dtype=np.int64)
    cache = {}
    for i, raw in enumerate(values):
        if isinstance(raw, bytes):
            text = raw.decode("utf-8")
        else:
            text = str(raw)
        text = text.split(".")[0].replace("-", "")
        if text not in cache:
            try:
                d = datetime.datetime.strptime(text[:8], "%Y%m%d").date()
                cache[text] = d.toordinal()
            except ValueError:
                try:
                    cache[text] = int(float(text))
                except ValueError:
                    cache[text] = 0
        out[i] = cache[text]
    return out


def make_recency_weights(dates):
    ords = date_ordinals(dates)
    reference = datetime.date(2022, 4, 21).toordinal()
    if len(ords) and (ords.max() < reference - 3650 or ords.max() > reference + 3650):
        reference = int(ords.max())
    age = np.maximum(0, reference - ords).astype(np.float32)
    weights = np.exp(-np.log(2.0) * age / 7.0).astype(np.float32)
    mean = float(weights.mean()) if len(weights) else 1.0
    return weights / max(mean, 1e-8)


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def score_model(model, X, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            pieces.append(model(X[start:start + batch_size]).cpu().numpy())
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)


def train_segment(model, optimizer, X, y, weights, pair_pos, pair_neg, permutation,
                  pair_permutation, start, end, batch_size):
    model.train()
    pair_count = len(pair_pos)
    pair_cursor = 0
    for offset in range(start, end, batch_size):
        idx = permutation[offset:min(offset + batch_size, end)]
        optimizer.zero_grad()
        if pair_count:
            needed = len(idx)
            if pair_cursor + needed <= pair_count:
                psel = pair_permutation[pair_cursor:pair_cursor + needed]
            else:
                first = pair_permutation[pair_cursor:]
                remain = needed - len(first)
                psel = torch.cat((first, pair_permutation[:remain]))
            pair_cursor = (pair_cursor + needed) % pair_count
            pidx = pair_pos[psel]
            nidx = pair_neg[psel]
            joined = torch.cat((X[idx], X[pidx], X[nidx]), dim=0)
            logits = model(joined)
            b = len(idx)
            p = len(pidx)
            main_logits = logits[:b]
            pos_logits = logits[b:b + p]
            neg_logits = logits[b + p:]
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                main_logits, y[idx], reduction="none"
            )
            bce_loss = (raw_bce * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-8)
            raw_bpr = torch.nn.functional.softplus(-(pos_logits - neg_logits))
            pair_weights = weights[pidx]
            bpr_loss = (raw_bpr * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
            loss = 0.5 * bce_loss + 0.5 * bpr_loss
        else:
            logits = model(X[idx])
            raw_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y[idx], reduction="none"
            )
            loss = (raw_bce * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-8)
        loss.backward()
        optimizer.step()


def run_probe(config, arrays, evaluator, epochs, seed):
    X, y, users, weights, Xv, val_users, val_y, total_dim = arrays
    torch.manual_seed(seed)
    model = DCNLite(total_dim, dropout=config[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config[1])
    pair_pos_np, pair_neg_np = build_pairs(users, y.numpy(), seed + 31)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)
    generator = torch.Generator().manual_seed(seed + 71)
    best = -1.0
    for _ in range(epochs):
        permutation = torch.randperm(len(y), generator=generator)
        pair_permutation = torch.randperm(len(pair_pos), generator=generator) if len(pair_pos) else torch.empty(0, dtype=torch.int64)
        train_segment(model, optimizer, X, y, weights, pair_pos, pair_neg,
                      permutation, pair_permutation, 0, len(y), 8192)
        scores = score_model(model, Xv)
        metric = evaluator(val_users, val_y, scores)
        best = max(best, float(metric["primary"]))
        if config[2]:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    return best


def run_final(config, arrays, evaluator, epochs, seed):
    X, y, users, weights, Xv, val_users, val_y, total_dim = arrays
    torch.manual_seed(seed)
    model = DCNLite(total_dim, dropout=config[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config[1])
    pair_pos_np, pair_neg_np = build_pairs(users, y.numpy(), seed + 131)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)
    generator = torch.Generator().manual_seed(seed + 171)
    best_primary = -1.0
    best_scores = None
    n = len(y)
    for _ in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        pair_permutation = torch.randperm(len(pair_pos), generator=generator) if len(pair_pos) else torch.empty(0, dtype=torch.int64)
        midpoint = (n + 1) // 2
        for left, right in ((0, midpoint), (midpoint, n)):
            if left >= right:
                continue
            train_segment(model, optimizer, X, y, weights, pair_pos, pair_neg,
                          permutation, pair_permutation, left, right, 8192)
            scores = score_model(model, Xv)
            metric = evaluator(val_users, val_y, scores)
            primary = float(metric["primary"])
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
        if config[2]:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    return best_scores


def read_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    with open(train_path, newline="") as fh:
        for row in csv.DictReader(fh):
            train_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })
    val_rows = []
    with open(val_path, newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })
    quantiles = np.quantile(
        np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64),
        np.linspace(0.1, 0.9, 9),
    )
    maps = []
    for key in ("user_id", "video_id"):
        values = sorted({r[key] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(values)})
    tab_values = sorted({r["tab"] for r in train_rows})
    tab_map = {v: i + 1 for i, v in enumerate(tab_values)}
    field_dims = np.asarray([len(maps[0]) + 1, len(maps[1]) + 1, 1,
                             len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(field_dims)[:-1]]

    def encode(rows):
        X = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            X[i, 0] = maps[0].get(row["user_id"], 0)
            X[i, 1] = maps[1].get(row["video_id"], 0)
            X[i, 2] = 0
            X[i, 3] = tab_map.get(row["tab"], 0)
            X[i, 4] = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
        return X + offsets

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "train_user": np.asarray([r["user_id"] for r in train_rows]),
        "train_date": np.asarray([r["date"] for r in train_rows]),
        "Xv": encode(val_rows),
        "val_y": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "val_user": np.asarray([r["user_id"] for r in val_rows]),
        "val_video": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    npz_fast = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                os.path.exists(os.path.join(args.data_dir, "val.npz")))
    if npz_fast:
        from data.official.evaluate import evaluate
        tr = np.load(os.path.join(args.data_dir, "train.npz"), allow_pickle=False)
        va = np.load(os.path.join(args.data_dir, "val.npz"), allow_pickle=False)
        Xt_np = tr["X"].astype(np.int64)
        yt_np = tr["y"].astype(np.float32)
        train_users = tr["user"]
        train_dates = tr["date"]
        Xv_np = va["X"].astype(np.int64)
        val_y = va["y"].astype(np.int64)
        val_users = va["user"]
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        val_videos = Xv_np[:, 1] - video_offset
    else:
        from harness.evaluate_provisional import evaluate
        loaded = read_csv_data(args.data_dir)
        Xt_np = loaded["Xt"]
        yt_np = loaded["yt"]
        train_users = loaded["train_user"]
        train_dates = loaded["train_date"]
        Xv_np = loaded["Xv"]
        val_y = loaded["val_y"]
        val_users = loaded["val_user"]
        val_videos = loaded["val_video"]
        field_dims = loaded["field_dims"]

    Xt = torch.from_numpy(Xt_np)
    yt = torch.from_numpy(yt_np)
    Xv = torch.from_numpy(Xv_np)
    recency = torch.from_numpy(make_recency_weights(train_dates))
    total_dim = int(field_dims.sum())

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    probe_epochs = 2
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
    probe_epochs = max(1, probe_epochs)
    final_epochs = max(1, final_epochs)

    rng = np.random.RandomState(args.seed + 211)
    probe_size = max(1, len(yt_np) // 2)
    probe_idx = np.sort(rng.choice(len(yt_np), size=probe_size, replace=False))
    probe_arrays = (
        torch.from_numpy(Xt_np[probe_idx]),
        torch.from_numpy(yt_np[probe_idx]),
        np.asarray(train_users)[probe_idx],
        torch.from_numpy(make_recency_weights(np.asarray(train_dates)[probe_idx])),
        Xv,
        val_users,
        val_y,
        total_dim,
    )
    configs = [
        (dropout, weight_decay, decay)
        for dropout in (0.2, 0.3)
        for weight_decay in (1e-4, 1e-3)
        for decay in (False, True)
    ]
    probe_results = []
    for config in configs:
        value = run_probe(config, probe_arrays, evaluate, probe_epochs, args.seed + 307)
        probe_results.append(value)
    winner = configs[int(np.argmax(np.asarray(probe_results)))]

    full_arrays = (Xt, yt, train_users, recency, Xv, val_users, val_y, total_dim)
    best_scores = run_final(winner, full_arrays, evaluate, final_epochs, args.seed + 401)
    metrics = evaluate(val_users, val_y, best_scores)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
            "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
            "primary": float(metrics["primary"]),
        }, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, val_users[i], val_videos[i], "%.9g" % float(score)])


if __name__ == "__main__":
    main()
