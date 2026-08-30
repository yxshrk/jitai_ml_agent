import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    Xtr = np.asarray(tr["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    Xva = np.asarray(va["X"], dtype=np.int64)
    yva = np.asarray(va["y"], dtype=np.float32)
    uva = np.asarray(va["user"])
    if "field_dims" in tr.files:
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    else:
        field_dims = np.asarray([int(Xtr[:, j].max()) + 1 for j in range(Xtr.shape[1])], dtype=np.int64)
    video_out = Xva[:, 1].astype(str)
    return Xtr, ytr, utr, Xva, yva, uva, field_dims, uva.astype(str), video_out


def read_csv_rows(path, is_train):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "user_id": r.get("user_id", ""),
                "video_id": r.get("video_id", ""),
                "author_id": r.get("author_id", r.get("video_id", "")),
                "tab": r.get("tab", ""),
                "duration_ms": float(r.get("duration_ms", 0) or 0),
                "long_view": float(r.get("long_view", 0) or 0),
            }
            rows.append(row)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    fields = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    dims = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        m = {v: i + 1 for i, v in enumerate(values)}
        maps.append(m)
        dims.append(len(m) + 1)
    dims.append(10)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            for j, field in enumerate(fields):
                X[i, j] = maps[j].get(r[field], 0) + offsets[j]
            X[i, 4] = int(np.searchsorted(quantiles, r["duration_ms"], side="right")) + offsets[4]
        return X

    Xtr = encode(train_rows)
    Xva = encode(val_rows)
    ytr = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    yva = np.asarray([r["long_view"] for r in val_rows], dtype=np.float32)
    utr = np.asarray([r["user_id"] for r in train_rows])
    uva = np.asarray([r["user_id"] for r in val_rows])
    user_out = np.asarray([r["user_id"] for r in val_rows], dtype=str)
    video_out = np.asarray([r["video_id"] for r in val_rows], dtype=str)
    return Xtr, ytr, utr, Xva, yva, uva, np.asarray(dims, dtype=np.int64), user_out, video_out


class CrossLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DCNLite(nn.Module):
    def __init__(self, total_dim, n_fields=5, emb_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(total_dim, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        width = n_fields * emb_dim
        self.cross1 = CrossLayer(width)
        self.cross2 = CrossLayer(width)
        self.deep = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.output = nn.Linear(width + 64, 1)

    def forward(self, x):
        x0 = self.embedding(x).flatten(1)
        cross = self.cross1(x0, x0)
        cross = self.cross2(x0, cross)
        deep = self.deep(x0)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


def make_user_pair_pools(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    if len(order) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg)
    return positives, negatives


def make_pairs(pos_groups, neg_groups, rng):
    pos_parts = []
    neg_parts = []
    for pos, neg in zip(pos_groups, neg_groups):
        pos_parts.append(pos)
        neg_parts.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def predict(model, X, device, batch_size=32768):
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            xb = torch.as_tensor(X[start:stop], dtype=torch.long, device=device)
            out[start:stop] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def official_evaluate(users, labels, scores, npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    npz_mode = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_mode:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    Xtr, ytr, utr, Xva, yva, uva, field_dims, user_out, video_out = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(int(np.sum(field_dims))).to(device)
    embedding_params = list(model.embedding.parameters())
    dense_params = [p for name, p in model.named_parameters() if not name.startswith("embedding.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    rng = np.random.default_rng(args.seed)
    pos_groups, neg_groups = make_user_pair_pools(utr, ytr)
    batch_size = 8192
    best_gauc = -1.0
    best_state = None
    stale = 0

    for epoch in range(14):
        model.train()
        order = rng.permutation(len(Xtr))
        pair_pos, pair_neg = make_pairs(pos_groups, neg_groups, rng)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            xb = torch.as_tensor(Xtr[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[ids], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_pos):
                chosen = rng.integers(0, len(pair_pos), size=len(ids))
                pidx = pair_pos[chosen]
                nidx = pair_neg[chosen]
                xp = torch.as_tensor(Xtr[pidx], dtype=torch.long, device=device)
                xn = torch.as_tensor(Xtr[nidx], dtype=torch.long, device=device)
                pair_loss = F.softplus(-(model(xp) - model(xn))).mean()
                accessed = torch.unique(torch.cat([xb.reshape(-1), xp.reshape(-1), xn.reshape(-1)]))
                task_loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                accessed = torch.unique(xb.reshape(-1))
                task_loss = point_loss

            row_vectors = model.embedding.weight[accessed]
            row_l2 = torch.sum(row_vectors * row_vectors, dim=1).mean()
            loss = task_loss + 1e-4 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, Xva, device)
        metrics = official_evaluate(uva, yva, val_scores, npz_mode)
        gauc = float(metrics["GAUC"])
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if epoch >= 5 and stale >= 4:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, Xva, device)
    metrics = official_evaluate(uva, yva, scores, npz_mode)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(user_out, video_out, scores)):
            writer.writerow([i, u, v, format(float(s), ".10g")])

    result = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
