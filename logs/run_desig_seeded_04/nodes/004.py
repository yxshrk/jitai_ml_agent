import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    xva = np.asarray(va["X"], dtype=np.int64)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1])))
    videos = xva[:, 1] - offsets[1]
    return xtr, ytr, utr, xva, yva, uva, videos, dims, True


def read_csv_rows(path, train):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", r["video_id"]),
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"] or 0.0),
                "long_view": float(r["long_view"]),
            }
            if train:
                row["click"] = float(r.get("click", 0.0) or 0.0)
            rows.append(row)
    return rows


def load_csv(data_dir):
    tr = read_csv_rows(Path(data_dir) / "train.csv", True)
    va = read_csv_rows(Path(data_dir) / "val.csv", False)
    quantiles = np.quantile(
        np.asarray([r["duration_ms"] for r in tr], dtype=np.float64),
        np.linspace(0.1, 0.9, 9),
    )
    fields = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    for field in fields:
        values = sorted({r[field] for r in tr})
        maps.append({v: i + 1 for i, v in enumerate(values)})
    dims = np.asarray([len(m) + 1 for m in maps] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1])))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for j, (field, mapping) in enumerate(zip(fields, maps)):
            x[:, j] = np.fromiter(
                (mapping.get(r[field], 0) + offsets[j] for r in rows),
                dtype=np.int64,
                count=len(rows),
            )
        x[:, 4] = np.searchsorted(
            quantiles,
            np.asarray([r["duration_ms"] for r in rows]),
            side="right",
        ) + offsets[4]
        return x

    xtr = encode(tr)
    xva = encode(va)
    ytr = np.asarray([r["long_view"] for r in tr], dtype=np.float32)
    yva = np.asarray([r["long_view"] for r in va], dtype=np.float32)
    utr = np.asarray([r["user_id"] for r in tr])
    uva = np.asarray([r["user_id"] for r in va])
    videos = np.asarray([r["video_id"] for r in va])
    return xtr, ytr, utr, xva, yva, uva, videos, dims, False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            rng.shuffle(pos)
            rng.shuffle(neg)
            pos_parts.append(pos)
            neg_parts.append(neg[np.arange(len(pos)) % len(neg)])
        start = end
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class RegularizedDCN(nn.Module):
    def __init__(self, total_values, fields=5, k=16, hidden=128, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(total_values, k)
        d = fields * k
        self.cross_w = nn.Parameter(torch.empty(d))
        self.cross_b = nn.Parameter(torch.zeros(d))
        self.deep1 = nn.Linear(d, hidden)
        self.deep2 = nn.Linear(hidden, hidden // 2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d + hidden // 2, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.cross_w, std=0.01)

    def forward(self, x, return_row_l2=False):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross = x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b + x0
        deep = self.dropout(F.relu(self.deep1(x0)))
        deep = self.dropout(F.relu(self.deep2(deep)))
        logits = self.output(torch.cat((cross, deep), dim=1)).squeeze(1)
        if return_row_l2:
            return logits, emb.square().sum(dim=2).mean()
        return logits


def predict(model, x, device, batch_size=16384):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            out[start:end] = model(xb).detach().cpu().numpy()
    return out


def metric_eval(is_npz, users, labels, scores):
    if is_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def train_member(xtr, ytr, utr, xva, yva, uva, dims, seed, device, is_npz, epochs):
    set_seed(seed)
    model = RegularizedDCN(int(np.sum(dims))).to(device)
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
    pair_pos, pair_neg = make_pairs(utr, ytr, seed)
    rng = np.random.default_rng(seed)
    batch_size = 4096
    best_gauc = -1.0
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        point_order = rng.permutation(len(xtr))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(point_order), batch_size):
            ids = point_order[start:start + batch_size]
            xb = torch.as_tensor(xtr[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[ids], dtype=torch.float32, device=device)
            logits, row_l2 = model(xb, True)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_order):
                need = min(len(ids), len(pair_order))
                if pair_cursor + need > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                pids = pair_pos[pair_order[pair_cursor:pair_cursor + need]]
                nids = pair_neg[pair_order[pair_cursor:pair_cursor + need]]
                pair_cursor += need
                px = torch.as_tensor(xtr[pids], dtype=torch.long, device=device)
                nx = torch.as_tensor(xtr[nids], dtype=torch.long, device=device)
                pair_loss = F.softplus(-(model(px) - model(nx))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-4 * row_l2
            else:
                loss = point_loss + 1e-4 * row_l2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        scores = predict(model, xva, device)
        metrics = metric_eval(is_npz, uva, yva, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 3:
            break

    model.load_state_dict(best_state)
    model.to(device)
    return predict(model, xva, device)


def per_user_rank_average(users, member_scores):
    result = np.zeros(len(users), dtype=np.float64)
    user_order = np.argsort(users, kind="stable")
    sorted_users = users[user_order]
    start = 0
    while start < len(user_order):
        end = start + 1
        while end < len(user_order) and sorted_users[end] == sorted_users[start]:
            end += 1
        idx = user_order[start:end]
        n = len(idx)
        for scores in member_scores:
            local_order = np.argsort(scores[idx], kind="stable")
            ranks = np.empty(n, dtype=np.float64)
            ranks[local_order] = np.arange(n, dtype=np.float64)
            if n > 1:
                ranks /= n - 1
            result[idx] += ranks
        start = end
    result /= len(member_scores)
    return result.astype(np.float32)


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists():
        data = load_npz(data_dir)
    else:
        data = load_csv(data_dir)
    xtr, ytr, utr, xva, yva, uva, videos, dims, is_npz = data

    default_epochs = 12
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = min(default_epochs, max(1, int(smoke))) if smoke is not None else default_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    members = []
    for member in range(5):
        members.append(
            train_member(
                xtr, ytr, utr, xva, yva, uva, dims,
                args.seed + member, device, is_npz, epochs,
            )
        )
    scores = per_user_rank_average(uva, members)
    metrics = metric_eval(is_npz, uva, yva, scores)
    gauc = float(metrics.get("GAUC", metrics.get("gauc")))
    ndcg5 = float(metrics.get("nDCG@5", metrics.get("ndcg5")))
    primary = float(metrics["primary"])

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(uva, videos, scores)):
            writer.writerow([i, user, video, float(score)])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, f)


if __name__ == "__main__":
    main()
