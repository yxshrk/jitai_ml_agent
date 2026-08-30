"""DCN-lite recommender with a hybrid pointwise and within-user BPR loss.

Uses the five offset-encoded fields from the official NPZ fast path, one DCN
cross layer, an MLP128 branch, and validation-GAUC early stopping.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16, hidden=128):
        super().__init__()
        self.num_fields = num_fields
        self.k = k
        d = num_fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.Parameter(torch.empty(d))
        self.cross_b = torch.nn.Parameter(torch.zeros(d))
        self.cross_out = torch.nn.Linear(d, 1, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x0 = self.emb(x).reshape(x.shape[0], -1)
        cross_scale = torch.matmul(x0, self.cross_w).unsqueeze(1)
        x1 = x0 * cross_scale + self.cross_b + x0
        linear = self.lin(x).sum((1, 2))
        cross_score = self.cross_out(x1).squeeze(1)
        deep_score = self.mlp(x0).squeeze(1)
        return self.bias + linear + cross_score + deep_score


def make_user_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.array([0]), cuts))
    ends = np.concatenate((cuts, np.array([len(order)])))
    pos_parts = []
    neg_parts = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (np.concatenate(pos_parts).astype(np.int64),
            np.concatenate(neg_parts).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt_np = tr["y"].astype(np.float32)
    yt = torch.from_numpy(yt_np)
    xv = torch.from_numpy(va["X"].astype(np.int64))

    pair_rng = np.random.RandomState(a.seed + 1)
    pair_pos_np, pair_neg_np = make_user_pairs(tr["user"], yt_np, pair_rng)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)

    model = DCNLite(total_dim=total_dim, num_fields=xt.shape[1], k=16, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    pair_n = len(pair_pos)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for _ in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        pair_perm = torch.randperm(pair_n) if pair_n else None
        pair_cursor = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            point_loss = bce(model(xt[idx]), yt[idx])
            if pair_n:
                need = len(idx)
                selected_parts = []
                while need > 0:
                    available = pair_n - pair_cursor
                    take = min(need, available)
                    selected_parts.append(pair_perm[pair_cursor:pair_cursor + take])
                    pair_cursor += take
                    need -= take
                    if pair_cursor == pair_n:
                        pair_perm = torch.randperm(pair_n)
                        pair_cursor = 0
                pidx = torch.cat(selected_parts)
                pos_scores = model(xt[pair_pos[pidx]])
                neg_scores = model(xt[pair_neg[pidx]])
                pair_loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(xv[i:i + 65536]).numpy()
                for i in range(0, len(xv), 65536)
            ])
        metrics = evaluate(va["user"], va["y"].astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(a.out_dir, exist_ok=True)
    metrics = evaluate(va["user"], va["y"].astype(int), best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"],
            "ndcg5": metrics.get("nDCG@5", metrics.get("ndcg5")),
            "primary": metrics["primary"],
        }, fh)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
