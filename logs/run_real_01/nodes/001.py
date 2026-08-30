"""FM baseline with a 0.5 logloss / 0.5 within-user BPR objective.

Uses the five pre-encoded fields from the workspace NPZ fast path and selects the
best epoch by validation primary score.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


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


def build_user_groups(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    ends = np.concatenate((boundaries, np.array([len(order)])))
    groups = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos, neg))
    return groups


def sample_pairs(groups, rng):
    positives = []
    negatives = []
    for pos, neg in groups:
        count = max(len(pos), len(neg))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    return (torch.from_numpy(np.concatenate(positives).astype(np.int64)),
            torch.from_numpy(np.concatenate(negatives).astype(np.int64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.int64))
    groups = build_user_groups(tr["user"], tr["y"])

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    best = -1.0
    best_scores = None
    patience = 0

    for epoch in range(a.epochs):
        model.train()
        pos_idx, neg_idx = sample_pairs(groups, rng)
        pair_count = len(pos_idx)
        perm = torch.randperm(n)

        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pair_sel = torch.randint(pair_count, (len(idx),))
            pidx = pos_idx[pair_sel]
            nidx = neg_idx[pair_sel]

            opt.zero_grad()
            point_loss = bce(model(Xt[idx]), yt[idx])
            pair_x = torch.cat((Xt[pidx], Xt[nidx]), dim=0)
            pair_scores = model(pair_x)
            half = len(pidx)
            bpr_loss = torch.nn.functional.softplus(
                -(pair_scores[:half] - pair_scores[half:])
            ).mean()
            loss = 0.5 * point_loss + 0.5 * bpr_loss
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(va["user"], va["y"].astype(int), scores)
        primary = metrics["primary"]
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
            "primary": metrics["primary"]
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
