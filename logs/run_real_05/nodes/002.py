"""FM baseline with a 0.5 BCE + 0.5 within-user BPR hybrid objective."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))

    total_dim = int(tr["field_dims"].sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    train_users = np.asarray(tr["user"])
    _, user_inverse = np.unique(train_users, return_inverse=True)
    num_users = int(user_inverse.max()) + 1
    labels_np = np.asarray(tr["y"]) >= 0.5

    pos_indices = np.flatnonzero(labels_np)
    neg_indices = np.flatnonzero(~labels_np)
    pos_counts = np.bincount(user_inverse[pos_indices], minlength=num_users)
    neg_counts = np.bincount(user_inverse[neg_indices], minlength=num_users)
    eligible_users = (pos_counts > 0) & (neg_counts > 0)
    eligible_pos = pos_indices[eligible_users[user_inverse[pos_indices]]]

    neg_order = neg_indices[
        np.argsort(user_inverse[neg_indices], kind="stable")
    ]
    neg_starts = np.zeros(num_users, dtype=np.int64)
    if num_users > 1:
        neg_starts[1:] = np.cumsum(neg_counts[:-1], dtype=np.int64)

    eligible_pos_t = torch.from_numpy(eligible_pos.astype(np.int64))
    eligible_pos_users_t = torch.from_numpy(
        user_inverse[eligible_pos].astype(np.int64)
    )
    neg_order_t = torch.from_numpy(neg_order.astype(np.int64))
    neg_starts_t = torch.from_numpy(neg_starts)
    neg_counts_t = torch.from_numpy(neg_counts.astype(np.int64))

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
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pair_count = len(idx)

            sampled_pos_slots = torch.randint(
                0, len(eligible_pos_t), (pair_count,)
            )
            pair_pos = eligible_pos_t[sampled_pos_slots]
            pair_users = eligible_pos_users_t[sampled_pos_slots]
            counts = neg_counts_t[pair_users]
            random_offsets = (
                torch.rand(pair_count) * counts.to(torch.float32)
            ).to(torch.int64)
            pair_neg = neg_order_t[neg_starts_t[pair_users] + random_offsets]

            all_x = torch.cat((Xt[idx], Xt[pair_pos], Xt[pair_neg]), dim=0)
            logits = model(all_x)
            batch_count = len(idx)
            impression_logits = logits[:batch_count]
            pos_logits = logits[batch_count:batch_count + pair_count]
            neg_logits = logits[batch_count + pair_count:]

            point_loss = bce(impression_logits, yt[idx])
            pair_loss = torch.nn.functional.softplus(
                -(pos_logits - neg_logits)
            ).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss

            opt.zero_grad()
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
