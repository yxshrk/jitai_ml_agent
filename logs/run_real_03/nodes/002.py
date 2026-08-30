"""FM baseline augmented with a DCN-lite residual interaction head.

Uses the five offset-encoded fields from the official NPZ fast path and selects
checkpoints by validation primary score.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FMDCNLite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16, hidden=128):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        width = num_fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.zeros_(self.mlp[2].weight)
        torch.nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        x0 = e.flatten(1)
        cross = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True)
        cross = cross + self.cross_b
        return fm + self.mlp(cross).squeeze(1)


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
    field_dims = tr["field_dims"].astype(np.int64)
    total_dim = int(field_dims.sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    model = FMDCNLite(total_dim, num_fields=Xt.shape[1])
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
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])
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
            "primary": metrics["primary"],
        }, fh)

    video_offset = int(field_dims[0])
    video_ids = va["X"][:, 1].astype(np.int64) - video_offset
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},{video_ids[i]},{score:.6g}\n")


if __name__ == "__main__":
    main()
