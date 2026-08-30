"""FM baseline with late-checkpoint stochastic weight averaging.

Uses the same five offset-encoded fields, optimizer, loss, and training schedule as
the parent, but evaluates cumulative averaged weights from epoch 5 onward.
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
    yt = torch.from_numpy(tr["y"])
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    swa_start = 5
    swa_params = None
    swa_count = 0
    best = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

        use_swa = epoch + 1 >= swa_start
        raw_params = None
        if use_swa:
            with torch.no_grad():
                current = [p.detach().clone() for p in model.parameters()]
                if swa_params is None:
                    swa_params = current
                    swa_count = 1
                else:
                    swa_count += 1
                    alpha = 1.0 / float(swa_count)
                    for avg, cur in zip(swa_params, current):
                        avg.add_(cur - avg, alpha=alpha)
                raw_params = current
                for p, avg in zip(model.parameters(), swa_params):
                    p.copy_(avg)

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])

        if use_swa:
            with torch.no_grad():
                for p, raw in zip(model.parameters(), raw_params):
                    p.copy_(raw)

        m = evaluate(va["user"], va["y"].astype(int), scores)
        primary = m["primary"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(loss.item()), 5),
            "val_gauc": round(m.get("GAUC", 0.0), 6),
            "val_primary": round(primary, 6),
            "swa_checkpoints": swa_count if use_swa else 0
        })

        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(a.out_dir, exist_ok=True)
    m = evaluate(va["user"], va["y"].astype(int), best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": m["GAUC"] if "GAUC" in m else m["gauc"],
            "ndcg5": m.get("nDCG@5", m.get("ndcg5")),
            "primary": m["primary"],
            "history": history
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
