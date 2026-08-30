"""Aggressively regularized FM using the official NPZ fast path."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.30):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair

    def accessed_row_l2(self, x):
        rows = torch.unique(x.reshape(-1))
        return self.emb(rows).pow(2).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    model = FM(total_dim, k=16, dropout=0.30)
    opt = torch.optim.AdamW(
        [
            {"params": model.emb.parameters(), "weight_decay": 0.0},
            {"params": model.lin.parameters(), "weight_decay": 1e-3},
            {"params": [model.bias], "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.75)
    bce = torch.nn.BCEWithLogitsLoss()

    n = len(yt)
    bs = 8192
    best_gauc = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        last_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yt[idx]) + 1e-3 * model.accessed_row_l2(xb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).cpu().numpy()
                for i in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(va["user"], va["y"].astype(int), scores)
        gauc = metrics.get("GAUC", metrics.get("gauc"))
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 5),
            "lr": round(float(opt.param_groups[0]["lr"]), 9),
            "val_gauc": round(float(gauc), 6),
            "val_primary": round(float(metrics["primary"]), 6),
        })
        if gauc > best_gauc + 1e-6:
            best_gauc = float(gauc)
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    os.makedirs(a.out_dir, exist_ok=True)
    metrics = evaluate(va["user"], va["y"].astype(int), best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": metrics.get("GAUC", metrics.get("gauc")),
            "ndcg5": metrics.get("nDCG@5", metrics.get("ndcg5")),
            "primary": metrics["primary"],
            "history": history,
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
