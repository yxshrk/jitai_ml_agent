"""FM baseline with an ordinal watch-ratio auxiliary objective.

Uses the five offset-encoded fields from the NPZ fast path. The primary long-view
BCE is augmented by a 0.3-weight cumulative ordinal loss predicting watch-ratio
thresholds from shared embeddings. Model selection uses validation GAUC/primary.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FMOrdinal(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16, num_thresholds=5):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.ordinal = torch.nn.Linear(num_fields * k, num_thresholds)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.ordinal.weight)
        torch.nn.init.zeros_(self.ordinal.bias)

    def forward(self, x, return_ordinal=False):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        primary = self.bias + self.lin(x).sum((1, 2)) + pair
        if return_ordinal:
            return primary, self.ordinal(e.flatten(1))
        return primary


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

    play = tr["play_time_ms"].astype(np.float32)
    duration = tr["duration_ms"].astype(np.float32)
    denominator = np.maximum(np.minimum(duration, 18000.0), 1.0)
    ratio = np.clip(np.nan_to_num(play / denominator, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    thresholds = np.asarray([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    ordinal_targets = torch.from_numpy((ratio[:, None] >= thresholds[None, :]).astype(np.float32))

    model = FMOrdinal(total_dim)
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
            primary_logits, ordinal_logits = model(Xt[idx], return_ordinal=True)
            primary_loss = bce(primary_logits, yt[idx])
            ordinal_loss = bce(ordinal_logits, ordinal_targets[idx])
            loss = primary_loss + 0.3 * ordinal_loss
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
