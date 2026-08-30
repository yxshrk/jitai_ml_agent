"""FM baseline with an ordinal watch-ratio auxiliary training objective.

Uses the five offset-encoded fast-path fields and trains the primary long_view
logloss together with cumulative watch-ratio targets derived only from training
outcomes. Validation outcomes other than long_view are never read.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, ordinal_levels=5):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.ordinal_head = torch.nn.Linear(k, ordinal_levels)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.ordinal_head.weight)
        torch.nn.init.zeros_(self.ordinal_head.bias)

    def forward(self, x, return_ordinal=False):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        main = self.bias + self.lin(x).sum((1, 2)) + pair
        if return_ordinal:
            return main, self.ordinal_head(s)
        return main


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
    ratio = np.nan_to_num(play / denominator, nan=0.0, posinf=1.0, neginf=0.0)
    ratio = np.clip(ratio, 0.0, 1.0)
    thresholds = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    ordinal_targets = (ratio[:, None] >= thresholds[None, :]).astype(np.float32)
    ot = torch.from_numpy(ordinal_targets)

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    best, best_scores, patience = -1.0, None, 0

    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            main_logits, ordinal_logits = model(Xt[idx], return_ordinal=True)
            loss = bce(main_logits, yt[idx]) + 0.3 * bce(ordinal_logits, ot[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        m = evaluate(va["user"], va["y"].astype(int), scores)
        primary = m["primary"]
        if primary > best + 1e-6:
            best, best_scores, patience = primary, scores, 0
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
            "primary": m["primary"]
        }, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, s in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{s:.6g}\n")


if __name__ == "__main__":
    main()
