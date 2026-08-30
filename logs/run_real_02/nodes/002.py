"""DCNv2-lite model over the official five-field NPZ representation.

Uses the same data, embedding size, pointwise logloss, optimizer, and validation
model selection as the parent FM, changing only the interaction architecture.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNv2Lite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=5, k=16, hidden=128):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        input_dim = num_fields * k
        self.cross = torch.nn.Linear(input_dim, input_dim)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(input_dim + hidden, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.cross.weight)
        torch.nn.init.zeros_(self.cross.bias)
        torch.nn.init.xavier_uniform_(self.deep[0].weight)
        torch.nn.init.zeros_(self.deep[0].bias)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x0 = self.emb(x).flatten(1)
        cross = x0 * self.cross(x0) + x0
        deep = self.deep(x0)
        interaction = self.out(torch.cat((cross, deep), dim=1)).squeeze(1)
        first_order = self.lin(x).sum((1, 2))
        return self.bias + first_order + interaction


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

    model = DCNv2Lite(total_dim)
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
            loss = bce(model(Xt[idx]), yt[idx])
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
