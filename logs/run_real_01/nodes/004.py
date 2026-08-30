"""DCNv2-lite model with a 50-bin train-quantile duration field."""
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
        d = num_fields * k
        self.cross = torch.nn.Linear(d, d)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(d, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(d + hidden, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        x0 = self.emb(x).flatten(1)
        xc = x0 * self.cross(x0) + x0
        xd = self.deep(x0)
        return self.bias + self.lin(x).sum((1, 2)) + self.head(
            torch.cat((xc, xd), dim=1)
        ).squeeze(1)


def replace_duration_field(x, durations, first_four_dim, edges):
    duration_bucket = np.searchsorted(edges, durations, side="right").astype(np.int64)
    out = np.empty((len(x), 5), dtype=np.int64)
    out[:, :4] = x[:, :4].astype(np.int64)
    out[:, 4] = first_four_dim + duration_bucket
    return out


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
    first_four_dim = int(field_dims[:4].sum())
    quantiles = np.arange(1, 50, dtype=np.float64) / 50.0
    edges = np.unique(np.quantile(tr["duration_ms"].astype(np.float64), quantiles))

    Xt_np = replace_duration_field(
        tr["X"], tr["duration_ms"].astype(np.float64), first_four_dim, edges
    )
    Xv_np = replace_duration_field(
        va["X"], va["duration_ms"].astype(np.float64), first_four_dim, edges
    )
    total_dim = first_four_dim + 50

    Xt = torch.from_numpy(Xt_np)
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(Xv_np)

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
