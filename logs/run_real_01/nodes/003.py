"""DCNv2-lite model with temporal context over the official NPZ fast path."""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNv2Lite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=7, k=16, hidden=128):
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


def hour_bucket(hourmin):
    values = np.asarray(hourmin).astype(np.int64).reshape(-1)
    return ((values // 100) % 24).astype(np.int64)


def weekday_bucket(date):
    values = np.asarray(date).astype(np.int64).reshape(-1)
    year = values // 10000
    month = (values // 100) % 100
    day = values % 100
    valid = (year >= 1900) & (month >= 1) & (month <= 12) & (day >= 1) & (day <= 31)
    result = np.mod(values, 7).astype(np.int64)
    if np.any(valid):
        table = np.array([0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4], dtype=np.int64)
        y = year[valid] - (month[valid] < 3).astype(np.int64)
        sunday_zero = (
            y + y // 4 - y // 100 + y // 400
            + table[month[valid] - 1] + day[valid]
        ) % 7
        result[valid] = (sunday_zero + 6) % 7
    return result


def add_temporal_fields(x, hourmin, date, base_dim):
    hour = hour_bucket(hourmin) + base_dim
    weekday = weekday_bucket(date) + base_dim + 24
    return np.column_stack((x.astype(np.int64), hour, weekday)).astype(np.int64)


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
    base_dim = int(tr["field_dims"].sum())
    total_dim = base_dim + 24 + 7
    Xt = torch.from_numpy(add_temporal_fields(
        tr["X"], tr["hourmin"], tr["date"], base_dim
    ))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(add_temporal_fields(
        va["X"], va["hourmin"], va["date"], base_dim
    ))

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
