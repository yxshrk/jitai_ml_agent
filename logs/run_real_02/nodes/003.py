"""DCNv2-lite with categorical hour-of-day and day-of-week context."""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNv2Lite(torch.nn.Module):
    def __init__(self, total_dim, num_fields=7, k=16, hidden=128):
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


def make_temporal_fields(hourmin, dates, base_offset, hour_is_hhmm):
    hm = np.asarray(hourmin).astype(np.int64)
    if hour_is_hhmm:
        hours = (hm // 100) % 24
    else:
        hours = (hm // 60) % 24

    raw_dates = np.asarray(dates).astype(np.int64)
    weekdays = np.empty(len(raw_dates), dtype=np.int64)
    cache = {}
    for i, value in enumerate(raw_dates):
        key = int(value)
        if key not in cache:
            text = str(key).zfill(8)
            try:
                cache[key] = datetime.date(
                    int(text[:4]), int(text[4:6]), int(text[6:8])
                ).weekday()
            except ValueError:
                cache[key] = key % 7
        weekdays[i] = cache[key]

    return np.column_stack((
        base_offset + hours,
        base_offset + 24 + weekdays,
    )).astype(np.int64)


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
    hour_is_hhmm = int(np.max(tr["hourmin"])) > 1439

    train_time = make_temporal_fields(
        tr["hourmin"], tr["date"], base_dim, hour_is_hhmm
    )
    val_time = make_temporal_fields(
        va["hourmin"], va["date"], base_dim, hour_is_hhmm
    )
    train_x = np.concatenate((tr["X"].astype(np.int64), train_time), axis=1)
    val_x = np.concatenate((va["X"].astype(np.int64), val_time), axis=1)

    total_dim = base_dim + 24 + 7
    Xt = torch.from_numpy(train_x)
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(val_x)

    model = DCNv2Lite(total_dim)
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

    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
