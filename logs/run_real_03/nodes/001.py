"""FM baseline with hybrid pointwise log-loss and within-user BPR loss."""
import argparse, json, os, sys
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


def build_user_groups(users, labels):
    labels = labels.astype(np.int8)
    order = np.lexsort((labels, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    sizes = ends - starts
    positives = np.add.reduceat(labels[order].astype(np.int64), starts)
    negatives = sizes - positives
    keep = (positives > 0) & (negatives > 0)
    return (order, starts[keep], negatives[keep].astype(np.int64),
            positives[keep].astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.RandomState(a.seed)

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    Xt = torch.from_numpy(tr["X"].astype(np.int64))
    yt = torch.from_numpy(tr["y"].astype(np.float32))
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    pair_order, pair_starts, pair_neg, pair_pos = build_user_groups(
        tr["user"], tr["y"]
    )

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
            pair_bs = max(1, len(idx) // 2)
            groups = rng.randint(0, len(pair_starts), size=pair_bs)
            neg_offsets = (rng.random_sample(pair_bs) * pair_neg[groups]).astype(np.int64)
            pos_offsets = (rng.random_sample(pair_bs) * pair_pos[groups]).astype(np.int64)
            neg_idx = pair_order[pair_starts[groups] + neg_offsets]
            pos_idx = pair_order[pair_starts[groups] + pair_neg[groups] + pos_offsets]
            pair_idx = torch.from_numpy(np.concatenate([pos_idx, neg_idx]).astype(np.int64))

            opt.zero_grad()
            point_loss = bce(model(Xt[idx]), yt[idx])
            pair_scores = model(Xt[pair_idx])
            pos_scores = pair_scores[:pair_bs]
            neg_scores = pair_scores[pair_bs:]
            pair_loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
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
