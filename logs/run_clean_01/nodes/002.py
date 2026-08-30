"""FM baseline with a hybrid pointwise BCE and within-user BPR objective."""
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


def make_within_user_pairs(users, labels, rng, max_pairs=200000):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), cuts))
    ends = np.concatenate((cuts, np.array([len(order)], dtype=np.int64)))
    positives = []
    negatives = []
    total = 0
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = min(max(len(pos), len(neg)), max_pairs - total)
        if count <= 0:
            break
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
        total += count
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return (np.concatenate(positives).astype(np.int64, copy=False),
            np.concatenate(negatives).astype(np.int64, copy=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    epochs = a.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, int(os.environ["SMOKE_EPOCHS"]))

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    total_dim = int(tr["field_dims"].sum())
    train_x_np = tr["X"].astype(np.int64)
    train_y_np = tr["y"].astype(np.float32)
    train_users = np.asarray(tr["user"])
    Xt = torch.from_numpy(train_x_np)
    yt = torch.from_numpy(train_y_np)
    Xv = torch.from_numpy(va["X"].astype(np.int64))

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    bs = 8192
    pair_bs = 1024
    best, best_scores, patience = -1.0, None, 0
    history = []

    for epoch in range(epochs):
        rng = np.random.RandomState(a.seed + epoch)
        pos_np, neg_np = make_within_user_pairs(train_users, train_y_np, rng)
        pos_idx = torch.from_numpy(pos_np)
        neg_idx = torch.from_numpy(neg_np)
        pair_count = len(pos_idx)

        model.train()
        perm = torch.randperm(n)
        pair_perm = torch.randperm(pair_count) if pair_count else None
        pair_cursor = 0
        last_loss = None
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = bce(model(Xt[idx]), yt[idx])

            if pair_count:
                take = min(pair_bs, pair_count)
                if pair_cursor + take <= pair_count:
                    selected = pair_perm[pair_cursor:pair_cursor + take]
                    pair_cursor += take
                else:
                    first = pair_perm[pair_cursor:]
                    pair_perm = torch.randperm(pair_count)
                    remaining = take - len(first)
                    selected = torch.cat((first, pair_perm[:remaining]))
                    pair_cursor = remaining
                pair_x = torch.cat((Xt[pos_idx[selected]], Xt[neg_idx[selected]]), dim=0)
                pair_scores = model(pair_x)
                pos_scores = pair_scores[:take]
                neg_scores = pair_scores[take:]
                bpr_loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
                loss = loss + 0.3 * bpr_loss

            loss.backward()
            opt.step()
            last_loss = loss

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([model(Xv[i:i + 65536]).numpy()
                                     for i in range(0, len(Xv), 65536)])
        m = evaluate(va["user"], va["y"].astype(int), scores)
        primary = m["primary"]
        history.append({"epoch": epoch + 1, "train_loss": round(float(last_loss.item()), 5),
                        "val_gauc": round(m.get("GAUC", m.get("gauc", 0.0)), 6),
                        "val_primary": round(primary, 6)})
        if primary > best + 1e-6 or best_scores is None:
            best, best_scores, patience = primary, scores, 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(a.out_dir, exist_ok=True)
    m = evaluate(va["user"], va["y"].astype(int), best_scores)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": m["GAUC"] if "GAUC" in m else m["gauc"],
                   "ndcg5": m.get("nDCG@5", m.get("ndcg5")),
                   "primary": m["primary"], "history": history}, fh)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, s in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{s:.6g}\n")


if __name__ == "__main__":
    main()
