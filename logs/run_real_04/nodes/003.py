"""FM baseline with a hybrid binary and per-user listwise softmax objective."""
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

    train_users = np.asarray(tr["user"])
    sorted_idx = np.argsort(train_users, kind="stable")
    sorted_users = train_users[sorted_idx]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(sorted_idx)]))
    groups = [sorted_idx[s:e] for s, e in zip(starts, ends)]

    model = FM(total_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    bs = 8192
    best, best_scores, patience = -1.0, None, 0

    for epoch in range(a.epochs):
        model.train()
        group_order = torch.randperm(len(groups)).numpy()
        pending = []
        pending_count = 0

        def train_batch(batch_groups):
            idx_np = np.concatenate(batch_groups).astype(np.int64, copy=False)
            lengths_np = np.fromiter((len(g) for g in batch_groups), dtype=np.int64)
            gid_np = np.repeat(np.arange(len(batch_groups), dtype=np.int64), lengths_np)
            idx = torch.from_numpy(idx_np)
            gid = torch.from_numpy(gid_np)
            lengths = torch.from_numpy(lengths_np)
            labels = yt[idx]

            opt.zero_grad()
            logits = model(Xt[idx])
            binary_loss = bce(logits, labels)

            group_count = len(batch_groups)
            max_score = torch.full((group_count,), -torch.inf, dtype=logits.dtype)
            max_score.scatter_reduce_(0, gid, logits.detach(), reduce="amax", include_self=True)
            exp_sum = torch.zeros(group_count, dtype=logits.dtype)
            exp_sum.scatter_add_(0, gid, torch.exp(logits - max_score[gid]))
            log_partition = max_score + torch.log(exp_sum.clamp_min(1e-12))

            positive_count = torch.zeros(group_count, dtype=logits.dtype)
            positive_count.scatter_add_(0, gid, labels)
            positive_score_sum = torch.zeros(group_count, dtype=logits.dtype)
            positive_score_sum.scatter_add_(0, gid, logits * labels)
            valid = (positive_count > 0) & (positive_count < lengths.to(logits.dtype))
            if bool(valid.any()):
                listwise_loss = (
                    log_partition[valid]
                    - positive_score_sum[valid] / positive_count[valid]
                ).mean()
                loss = 0.5 * binary_loss + 0.5 * listwise_loss
            else:
                loss = binary_loss
            loss.backward()
            opt.step()

        for group_id in group_order:
            group = groups[int(group_id)]
            if pending and pending_count + len(group) > bs:
                train_batch(pending)
                pending = []
                pending_count = 0
            pending.append(group)
            pending_count += len(group)
        if pending:
            train_batch(pending)

        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[i:i + 65536]).numpy()
                for i in range(0, len(Xv), 65536)
            ])
        m = evaluate(va["user"], va["y"].astype(int), scores)
        primary = float(m["primary"])
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
            "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
            "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
            "primary": float(m["primary"])
        }, fh)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{va['user'][i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
