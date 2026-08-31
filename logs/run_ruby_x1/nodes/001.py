import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.cross_out = torch.nn.Linear(width, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.zeros_(self.cross_out.weight)
        torch.nn.init.zeros_(self.cross_out.bias)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        e = self.emb_dropout(self.emb(x))
        summed = e.sum(1)
        fm = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        linear = self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b + x0
        deep = self.mlp(x0).squeeze(1)
        return self.bias + linear + fm + self.cross_out(cross).squeeze(1) + deep


def metric_dict(user, labels, scores):
    raw = evaluate(user, labels, scores)
    return {
        "gauc": float(raw["GAUC"] if "GAUC" in raw else raw["gauc"]),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw["primary"]),
    }


def date_recency_weights(values, half_life):
    vals = np.asarray(values)
    unique = np.unique(vals)
    parsed = {}
    for value in unique:
        text = str(int(value))
        try:
            parsed[value] = datetime.strptime(text, "%Y%m%d").toordinal()
        except ValueError:
            parsed[value] = int(value)
    latest = max(parsed.values())
    ages = np.fromiter((latest - parsed[v] for v in vals), dtype=np.float32, count=len(vals))
    weights = np.exp2(-ages / float(half_life)).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def build_user_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        shuffled_neg = neg[rng.permutation(len(neg))]
        paired_neg = np.resize(shuffled_neg, len(pos))
        positives.append(pos.astype(np.int64, copy=False))
        negatives.append(paired_neg.astype(np.int64, copy=False))
    if not positives:
        raise RuntimeError("No users with both positive and negative training impressions")
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, batch_size=131072):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            chunks.append(model(x[start:start + batch_size]).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64, copy=False)


def train_one(config, seed, epochs, tensors, arrays, device, keep_scores):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    model = DCNLite(
        arrays["total_dim"],
        fields=arrays["fields"],
        k=16,
        hidden=128,
        dropout=float(config["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    recency_np = date_recency_weights(arrays["date"], float(config["half_life"]))
    recency = torch.from_numpy(recency_np).to(device)
    xt = tensors["xt"]
    yt = tensors["yt"]
    xv = tensors["xv"]
    pair_pos = arrays["pair_pos"]
    pair_neg = arrays["pair_neg"]
    n = len(yt)
    batch_size = 65536
    last_loss = 0.0
    half_step = 0
    for epoch in range(epochs):
        permutation = rng.permutation(n)
        split = (n + 1) // 2
        for indices in (permutation[:split], permutation[split:]):
            model.train()
            loss_sum = 0.0
            steps = 0
            for start in range(0, len(indices), batch_size):
                batch_np = indices[start:start + batch_size]
                if len(batch_np) == 0:
                    continue
                pair_choice = rng.randint(0, len(pair_pos), size=len(batch_np))
                pos_np = pair_pos[pair_choice]
                neg_np = pair_neg[pair_choice]
                batch = torch.from_numpy(batch_np.astype(np.int64, copy=False)).to(device)
                pos_idx = torch.from_numpy(pos_np.astype(np.int64, copy=False)).to(device)
                neg_idx = torch.from_numpy(neg_np.astype(np.int64, copy=False)).to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xt[batch])
                bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt[batch], reduction="none")
                bce_loss = (bce_each * recency[batch]).sum() / recency[batch].sum().clamp_min(1e-8)
                pos_scores = model(xt[pos_idx])
                neg_scores = model(xt[neg_idx])
                pair_each = torch.nn.functional.softplus(-(pos_scores - neg_scores))
                pair_weight = 0.5 * (recency[pos_idx] + recency[neg_idx])
                pair_loss = (pair_each * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                loss = 0.5 * bce_loss + 0.5 * pair_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                loss_sum += float(loss.detach().cpu())
                steps += 1
            last_loss = loss_sum / max(steps, 1)
            half_step += 1
            if half_step % int(config["step_interval"]) == 0:
                for group in opt.param_groups:
                    group["lr"] *= float(config["step_gamma"])

    scores = predict(model, xv)
    metrics = metric_dict(arrays["val_user"], arrays["val_y"], scores)
    checkpoints = [{
        "epoch": float(epochs),
        "train_loss": round(float(last_loss), 6),
        "lr": float(opt.param_groups[0]["lr"]),
        "gauc": round(metrics["gauc"], 6),
        "ndcg5": round(metrics["ndcg5"], 6),
        "primary": round(metrics["primary"], 6),
    }]
    return metrics["primary"], (scores if keep_scores else None), checkpoints


def rank_scores(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def clean_config(config):
    return {
        "dropout": round(float(config["dropout"]), 6),
        "weight_decay": float(config["weight_decay"]),
        "lr": float(config["lr"]),
        "half_life": round(float(config["half_life"]), 6),
        "step_interval": int(config["step_interval"]),
        "step_gamma": round(float(config["step_gamma"]), 6),
    }


def append_progress(path, stage, index, config, primary):
    record = {"stage": stage, "probe": index, "config": clean_config(config), "primary": round(float(primary), 8)}
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(args.data_dir, "train.npz")
    val_path = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("This package-dial sweep requires the train.npz and val.npz fast path")
    train = np.load(train_path)
    val = np.load(val_path)
    train_x = train["X"].astype(np.int64, copy=False)
    val_x = val["X"].astype(np.int64, copy=False)
    train_y = train["y"].astype(np.float32, copy=False)
    val_y = val["y"].astype(np.int64, copy=False)
    pair_pos, pair_neg = build_user_pairs(train["user"], train_y, args.seed + 7919)

    tensors = {
        "xt": torch.from_numpy(train_x).to(device),
        "yt": torch.from_numpy(train_y).to(device),
        "xv": torch.from_numpy(val_x).to(device),
    }
    arrays = {
        "date": train["date"],
        "pair_pos": pair_pos,
        "pair_neg": pair_neg,
        "val_user": val["user"],
        "val_y": val_y,
        "total_dim": int(train["field_dims"].sum()),
        "fields": int(train_x.shape[1]),
    }

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    train_epochs = max(1, min(args.epochs, smoke_cap if smoke_cap is not None else 4))

    search_rng = np.random.RandomState(args.seed + 104729)
    config = {
        "dropout": float(search_rng.uniform(0.13, 0.42)),
        "weight_decay": float(10.0 ** search_rng.uniform(math.log10(3e-5), math.log10(3e-3))),
        "lr": float(search_rng.choice(np.array([4.5e-4, 6.5e-4, 9.0e-4, 1.2e-3, 1.6e-3], dtype=np.float64))),
        "half_life": float(search_rng.choice(np.array([3.5, 5.0, 7.0, 10.0, 14.0], dtype=np.float64))),
        "step_interval": int(search_rng.choice([2, 3, 4])),
        "step_gamma": float(search_rng.uniform(0.24, 0.68)),
    }

    primary, selected_scores, checkpoints = train_one(config, args.seed + 1000, train_epochs, tensors, arrays, device, True)
    selected_metrics = metric_dict(arrays["val_user"], arrays["val_y"], selected_scores)

    history = [{
        "stage": "train",
        "seed": args.seed + 1000,
        "epochs": train_epochs,
        "config": clean_config(config),
        "best_primary": round(float(primary), 8),
        "checkpoints": checkpoints,
    }]
    append_progress(progress_path, "train", 0, config, primary)

    metrics_payload = {
        "gauc": selected_metrics["gauc"],
        "ndcg5": selected_metrics["ndcg5"],
        "primary": selected_metrics["primary"],
        "selected_output": "best_checkpoint",
        "winning_config": clean_config(config),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics_payload, handle)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(selected_scores):
            handle.write(f"{row_id},{val['user'][row_id]},0,{float(score):.9g}\n")


if __name__ == "__main__":
    main()
