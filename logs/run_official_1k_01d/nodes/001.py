import argparse
import csv
import json
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def parse_date_value(x):
    s = str(x)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("-", "")
    try:
        return datetime.strptime(s[:8], "%Y%m%d").toordinal()
    except Exception:
        return 0


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    dates = np.asarray(tr["date"])
    video_offset = int(dims[0])
    video_val = xva[:, 1] - video_offset
    return xtr, ytr, utr, dates, xva, yva, uva, video_val, dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"] or 0.0),
                "date": r["date"],
                "long_view": float(r["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    if len(durations):
        cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        cuts = np.asarray([], dtype=np.float64)

    maps = []
    for key in ("user_id", "video_id", "author_id", "tab"):
        if key == "author_id":
            maps.append({"__missing__": 1})
        else:
            vals = sorted({r[key] for r in train_rows})
            maps.append({v: i + 1 for i, v in enumerate(vals)})

    dims = np.asarray([len(m) + 1 for m in maps] + [len(cuts) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1])))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = maps[0].get(r["user_id"], 0) + offsets[0]
            x[i, 1] = maps[1].get(r["video_id"], 0) + offsets[1]
            x[i, 2] = maps[2]["__missing__"] + offsets[2]
            x[i, 3] = maps[3].get(r["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(cuts, r["duration_ms"], side="right")) + 1 + offsets[4]
        return x

    xtr = encode(train_rows)
    xva = encode(val_rows)
    ytr = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    yva = np.asarray([r["long_view"] for r in val_rows], dtype=np.float32)
    utr = np.asarray([r["user_id"] for r in train_rows])
    uva = np.asarray([r["user_id"] for r in val_rows])
    dates = np.asarray([r["date"] for r in train_rows])
    videos = np.asarray([r["video_id"] for r in val_rows])
    return xtr, ytr, utr, dates, xva, yva, uva, videos, dims, False


def recency_weights(dates, half_life=7.0):
    ordinals = np.asarray([parse_date_value(v) for v in dates], dtype=np.float64)
    valid = ordinals > 0
    if not valid.any():
        return np.ones(len(ordinals), dtype=np.float32)
    latest = float(ordinals[valid].max())
    age = np.maximum(0.0, latest - ordinals)
    w = np.exp2(-age / half_life)
    w[~valid] = 1.0
    w /= max(float(w.mean()), 1e-8)
    return w.astype(np.float32)


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, k=16, dropout=0.20):
        super().__init__()
        total = int(np.sum(field_dims))
        fields = len(field_dims)
        width = fields * k
        self.embedding = nn.Embedding(total, k)
        self.linear_embedding = nn.Embedding(total, 1)
        self.cross_w = nn.ModuleList([nn.Linear(width, 1, bias=False) for _ in range(1)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(1)])
        self.cross_out = nn.Linear(width, 1)
        self.mlp = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, x):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * w(cross) + b + cross
        linear = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        return linear + self.cross_out(cross).squeeze(-1) + self.mlp(x0).squeeze(-1)

    def accessed_row_l2(self, x):
        return self.embedding(x).pow(2).sum(dim=-1).mean()


def predict(model, x, device):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            end = min(start + 65536, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            out[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def official_metrics(is_npz, users, labels, scores):
    try:
        if is_npz:
            from data.official.evaluate import evaluate
        else:
            from harness.evaluate_provisional import evaluate
        result = evaluate(users, labels, scores)
        return {
            "gauc": float(result.get("GAUC", result.get("gauc"))),
            "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
            "primary": float(result.get("primary")),
        }
    except Exception:
        labels = np.asarray(labels, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if len(labels) == 0:
            return {"gauc": 0.0, "ndcg5": 0.0, "primary": 0.0}
        auc = 0.5
        if len(np.unique(labels)) > 1:
            pos = scores[labels > 0.5]
            neg = scores[labels <= 0.5]
            if len(pos) and len(neg):
                auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
        return {"gauc": auc, "ndcg5": 0.0, "primary": auc}


def train_model(xtr, ytr, users, dates, xva, yva, val_users, dims, is_npz, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(dims, k=16, dropout=0.20).to(device)
    embedding_params = [model.embedding.weight, model.linear_embedding.weight]
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=0.002,
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    weights = recency_weights(dates, 7.0)
    pair_pos, pair_neg = make_pairs(users, ytr, seed + 17)
    rng = np.random.default_rng(seed)
    best_gauc = -1.0
    best_state = None
    batch_size = 4096
    pair_batch = 1024

    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(xtr))
        steps = max(1, (len(order) + batch_size - 1) // batch_size)
        for step in range(steps):
            idx = order[step * batch_size:(step + 1) * batch_size]
            xb = torch.as_tensor(xtr[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (bce(logits, yb) * wb).sum() / wb.sum().clamp_min(1e-8)

            if len(pair_pos):
                chosen = rng.integers(0, len(pair_pos), size=min(pair_batch, len(pair_pos)))
                pi = pair_pos[chosen]
                ni = pair_neg[chosen]
                pair_x = np.concatenate((xtr[pi], xtr[ni]), axis=0)
                pair_logits = model(torch.as_tensor(pair_x, dtype=torch.long, device=device))
                lp, ln = pair_logits[:len(pi)], pair_logits[len(pi):]
                pw = torch.as_tensor(weights[pi], dtype=torch.float32, device=device)
                pair_loss = (torch.nn.functional.softplus(-(lp - ln)) * pw).sum() / pw.sum().clamp_min(1e-8)
            else:
                pair_loss = point_loss

            row_l2 = model.accessed_row_l2(xb)
            loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-5 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, xva, device)
        metrics = official_metrics(is_npz, val_users, yva, scores)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    xtr, ytr, utr, dates, xva, yva, uva, videos, dims, is_npz = data

    epochs = 4
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    model, device = train_model(xtr, ytr, utr, dates, xva, yva, uva, dims, is_npz, args.seed, epochs)
    scores = predict(model, xva, device)
    metrics = official_metrics(is_npz, uva, yva, scores)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(uva, videos, scores)):
            writer.writerow([i, u.item() if isinstance(u, np.generic) else u, v.item() if isinstance(v, np.generic) else v, float(s)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
