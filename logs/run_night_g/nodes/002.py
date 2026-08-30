import argparse
import csv
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.30, cross_layers=2):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.input_dim = self.num_fields * embed_dim
        self.field_dropout = dropout
        total_dim = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.linear = nn.Embedding(total_dim, 1)
        self.cross_w = nn.ModuleList([nn.Linear(self.input_dim, 1, bias=False) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(self.input_dim)) for _ in range(cross_layers)])
        self.cross_out = nn.Linear(self.input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        emb = self.embedding(x)
        if self.training and self.field_dropout > 0.0:
            keep = 1.0 - self.field_dropout
            mask = torch.empty((x.shape[0], self.num_fields, 1), device=emb.device).bernoulli_(keep)
            emb = emb * mask / keep
        x0 = emb.reshape(x.shape[0], -1)
        cross = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            cross = x0 * weight(cross) + bias + cross
        linear_logit = self.linear(x).sum(dim=1).squeeze(1)
        return linear_logit + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1) + self.bias


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def scalar(v):
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return v.item()
    return v


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as tr:
        train = {k: tr[k] for k in tr.files}
    with np.load(val_path, allow_pickle=False) as va:
        val = {k: va[k] for k in va.files}
    x_train = np.asarray(train["X"], dtype=np.int64)
    y_train = np.asarray(train["y"], dtype=np.float32)
    user_train = np.asarray(train["user"])
    x_val = np.asarray(val["X"], dtype=np.int64)
    y_val = np.asarray(val["y"], dtype=np.float32)
    user_val = np.asarray(val["user"])
    if "field_dims" in train:
        field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    else:
        field_dims = np.asarray(val["field_dims"], dtype=np.int64)
    video_val = x_val[:, 1].copy()
    return x_train, y_train, user_train, x_val, y_val, user_val, video_val, field_dims


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"] or 0.0),
                "label": float(row["long_view"]),
            }
            rows.append(record)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))

    def duration_bucket(value):
        return str(int(np.searchsorted(quantiles, value, side="right")))

    user_values = [r["user"] for r in train_rows]
    video_values = [r["video"] for r in train_rows]
    tab_values = [r["tab"] for r in train_rows]
    dur_values = [duration_bucket(r["duration"]) for r in train_rows]
    author_values = ["__unknown_author__"] * len(train_rows)
    columns = [user_values, video_values, author_values, tab_values, dur_values]
    maps = []
    dims = []
    for values in columns:
        mapping = {value: i + 1 for i, value in enumerate(sorted(set(values)))}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            values = [row["user"], row["video"], "__unknown_author__", row["tab"], duration_bucket(row["duration"])]
            for j, value in enumerate(values):
                x[i, j] = maps[j].get(value, 0) + offsets[j]
        y = np.asarray([r["label"] for r in rows], dtype=np.float32)
        users = np.asarray([r["user"] for r in rows])
        videos = np.asarray([r["video"] for r in rows])
        return x, y, users, videos

    x_train, y_train, user_train, _ = encode(train_rows)
    x_val, y_val, user_val, video_val = encode(val_rows)
    return x_train, y_train, user_train, x_val, y_val, user_val, video_val, np.asarray(dims, dtype=np.int64)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_metrics(users, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    values = evaluate(users, labels, scores)
    return {
        "gauc": float(values["GAUC"]),
        "ndcg5": float(values["nDCG@5"]),
        "primary": float(values["primary"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast_path:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, user_train, x_val, y_val, user_val, video_val, field_dims = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims, embed_dim=16, hidden_dim=128, dropout=0.30, cross_layers=2).to(device)
    embedding_params = list(model.embedding.parameters()) + list(model.linear.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    max_epochs = 20
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    batch_size = 8192
    pair_pos, pair_neg = make_pairs(user_train, y_train, args.seed + 17)
    rng = np.random.default_rng(args.seed + 31)
    best_gauc = -1.0
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        impression_order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        num_steps = max((len(impression_order) + batch_size - 1) // batch_size, (len(pair_order) + batch_size - 1) // batch_size if len(pair_order) else 0)
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            start = step * batch_size
            if start < len(impression_order):
                idx = impression_order[start:min(start + batch_size, len(impression_order))]
                xb = torch.from_numpy(x_train[idx]).to(device)
                yb = torch.from_numpy(y_train[idx]).to(device)
                logits = model(xb)
                losses.append(0.5 * F.binary_cross_entropy_with_logits(logits, yb))
                accessed = torch.unique(xb.reshape(-1))
                row_norm = model.embedding(accessed).pow(2).sum(dim=1).mean()
                losses.append(1e-4 * row_norm)
            if start < len(pair_order):
                pidx = pair_order[start:min(start + batch_size, len(pair_order))]
                pos_x = torch.from_numpy(x_train[pair_pos[pidx]]).to(device)
                neg_x = torch.from_numpy(x_train[pair_neg[pidx]]).to(device)
                pos_score = model(pos_x)
                neg_score = model(neg_x)
                losses.append(0.5 * F.softplus(-(pos_score - neg_score)).mean())
                pair_rows = torch.unique(torch.cat((pos_x.reshape(-1), neg_x.reshape(-1))))
                losses.append(1e-4 * model.embedding(pair_rows).pow(2).sum(dim=1).mean())
            loss = torch.stack(losses).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, x_val, device)
        metrics = official_metrics(user_val, y_val, val_scores, fast_path)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 4:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, device)
    metrics = official_metrics(user_val, y_val, scores, fast_path)

    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(user_val, video_val, scores)):
            writer.writerow([i, scalar(user), scalar(video), "{:.9g}".format(float(score))])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
