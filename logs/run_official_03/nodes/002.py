import argparse
import csv
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    users_train = np.asarray(tr["user"])
    users_val = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    videos_val = x_val[:, 1] - offsets[1]
    return x_train, y_train, users_train, x_val, y_val, users_val, videos_val, field_dims, True


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))
    durations = np.asarray([safe_int(r.get("duration_ms"), 0) for r in train_rows], dtype=np.float64)
    if durations.size:
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    def raw_fields(row):
        duration = safe_int(row.get("duration_ms"), 0)
        bucket = int(np.searchsorted(edges, duration, side="right"))
        return [
            str(row.get("user_id", "")),
            str(row.get("video_id", "")),
            str(row.get("author_id", "0")),
            str(row.get("tab", "")),
            str(bucket),
        ]

    mappings = []
    for field in range(5):
        values = sorted({raw_fields(r)[field] for r in train_rows})
        mappings.append({value: i + 1 for i, value in enumerate(values)})
    field_dims = np.asarray([len(m) + 1 for m in mappings], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            values = raw_fields(row)
            for j in range(5):
                x[i, j] = mappings[j].get(values[j], 0) + offsets[j]
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([float(r.get("long_view", 0.0)) for r in train_rows], dtype=np.float32)
    y_val = np.asarray([float(r.get("long_view", 0.0)) for r in val_rows], dtype=np.float32)
    users_train = np.asarray([r.get("user_id", "") for r in train_rows])
    users_val = np.asarray([r.get("user_id", "") for r in val_rows])
    videos_val = np.asarray([r.get("video_id", "") for r in val_rows])
    return x_train, y_train, users_train, x_val, y_val, users_val, videos_val, field_dims, False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    start = 0
    n = len(order)
    while start < n:
        end = start + 1
        while end < n and sorted_users[end] == sorted_users[start]:
            end += 1
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
        start = end
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.3):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_out = nn.Linear(width, 1)
        self.deep_out = nn.Linear(hidden_dim // 2, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        embedded = self.embedding(x)
        x0 = embedded.flatten(1)
        crossed = x0
        for w, b in zip(self.cross_w, self.cross_b):
            crossed = x0 * torch.sum(crossed * w, dim=1, keepdim=True) + b + crossed
        deep = self.mlp(x0)
        score = self.cross_out(crossed) + self.deep_out(deep) + self.bias
        return score.squeeze(1)


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_metrics(users, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def train_model(x_train, y_train, users_train, x_val, y_val, users_val, field_dims, seed, epochs, fast_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims, embed_dim=16, hidden_dim=128, dropout=0.3).to(device)
    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=8e-4,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    pair_pos, pair_neg = make_pairs(users_train, y_train, seed)
    rng = np.random.default_rng(seed)
    batch_size = 8192 if device.type == "cuda" else 4096
    eval_batch = 32768
    best_state = None
    best_gauc = -1.0
    stale = 0
    patience = 5

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(x_train))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(point_order), batch_size):
            idx = point_order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            pair_count = len(idx)
            if len(pair_order):
                if pair_cursor + pair_count > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + pair_count]
                pair_cursor += len(selected)
                if len(selected):
                    pidx = pair_pos[selected]
                    nidx = pair_neg[selected]
                    xp = torch.as_tensor(x_train[pidx], dtype=torch.long, device=device)
                    xn = torch.as_tensor(x_train[nidx], dtype=torch.long, device=device)
                    pos_logits = model(xp)
                    neg_logits = model(xn)
                    pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                    accessed = torch.unique(torch.cat((xb.flatten(), xp.flatten(), xn.flatten())))
                else:
                    pair_loss = point_loss.new_zeros(())
                    accessed = torch.unique(xb.flatten())
            else:
                pair_loss = point_loss.new_zeros(())
                accessed = torch.unique(xb.flatten())

            row_penalty = model.embedding.weight.index_select(0, accessed).pow(2).sum(dim=1).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-3 * row_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, x_val, device, eval_batch)
        metrics = official_metrics(users_val, y_val, val_scores, fast_path)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def write_outputs(out_dir, users, videos, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, user, video, float(score)])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast_path:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, users_train, x_val, y_val, users_val, videos_val, field_dims, fast_path = data

    epochs = 20
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        try:
            epochs = min(epochs, max(1, int(smoke)))
        except ValueError:
            pass

    model, device = train_model(
        x_train, y_train, users_train, x_val, y_val, users_val,
        field_dims, args.seed, epochs, fast_path
    )
    scores = predict(model, x_val, device, 32768)
    metrics = official_metrics(users_val, y_val, scores, fast_path)
    write_outputs(args.out_dir, users_val, videos_val, scores, metrics)


if __name__ == "__main__":
    main()
