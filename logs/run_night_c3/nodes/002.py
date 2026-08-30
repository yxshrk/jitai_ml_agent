import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scalar_value(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=True) as z:
        tr = {k: scalar_value(z[k]) for k in z.files}
    with np.load(val_path, allow_pickle=True) as z:
        va = {k: scalar_value(z[k]) for k in z.files}

    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32).reshape(-1)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32).reshape(-1)
    train_users = np.asarray(tr["user"]).reshape(-1)
    val_users = np.asarray(va["user"]).reshape(-1)
    field_dims = np.asarray(tr.get("field_dims", va.get("field_dims")), dtype=np.int64).reshape(-1)

    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    if x_train.shape[1] != 5:
        raise RuntimeError("expected exactly five offset-encoded fields")
    val_videos = x_val[:, 1] - offsets[1]
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__missing_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def build_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_map = build_mapping([r["user_id"] for r in train_rows])
    video_map = build_mapping([r["video_id"] for r in train_rows])
    author_map = build_mapping([r["author_id"] for r in train_rows])
    tab_map = build_mapping([r["tab"] for r in train_rows])

    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_dims = np.asarray([
        len(user_map) + 2,
        len(video_map) + 2,
        len(author_map) + 2,
        len(tab_map) + 2,
        11,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user_id"], 0)
            x[i, 1] = video_map.get(r["video_id"], 0)
            x[i, 2] = author_map.get(r["author_id"], 0)
            x[i, 3] = tab_map.get(r["tab"], 0)
            x[i, 4] = int(np.searchsorted(quantiles, r["duration_ms"], side="right")) + 1
            x[i] += offsets
            y[i] = r["long_view"]
            users.append(r["user_id"])
            videos.append(r["video_id"])
        return x, y, np.asarray(users, dtype=object), np.asarray(videos, dtype=object)

    x_train, y_train, train_users, _ = encode(train_rows)
    x_val, y_val, val_users, val_videos = encode(val_rows)
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, False


class RegularizedDCN(nn.Module):
    def __init__(self, total_features, num_fields=5, embedding_dim=16, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(total_features, embedding_dim)
        input_dim = num_fields * embedding_dim
        self.cross1 = nn.Linear(input_dim, input_dim)
        self.cross2 = nn.Linear(input_dim, input_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.deep1 = nn.Linear(input_dim, 128)
        self.deep2 = nn.Linear(128, 64)
        self.deep_dropout1 = nn.Dropout(dropout)
        self.deep_dropout2 = nn.Dropout(dropout)
        self.cross_out = nn.Linear(input_dim, 1, bias=False)
        self.deep_out = nn.Linear(64, 1, bias=False)
        self.linear = nn.Embedding(total_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        embedded = self.embedding(x)
        x0 = embedded.flatten(1)
        cross = x0 + x0 * self.cross1(x0)
        cross = cross + x0 * self.cross2(cross)
        deep = self.input_dropout(x0)
        deep = self.deep_dropout1(F.relu(self.deep1(deep)))
        deep = self.deep_dropout2(F.relu(self.deep2(deep)))
        first_order = self.linear(x).sum(dim=1).squeeze(-1)
        return first_order + self.cross_out(cross).squeeze(-1) + self.deep_out(deep).squeeze(-1) + self.bias


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        pos = indices[labels[indices] > 0.5]
        neg = indices[labels[indices] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        positives.append(rng.choice(pos, size=count, replace=len(pos) < count))
        negatives.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def evaluate_scores(fast_path, users, labels, scores):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def train_model(x_train, y_train, train_users, x_val, y_val, val_users, field_dims, fast_path, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(int(np.sum(field_dims)), dropout=0.30).to(device)

    embedding_params = list(model.embedding.parameters()) + list(model.linear.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-3},
    ], lr=1e-3)

    max_epochs = 12
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))

    pair_pos, pair_neg = make_pairs(train_users, y_train, seed + 17)
    rng = np.random.default_rng(seed + 101)
    batch_size = 4096
    row_l2_weight = 1e-4
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(x_train))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)

        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            batch_indices = order[start:start + batch_size]
            xb = torch.from_numpy(x_train[batch_indices]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y_train[batch_indices]).to(device=device, dtype=torch.float32)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            touched = [xb.reshape(-1)]
            if len(pair_order):
                pair_start = (batch_number * batch_size) % len(pair_order)
                if pair_start + len(batch_indices) <= len(pair_order):
                    chosen = pair_order[pair_start:pair_start + len(batch_indices)]
                else:
                    needed = len(batch_indices)
                    chosen = np.concatenate((pair_order[pair_start:], pair_order[:needed - (len(pair_order) - pair_start)]))
                pos_x = torch.from_numpy(x_train[pair_pos[chosen]]).to(device=device, dtype=torch.long)
                neg_x = torch.from_numpy(x_train[pair_neg[chosen]]).to(device=device, dtype=torch.long)
                pair_loss = F.softplus(-(model(pos_x) - model(neg_x))).mean()
                touched.extend([pos_x.reshape(-1), neg_x.reshape(-1)])
            else:
                pair_loss = point_loss.new_zeros(())

            unique_rows = torch.unique(torch.cat(touched))
            row_l2 = model.embedding(unique_rows).pow(2).sum(dim=1).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss + row_l2_weight * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, x_val, device)
        metrics = evaluate_scores(fast_path, val_users, y_val, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        for group in optimizer.param_groups:
            group["lr"] *= 0.5
        if stale >= 4:
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
            writer.writerow([i, user, video, format(float(score), ".10g")])

    payload = {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))


def main():
    args = parse_args()
    set_seed(args.seed)
    npz_fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_fast:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, fast_path = data
    model, device = train_model(x_train, y_train, train_users, x_val, y_val, val_users, field_dims, fast_path, args.seed)
    scores = predict(model, x_val, device)
    metrics = evaluate_scores(fast_path, val_users, y_val, scores)
    write_outputs(args.out_dir, val_users, val_videos, scores, metrics)


if __name__ == "__main__":
    main()
