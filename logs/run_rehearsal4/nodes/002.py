import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


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


def load_npz_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    with np.load(train_path, allow_pickle=False) as z:
        train_x = np.asarray(z["X"], dtype=np.int64)
        train_y = np.asarray(z["y"], dtype=np.float32)
        train_user = np.asarray(z["user"])
        field_dims = np.asarray(z["field_dims"], dtype=np.int64)
    with np.load(val_path, allow_pickle=False) as z:
        val_x = np.asarray(z["X"], dtype=np.int64)
        val_y = np.asarray(z["y"], dtype=np.float32)
        val_user = np.asarray(z["user"])
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    video_codes = val_x[:, 1] - offsets[1]
    return train_x, train_y, train_user, val_x, val_y, val_user, video_codes, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row.get("tab", ""),
                "duration_ms": float(row.get("duration_ms", 0.0) or 0.0),
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def make_vocab(values):
    vocab = {}
    for value in values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)

    user_vocab = make_vocab(row["user_id"] for row in train_rows)
    video_vocab = make_vocab(row["video_id"] for row in train_rows)
    author_vocab = make_vocab(row["author_id"] for row in train_rows)
    tab_vocab = make_vocab(row["tab"] for row in train_rows)

    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    if durations.size:
        edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
        edges = np.maximum.accumulate(edges)
    else:
        edges = np.zeros(9, dtype=np.float64)

    field_dims = np.asarray([
        len(user_vocab) + 1,
        len(video_vocab) + 1,
        len(author_vocab) + 1,
        len(tab_vocab) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, row in enumerate(rows):
            local = [
                user_vocab.get(row["user_id"], len(user_vocab)),
                video_vocab.get(row["video_id"], len(video_vocab)),
                author_vocab.get(row["author_id"], len(author_vocab)),
                tab_vocab.get(row["tab"], len(tab_vocab)),
                int(np.searchsorted(edges, row["duration_ms"], side="right")),
            ]
            x[i] = np.asarray(local, dtype=np.int64) + offsets
            y[i] = row["long_view"]
            users.append(row["user_id"])
            videos.append(row["video_id"])
        return x, y, np.asarray(users), np.asarray(videos)

    train_x, train_y, train_user, _ = encode(train_rows)
    val_x, val_y, val_user, val_video = encode(val_rows)
    return train_x, train_y, train_user, val_x, val_y, val_user, val_video, field_dims, False


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        input_dim = int(len(field_dims) * embed_dim)
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.linear_embedding = nn.Embedding(total_dim, 1)
        self.cross_w = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim)) for _ in range(2)
        ])
        self.cross_b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(2)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + hidden_dim, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        embedded = self.embedding(x).flatten(1)
        crossed = embedded
        for weight, bias in zip(self.cross_w, self.cross_b):
            scalar = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = embedded * scalar + bias + crossed
        deep = self.mlp(embedded)
        interaction = self.output(torch.cat((crossed, deep), dim=1)).squeeze(1)
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return interaction + linear


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    if len(order) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group] >= 0.5]
        neg = group[labels[group] < 0.5]
        if pos.size and neg.size:
            chosen = neg[rng.integers(0, neg.size, size=pos.size)]
            positives.append(pos)
            negatives.append(chosen)
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            outputs.append(torch.sigmoid(model(xb)).cpu().numpy())
    if not outputs:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(outputs).astype(np.float32, copy=False)


def train_model(train_x, train_y, train_users, val_x, val_y, val_users, field_dims, seed, evaluator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    bce = nn.BCEWithLogitsLoss()
    pair_pos, pair_neg = build_pairs(train_users, train_y, seed)
    rng = np.random.default_rng(seed)
    batch_size = 4096
    pair_batch_size = 4096
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    for epoch in range(8):
        model.train()
        impression_order = rng.permutation(len(train_x))
        if pair_pos.size:
            pair_order = rng.permutation(pair_pos.size)
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0

        for start in range(0, len(impression_order), batch_size):
            idx = impression_order[start:start + batch_size]
            xb = torch.as_tensor(train_x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            point_loss = bce(model(xb), yb)

            if pair_order.size:
                if pair_cursor + pair_batch_size > pair_order.size:
                    pair_order = rng.permutation(pair_pos.size)
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + pair_batch_size]
                pair_cursor += pair_batch_size
                pos_x = torch.as_tensor(train_x[pair_pos[selected]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(train_x[pair_neg[selected]], dtype=torch.long, device=device)
                pair_loss = torch.nn.functional.softplus(-(model(pos_x) - model(neg_x))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()
        val_scores = predict(model, val_x, device)
        metric = evaluator(val_users, val_y, val_scores)
        gauc = float(metric.get("GAUC", metric.get("gauc")))
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

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
    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))

    if fast_path:
        data = load_npz_data(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        data = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate

    train_x, train_y, train_users, val_x, val_y, val_users, val_videos, field_dims, _ = data
    model, device = train_model(
        train_x, train_y, train_users, val_x, val_y, val_users,
        field_dims, args.seed, evaluate
    )
    scores = predict(model, val_x, device)
    result = evaluate(val_users, val_y, scores)
    metrics = {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(val_users, val_videos, scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user,
                             video.item() if isinstance(video, np.generic) else video,
                             float(score)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    main()
