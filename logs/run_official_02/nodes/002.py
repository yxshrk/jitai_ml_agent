import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    X_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    train_user = np.asarray(tr["user"])
    X_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    val_user = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))
    val_video = X_val[:, 1] - offsets[1]
    return X_train, y_train, train_user, X_val, y_val, val_user, val_video, field_dims


def duration_edges(values):
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, np.linspace(0.1, 0.9, 9))
    return np.maximum.accumulate(edges)


def load_csv(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    durations = []
    with open(train_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_rows.append(row)
            durations.append(float(row["duration_ms"]))
    edges = duration_edges(durations)
    field_names = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    for name in field_names:
        mapping = {"__UNK__": 0}
        for row in train_rows:
            if name == "author_id" and name not in row:
                value = "__MISSING_AUTHOR__"
            else:
                value = row.get(name, "__MISSING__")
            if value not in mapping:
                mapping[value] = len(mapping)
        maps.append(mapping)
    field_dims = np.asarray([len(m) for m in maps] + [10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1], dtype=np.int64)))

    def encode(rows, training):
        X = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, row in enumerate(rows):
            users.append(row["user_id"])
            videos.append(row["video_id"])
            for j, name in enumerate(field_names):
                if name == "author_id" and name not in row:
                    value = "__MISSING_AUTHOR__"
                else:
                    value = row.get(name, "__MISSING__")
                X[i, j] = maps[j].get(value, 0) + offsets[j]
            bucket = int(np.searchsorted(edges, float(row["duration_ms"]), side="right"))
            X[i, 4] = bucket + offsets[4]
            y[i] = float(row["long_view"])
        return X, y, np.asarray(users), np.asarray(videos)

    X_train, y_train, train_user, _ = encode(train_rows, True)
    val_rows = []
    with open(val_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_rows.append(row)
    X_val, y_val, val_user, val_video = encode(val_rows, False)
    return X_train, y_train, train_user, X_val, y_val, val_user, val_video, field_dims


def make_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)]))
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if pos.size and neg.size:
            positives.append(pos)
            negatives.append(neg[rng.integers(0, neg.size, size=pos.size)])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class RegularizedDCN(nn.Module):
    def __init__(self, total_features, fields=5, dim=16, hidden=128, dropout=0.3):
        super().__init__()
        width = fields * dim
        self.embedding = nn.Embedding(total_features, dim)
        self.linear_embedding = nn.Embedding(total_features, 1)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.cross_out = nn.Linear(width, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.zeros_(self.cross_out.bias)

    def forward(self, x, return_row_l2=False):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(cross * w, dim=1, keepdim=True)
            cross = cross + x0 * scale + b
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        logits = linear + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1) + self.bias
        if return_row_l2:
            row_l2 = emb.square().sum(dim=2).mean()
            return logits, row_l2
        return logits


def predict(model, X, device, batch_size=32768):
    model.eval()
    output = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(X[start:end]).to(device)
            output[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return output


def official_metrics(users, labels, scores, npz_path):
    if npz_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    use_npz = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if use_npz:
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)
    X_train, y_train, train_user, X_val, y_val, val_user, val_video, field_dims = data

    pair_pos, pair_neg = make_pairs(train_user, y_train, args.seed + 17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(int(field_dims.sum()), dropout=0.3).to(device)
    embedding_params = [model.embedding.weight, model.linear_embedding.weight]
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-3},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epochs = 12
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 16384
    pair_batch = 4096
    rng = np.random.default_rng(args.seed + 29)
    best_gauc = -float("inf")
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        permutation = rng.permutation(len(X_train))
        for start in range(0, len(permutation), batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            point_logits, point_l2 = model(xb, return_row_l2=True)
            point_loss = F.binary_cross_entropy_with_logits(point_logits, yb)
            if pair_pos.size:
                chosen = rng.integers(0, pair_pos.size, size=min(pair_batch, pair_pos.size))
                pi = pair_pos[chosen]
                ni = pair_neg[chosen]
                xp = torch.from_numpy(X_train[pi]).to(device)
                xn = torch.from_numpy(X_train[ni]).to(device)
                pos_logits, pos_l2 = model(xp, return_row_l2=True)
                neg_logits, neg_l2 = model(xn, return_row_l2=True)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                row_l2 = (point_l2 + pos_l2 + neg_l2) / 3.0
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-3 * row_l2
            else:
                loss = point_loss + 1e-3 * point_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_scores = predict(model, X_val, device)
        metrics = official_metrics(val_user, y_val, val_scores, use_npz)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, X_val, device)
    metrics = official_metrics(val_user, y_val, scores, use_npz)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(val_user, val_video, scores)):
            writer.writerow([i, user_id, video_id, float(score)])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
