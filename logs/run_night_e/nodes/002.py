import argparse
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(tr["X"], dtype=np.int64)
    y_train = np.asarray(tr["y"], dtype=np.float32)
    x_val = np.asarray(va["X"], dtype=np.int64)
    y_val = np.asarray(va["y"], dtype=np.float32)
    val_users = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    if x_train.min() >= 0 and np.all(x_train.max(axis=0) < field_dims):
        x_train = x_train + offsets
        x_val = x_val + offsets
    val_videos = x_val[:, 1] - offsets[1]
    return x_train, y_train, x_val, y_val, val_users, val_videos, field_dims


def duration_edges(values, buckets=10):
    q = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(values, dtype=np.float64), q))


def load_csv(data_dir):
    train_rows = []
    durations = []
    with open(os.path.join(data_dir, "train.csv"), "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"] or 0.0)
            label = float(row["long_view"])
            train_rows.append((user, video, tab, duration, label))
            durations.append(duration)
    edges = duration_edges(durations)
    maps = []
    for col in range(3):
        vals = sorted({r[col] for r in train_rows})
        maps.append({v: i + 1 for i, v in enumerate(vals)})
    field_dims = np.asarray([len(maps[0]) + 1, len(maps[1]) + 1,
                             len(maps[1]) + 1, len(maps[2]) + 1,
                             len(edges) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    x_train = np.empty((len(train_rows), 5), dtype=np.int64)
    y_train = np.empty(len(train_rows), dtype=np.float32)
    for i, (user, video, tab, duration, label) in enumerate(train_rows):
        video_code = maps[1][video]
        x_train[i] = [maps[0][user], video_code, video_code,
                      maps[2][tab], np.searchsorted(edges, duration, side="right")]
        y_train[i] = label
    x_train += offsets
    val_features = []
    y_val = []
    val_users = []
    val_videos = []
    with open(os.path.join(data_dir, "val.csv"), "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user = row["user_id"]
            video = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"] or 0.0)
            video_code = maps[1].get(video, 0)
            val_features.append([maps[0].get(user, 0), video_code, video_code,
                                 maps[2].get(tab, 0),
                                 np.searchsorted(edges, duration, side="right")])
            y_val.append(float(row["long_view"]))
            val_users.append(user)
            val_videos.append(video)
    x_val = np.asarray(val_features, dtype=np.int64) + offsets
    return (x_train, y_train, x_val, np.asarray(y_val, dtype=np.float32),
            np.asarray(val_users), np.asarray(val_videos), field_dims)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
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


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.30):
        super().__init__()
        self.embedding = nn.Embedding(int(np.sum(field_dims)), embed_dim)
        input_dim = len(field_dims) * embed_dim
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + 128, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

    def forward(self, x, return_row_l2=False):
        emb = self.embedding(x)
        flat = emb.flatten(1)
        cross = flat
        for w, b in zip(self.cross_w, self.cross_b):
            cross = flat * torch.sum(cross * w, dim=1, keepdim=True) + b + cross
        hidden = self.mlp(flat)
        logits = self.output(torch.cat([cross, hidden], dim=1)).squeeze(1)
        if return_row_l2:
            return logits, emb.square().mean()
        return logits


def predict(model, x, device, batch_size=8192):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            chunks.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def official_evaluate(users, labels, scores, npz_path):
    if npz_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(1)

    npz_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_path:
        x_train, y_train, x_val, y_val, val_users, val_videos, field_dims = load_npz(args.data_dir)
    else:
        x_train, y_train, x_val, y_val, val_users, val_videos, field_dims = load_csv(args.data_dir)

    pair_pos, pair_neg = make_pairs(x_train[:, 0], y_train, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims, embed_dim=16, dropout=0.30).to(device)
    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-3},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    epochs = 20
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))
    batch_size = 4096
    rng = np.random.default_rng(args.seed)
    best_gauc = -1.0
    best_state = None
    stale = 0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        steps = (len(order) + batch_size - 1) // batch_size
        for step in range(steps):
            idx = order[step * batch_size:(step + 1) * batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            logits, row_l2 = model(xb, return_row_l2=True)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)
            if len(pair_pos):
                pstart = (step * batch_size) % len(pair_pos)
                pidx = pair_order[pstart:min(pstart + batch_size, len(pair_pos))]
                if len(pidx) < batch_size:
                    pidx = np.concatenate([pidx, pair_order[:batch_size - len(pidx)]])
                pos_x = torch.as_tensor(x_train[pair_pos[pidx]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(x_train[pair_neg[pidx]], dtype=torch.long, device=device)
                pos_score, pos_l2 = model(pos_x, return_row_l2=True)
                neg_score, neg_l2 = model(neg_x, return_row_l2=True)
                pair_loss = F.softplus(-(pos_score - neg_score)).mean()
                row_l2 = (row_l2 + pos_l2 + neg_l2) / 3.0
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-3 * row_l2
            else:
                loss = point_loss + 1e-3 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scores = predict(model, x_val, device)
        metrics = official_evaluate(val_users, y_val, scores, npz_path)
        gauc = float(metrics["GAUC"])
        if gauc > best_gauc:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 5:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, x_val, device)
    metrics = official_evaluate(val_users, y_val, scores, npz_path)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(val_users, val_videos, scores)):
            writer.writerow([i, user, video, float(score)])
    result = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
