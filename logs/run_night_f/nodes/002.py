import argparse
import csv
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        self.field_dims = [int(v) for v in field_dims]
        self.num_fields = len(self.field_dims)
        self.embed_dim = embed_dim
        total = int(sum(self.field_dims))
        width = self.num_fields * embed_dim

        self.embedding = nn.Embedding(total, embed_dim, sparse=True)
        self.linear_embedding = nn.Embedding(total, 1, sparse=True)
        self.cross_w1 = nn.Parameter(torch.empty(width))
        self.cross_b1 = nn.Parameter(torch.zeros(width))
        self.cross_w2 = nn.Parameter(torch.empty(width))
        self.cross_b2 = nn.Parameter(torch.zeros(width))
        self.cross_out = nn.Linear(width, 1)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        nn.init.normal_(self.cross_w1, std=0.01)
        nn.init.normal_(self.cross_w2, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.cross_out.weight)
        nn.init.zeros_(self.cross_out.bias)

    def forward(self, x, return_rows=False):
        emb = self.embedding(x)
        x0 = emb.reshape(x.shape[0], -1)
        x1 = x0 * torch.sum(x0 * self.cross_w1, dim=1, keepdim=True) + self.cross_b1 + x0
        x2 = x0 * torch.sum(x1 * self.cross_w2, dim=1, keepdim=True) + self.cross_b2 + x1
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        score = linear + self.cross_out(x2).squeeze(1) + self.mlp(x0).squeeze(1) + self.bias
        if return_rows:
            row_penalty = emb.square().sum(dim=2).mean() + self.linear_embedding(x).square().mean()
            return score, row_penalty
        return score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode(values, mapping):
    return np.asarray([mapping.get(v, 0) for v in values], dtype=np.int64)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    train_cols = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")}
    with open(train_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            train_cols["user_id"].append(user)
            train_cols["video_id"].append(video)
            train_cols["author_id"].append(row["author_id"] if has_author else video)
            train_cols["tab"].append(row["tab"])
            train_cols["duration_ms"].append(float(row["duration_ms"] or 0.0))
            train_cols["long_view"].append(float(row["long_view"]))

    val_cols = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms", "long_view")}
    with open(val_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            val_cols["user_id"].append(user)
            val_cols["video_id"].append(video)
            val_cols["author_id"].append(row["author_id"] if has_author else video)
            val_cols["tab"].append(row["tab"])
            val_cols["duration_ms"].append(float(row["duration_ms"] or 0.0))
            val_cols["long_view"].append(float(row["long_view"]))

    maps = [
        make_mapping(train_cols["user_id"]),
        make_mapping(train_cols["video_id"]),
        make_mapping(train_cols["author_id"]),
        make_mapping(train_cols["tab"]),
    ]
    train_duration = np.asarray(train_cols["duration_ms"], dtype=np.float64)
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(edges, train_duration, side="right").astype(np.int64) + 1
    val_duration = np.asarray(val_cols["duration_ms"], dtype=np.float64)
    val_bucket = np.searchsorted(edges, val_duration, side="right").astype(np.int64) + 1

    train_fields = [
        encode(train_cols["user_id"], maps[0]),
        encode(train_cols["video_id"], maps[1]),
        encode(train_cols["author_id"], maps[2]),
        encode(train_cols["tab"], maps[3]),
        train_bucket,
    ]
    val_fields = [
        encode(val_cols["user_id"], maps[0]),
        encode(val_cols["video_id"], maps[1]),
        encode(val_cols["author_id"], maps[2]),
        encode(val_cols["tab"], maps[3]),
        val_bucket,
    ]
    field_dims = np.asarray([len(m) + 1 for m in maps] + [len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    train_x = np.stack(train_fields, axis=1) + offsets
    val_x = np.stack(val_fields, axis=1) + offsets

    return {
        "train_x": train_x.astype(np.int64),
        "train_y": np.asarray(train_cols["long_view"], dtype=np.float32),
        "train_user": np.asarray(train_cols["user_id"], dtype=object),
        "val_x": val_x.astype(np.int64),
        "val_y": np.asarray(val_cols["long_view"], dtype=np.float32),
        "val_user": np.asarray(val_cols["user_id"], dtype=object),
        "val_video": np.asarray(val_cols["video_id"], dtype=object),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=False) as tr:
            train_x = np.asarray(tr["X"], dtype=np.int64)
            train_y = np.asarray(tr["y"], dtype=np.float32)
            train_user = np.asarray(tr["user"])
            field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
        with np.load(val_npz, allow_pickle=False) as va:
            val_x = np.asarray(va["X"], dtype=np.int64)
            val_y = np.asarray(va["y"], dtype=np.float32)
            val_user = np.asarray(va["user"])
        offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
        val_video = val_x[:, 1] - offsets[1]
        return {
            "train_x": train_x,
            "train_y": train_y,
            "train_user": train_user,
            "val_x": val_x,
            "val_y": val_y,
            "val_user": val_user,
            "val_video": val_video,
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
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
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def metric_values(data, scores):
    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    raw = evaluate(data["val_user"], data["val_y"], scores)
    return {
        "gauc": float(raw.get("GAUC", raw.get("gauc"))),
        "ndcg5": float(raw.get("nDCG@5", raw.get("ndcg5"))),
        "primary": float(raw.get("primary")),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    data = load_data(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(data["field_dims"], embed_dim=16, hidden_dim=128, dropout=0.30).to(device)
    sparse_params = list(model.embedding.parameters()) + list(model.linear_embedding.parameters())
    sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=0.003)
    sparse_ids = {id(p) for p in sparse_params}
    dense_params = [p for p in model.parameters() if id(p) not in sparse_ids]
    dense_optimizer = torch.optim.AdamW(dense_params, lr=0.003, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dense_optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4, min_lr=3e-5
    )

    pair_pos, pair_neg = make_pairs(data["train_user"], data["train_y"], args.seed)
    rng = np.random.default_rng(args.seed)
    batch_size = 8192
    epochs = 10
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    best_gauc = -1.0
    best_epoch = -1
    checkpoint = os.path.join(args.out_dir, ".best_model.pt")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(data["train_x"]))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)

        for step, start in enumerate(range(0, len(point_order), batch_size)):
            ids = point_order[start:start + batch_size]
            xb = torch.as_tensor(data["train_x"][ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["train_y"][ids], dtype=torch.float32, device=device)
            logits, row_l2 = model(xb, return_rows=True)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_order):
                pstart = (step * batch_size) % len(pair_order)
                pids = pair_order[pstart:pstart + batch_size]
                if len(pids) < batch_size:
                    pids = np.concatenate((pids, pair_order[:batch_size - len(pids)]))
                pos_x = torch.as_tensor(data["train_x"][pair_pos[pids]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(data["train_x"][pair_neg[pids]], dtype=torch.long, device=device)
                pos_score, pos_l2 = model(pos_x, return_rows=True)
                neg_score, neg_l2 = model(neg_x, return_rows=True)
                pair_loss = F.softplus(-(pos_score - neg_score)).mean()
                row_l2 = (row_l2 + pos_l2 + neg_l2) / 3.0
                loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-3 * row_l2
            else:
                loss = point_loss + 1e-3 * row_l2

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense_params, 5.0)
            sparse_optimizer.step()
            dense_optimizer.step()

        val_scores = predict(model, data["val_x"], device)
        metrics = metric_values(data, val_scores)
        current_gauc = metrics["gauc"]
        scheduler.step(current_gauc)
        if current_gauc > best_gauc + 1e-6:
            best_gauc = current_gauc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            no_improve += 1
        if no_improve >= 4 and epoch >= 4:
            break

    if best_epoch >= 0:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
    final_scores = predict(model, data["val_x"], device)
    final_metrics = metric_values(data, final_scores)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["val_user"], data["val_video"], final_scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user,
                             video.item() if isinstance(video, np.generic) else video, float(score)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, separators=(",", ":"))
    if os.path.exists(checkpoint):
        os.remove(checkpoint)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            main()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
