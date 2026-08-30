import argparse
import csv
import json
import os
import random
import warnings
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        total_dim = int(np.sum(field_dims))
        flat_dim = self.num_fields * embed_dim
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.linear = nn.Embedding(total_dim, 1)
        self.cross_weight = nn.Parameter(torch.empty(flat_dim))
        self.cross_bias = nn.Parameter(torch.zeros(flat_dim))
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(flat_dim + 64, 1)
        self.embedding_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        emb = self.embedding(x)
        x0 = self.embedding_dropout(emb).reshape(x.shape[0], -1)
        cross_scale = torch.sum(x0 * self.cross_weight, dim=1, keepdim=True)
        cross = x0 * cross_scale + self.cross_bias + x0
        deep = self.mlp(x0)
        first_order = self.linear(x).sum(dim=1).squeeze(1)
        return self.output(torch.cat([cross, deep], dim=1)).squeeze(1) + first_order


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    train_data = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val_data = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    x_train = np.asarray(train_data["X"], dtype=np.int64)
    y_train = np.asarray(train_data["y"], dtype=np.float32)
    train_users = np.asarray(train_data["user"])
    x_val = np.asarray(val_data["X"], dtype=np.int64)
    y_val = np.asarray(val_data["y"], dtype=np.float32)
    val_users = np.asarray(val_data["user"])
    field_dims = np.asarray(train_data["field_dims"], dtype=np.int64).reshape(-1)
    video_offset = int(field_dims[0])
    val_videos = x_val[:, 1].astype(np.int64) - video_offset
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            duration = int(float(row.get("duration_ms", "0") or 0))
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", row["video_id"]),
                "tab": row.get("tab", "0"),
                "duration": duration,
            }
            if training:
                item["label"] = float(row["long_view"])
            else:
                item["label"] = float(row["long_view"])
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    quantiles = np.unique(quantiles)
    fields = ["user", "video", "author", "tab"]
    mappings = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        mappings.append({value: i + 1 for i, value in enumerate(values)})
    field_dims = np.asarray([len(m) + 1 for m in mappings] + [len(quantiles) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        y = np.zeros(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        videos = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = mappings[j].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right")) + 1
            x[i, 4] = bucket + offsets[4]
            y[i] = row["label"]
            users[i] = row["user"]
            videos[i] = row["video"]
        return x, y, users, videos

    x_train, y_train, train_users, _ = encode(train_rows)
    x_val, y_val, val_users, val_videos = encode(val_rows)
    return x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, False


def build_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if pos.size == 0 or neg.size == 0:
            continue
        count = min(max(pos.size, neg.size), 64)
        pos_parts.append(rng.choice(pos, size=count, replace=pos.size < count))
        neg_parts.append(rng.choice(neg, size=count, replace=neg.size < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def predict(model, x, device, batch_size=32768):
    model.eval()
    outputs = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:end]).to(device)
            outputs[start:end] = model(xb).detach().cpu().numpy()
    return outputs


def metric_values(result):
    gauc = result.get("GAUC", result.get("gauc"))
    ndcg = result.get("nDCG@5", result.get("ndcg5", result.get("NDCG@5")))
    primary = result.get("primary", (float(gauc) + float(ndcg)) / 2.0)
    return float(gauc), float(ndcg), float(primary)


def run_evaluator(use_npz, users, labels, scores):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate(users, labels, scores)


def train_member(x_train, y_train, train_users, x_val, y_val, val_users, field_dims, use_npz, seed, epochs, device):
    set_seed(seed)
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    pos_idx, neg_idx = build_pairs(train_users, y_train, seed + 991)
    rng = np.random.default_rng(seed + 17)
    batch_size = 8192
    pair_batch_size = 4096
    best_gauc = -np.inf
    best_scores = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(x_train))
        pair_order = rng.permutation(len(pos_idx)) if len(pos_idx) else np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, len(point_order), batch_size):
            point_ids = point_order[start:start + batch_size]
            xb_np = x_train[point_ids]
            yb_np = y_train[point_ids]
            if len(pair_order):
                if pair_cursor + pair_batch_size > len(pair_order):
                    pair_order = rng.permutation(len(pos_idx))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + pair_batch_size]
                pair_cursor += pair_batch_size
                pids = pos_idx[selected]
                nids = neg_idx[selected]
                combined = np.concatenate([xb_np, x_train[pids], x_train[nids]], axis=0)
            else:
                selected = np.empty(0, dtype=np.int64)
                combined = xb_np

            xt = torch.from_numpy(combined).to(device)
            yt = torch.from_numpy(yb_np).to(device)
            logits = model(xt)
            point_count = len(point_ids)
            point_loss = F.binary_cross_entropy_with_logits(logits[:point_count], yt)
            if len(selected):
                pair_count = len(selected)
                positive_scores = logits[point_count:point_count + pair_count]
                negative_scores = logits[point_count + pair_count:point_count + 2 * pair_count]
                pair_loss = F.softplus(-(positive_scores - negative_scores)).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for group in optimizer.param_groups:
            group["lr"] *= 0.5
        scores = predict(model, x_val, device)
        result = run_evaluator(use_npz, val_users, y_val, scores)
        gauc, _, _ = metric_values(result)
        if gauc > best_gauc + 1.0e-7:
            best_gauc = gauc
            best_scores = scores.copy()
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    return best_scores


def write_outputs(out_dir, users, videos, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    prediction_path = os.path.join(out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, user, video, "{:.10g}".format(float(score))])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    npz_path = os.path.join(args.data_dir, "train.npz")
    if os.path.exists(npz_path) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        loaded = load_npz(args.data_dir)
    else:
        loaded = load_csv(args.data_dir)
    x_train, y_train, train_users, x_val, y_val, val_users, val_videos, field_dims, use_npz = loaded

    max_epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        max_epochs = min(max_epochs, max(1, int(smoke)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_scores = []
    for offset in (0, 1009, 2027):
        scores = train_member(
            x_train, y_train, train_users, x_val, y_val, val_users,
            field_dims, use_npz, args.seed + offset, max_epochs, device
        )
        member_scores.append(scores.astype(np.float64))
    ensemble_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    result = run_evaluator(use_npz, val_users, y_val, ensemble_scores)
    gauc, ndcg, primary = metric_values(result)
    metrics = {"gauc": gauc, "ndcg5": ndcg, "primary": primary}
    write_outputs(args.out_dir, val_users, val_videos, ensemble_scores, metrics)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        main()
