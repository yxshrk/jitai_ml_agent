import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def silence_process():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    train_x = np.asarray(tr["X"], dtype=np.int64)
    val_x = np.asarray(va["X"], dtype=np.int64)
    train_y = np.asarray(tr["y"], dtype=np.float32)
    val_y = np.asarray(va["y"], dtype=np.float32)
    train_user = np.asarray(tr["user"])
    val_user = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    val_video = val_x[:, 1] - offsets[1]
    row_ids = np.arange(len(val_x), dtype=np.int64)
    tr.close()
    va.close()
    return train_x, train_y, train_user, val_x, val_y, val_user, val_video, row_ids, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "row_id": row.get("row_id", str(i)),
            }
            if training:
                item["y"] = float(row["long_view"])
            else:
                item["y"] = float(row["long_view"])
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping(r["user"] for r in train_rows)
    video_map = make_mapping(r["video"] for r in train_rows)
    tab_map = make_mapping(r["tab"] for r in train_rows)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    cuts = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user"], 0)
            x[i, 1] = video_map.get(r["video"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(r["tab"], 0)
            x[i, 4] = int(np.searchsorted(cuts, r["duration"], side="right"))
        x += offsets[None, :]
        return x

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    train_y = np.asarray([r["y"] for r in train_rows], dtype=np.float32)
    val_y = np.asarray([r["y"] for r in val_rows], dtype=np.float32)
    train_user = np.asarray([r["user"] for r in train_rows])
    val_user = np.asarray([r["user"] for r in val_rows])
    val_video = np.asarray([r["video"] for r in val_rows])
    row_ids = np.asarray([r["row_id"] for r in val_rows])
    return train_x, train_y, train_user, val_x, val_y, val_user, val_video, row_ids, field_dims, False


class DCNRegularized(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, cross_layers=2, dropout=0.3):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)])
        self.cross_out = nn.Linear(width, 1)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
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

    def forward(self, x):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(cross * w, dim=1, keepdim=True)
            cross = x0 * scale + b + cross
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(1)
        logits = first_order + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1) + self.bias
        row_l2 = emb.square().sum(dim=(1, 2)).mean()
        row_l2 = row_l2 + self.linear_embedding(x).square().sum(dim=(1, 2)).mean()
        return logits, row_l2


def prepare_pair_sampler(x, y):
    users = x[:, 0]
    neg_idx = np.flatnonzero(y < 0.5)
    if len(neg_idx) == 0:
        return None
    order = np.argsort(users[neg_idx], kind="stable")
    sorted_neg = neg_idx[order]
    neg_users = users[sorted_neg]
    unique_users, starts, counts = np.unique(neg_users, return_index=True, return_counts=True)
    positive = np.flatnonzero(y >= 0.5)
    locations = np.searchsorted(unique_users, users[positive])
    clipped = np.minimum(locations, len(unique_users) - 1)
    valid = (locations < len(unique_users)) & (unique_users[clipped] == users[positive])
    positive = positive[valid]
    locations = locations[valid]
    if len(positive) == 0:
        return None
    return positive, locations, sorted_neg, starts, counts


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(xb)
            result[start:end] = torch.sigmoid(logits).cpu().numpy()
    return result


def metric_values(evaluator, users, labels, scores):
    values = evaluator(users, labels, scores)
    return {
        "gauc": float(values["GAUC"]),
        "ndcg5": float(values["nDCG@5"]),
        "primary": float(values["primary"]),
    }


def train_model(train_x, train_y, train_user, val_x, val_y, val_user, field_dims, evaluator, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8192 if device.type == "cuda" else 4096
    model = DCNRegularized(field_dims, embed_dim=16, hidden_dim=128, cross_layers=2, dropout=0.3).to(device)
    embedding_params = [model.embedding.weight, model.linear_embedding.weight]
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW([
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": dense_params, "weight_decay": 1e-3},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    pair_data = prepare_pair_sampler(train_x, train_y)
    rng = np.random.default_rng(seed)
    epochs = 15
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    best_gauc = -float("inf")
    best_state = None
    stale = 0
    row_l2_weight = 1e-4

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(train_x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            point_logits, point_l2 = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(point_logits, yb)

            if pair_data is not None:
                positive, locations, sorted_neg, starts, counts = pair_data
                pair_count = max(1, len(idx) // 2)
                chosen = rng.integers(0, len(positive), size=pair_count)
                pos_idx = positive[chosen]
                loc = locations[chosen]
                offsets = (rng.random(pair_count) * counts[loc]).astype(np.int64)
                neg_idx = sorted_neg[starts[loc] + offsets]
                pos_x = torch.as_tensor(train_x[pos_idx], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(train_x[neg_idx], dtype=torch.long, device=device)
                pos_logits, pos_l2 = model(pos_x)
                neg_logits, neg_l2 = model(neg_x)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                row_l2 = (point_l2 + pos_l2 + neg_l2) / 3.0
                loss = 0.5 * point_loss + 0.5 * pair_loss + row_l2_weight * row_l2
            else:
                loss = point_loss + row_l2_weight * point_l2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_scores = predict(model, val_x, device, batch_size * 2)
        current = metric_values(evaluator, val_user, val_y, val_scores)
        if current["gauc"] > best_gauc + 1e-8:
            best_gauc = current["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 5:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, val_x, device, batch_size * 2)
    return scores


def write_outputs(out_dir, row_ids, users, videos, scores, metrics):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, user, video, score in zip(row_ids, users, videos, scores):
            writer.writerow([row_id, user, video, format(float(score), ".10g")])
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    silence_process()
    set_seed(args.seed)

    fast = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if fast:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate
    else:
        data = load_csv(args.data_dir)
        from harness.evaluate_provisional import evaluate

    train_x, train_y, train_user, val_x, val_y, val_user, val_video, row_ids, field_dims, _ = data
    scores = train_model(train_x, train_y, train_user, val_x, val_y, val_user, field_dims, evaluate, args.seed)
    metrics = metric_values(evaluate, val_user, val_y, scores)
    write_outputs(args.out_dir, row_ids, val_user, val_video, scores, metrics)


if __name__ == "__main__":
    main()
