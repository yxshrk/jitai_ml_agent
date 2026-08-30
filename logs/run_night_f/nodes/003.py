import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def scalar_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def load_fast(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return None
    train_npz = np.load(train_path, allow_pickle=False)
    val_npz = np.load(val_path, allow_pickle=False)
    x_train = np.asarray(train_npz["X"], dtype=np.int64)
    y_train = np.asarray(train_npz["y"], dtype=np.float32)
    user_train = np.asarray(train_npz["user"])
    x_val = np.asarray(val_npz["X"], dtype=np.int64)
    y_val = np.asarray(val_npz["y"], dtype=np.float32)
    user_val = np.asarray(val_npz["user"])
    field_dims = np.asarray(train_npz["field_dims"], dtype=np.int64)
    user_out = user_val
    video_offset = int(field_dims[0])
    video_out = x_val[:, 1] - video_offset
    return x_train, y_train, user_train, x_val, y_val, user_val, field_dims, user_out, video_out


def read_selected_csv(path, is_train):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        pos = {name: i for i, name in enumerate(header)}
        required = ["user_id", "video_id", "tab", "duration_ms", "long_view"]
        for name in required:
            if name not in pos:
                raise ValueError("missing required column")
        has_author = "author_id" in pos
        users = []
        videos = []
        authors = []
        tabs = []
        durations = []
        labels = []
        for row in reader:
            users.append(row[pos["user_id"]])
            videos.append(row[pos["video_id"]])
            authors.append(row[pos["author_id"]] if has_author else "__NO_AUTHOR__")
            tabs.append(row[pos["tab"]])
            try:
                durations.append(float(row[pos["duration_ms"]]))
            except ValueError:
                durations.append(0.0)
            labels.append(float(row[pos["long_view"]]))
    return {
        "user": users,
        "video": videos,
        "author": authors,
        "tab": tabs,
        "duration": np.asarray(durations, dtype=np.float64),
        "label": np.asarray(labels, dtype=np.float32),
    }


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_values(values, mapping):
    return np.fromiter((mapping.get(v, 0) for v in values), dtype=np.int64, count=len(values))


def load_csv_fallback(data_dir):
    train = read_selected_csv(os.path.join(data_dir, "train.csv"), True)
    val = read_selected_csv(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping(train["user"])
    video_map = make_mapping(train["video"])
    author_map = make_mapping(train["author"])
    tab_map = make_mapping(train["tab"])
    finite_duration = train["duration"][np.isfinite(train["duration"])]
    if finite_duration.size:
        thresholds = np.unique(np.quantile(finite_duration, np.linspace(0.1, 0.9, 9)))
    else:
        thresholds = np.asarray([], dtype=np.float64)

    def assemble(data):
        fields = [
            encode_values(data["user"], user_map),
            encode_values(data["video"], video_map),
            encode_values(data["author"], author_map),
            encode_values(data["tab"], tab_map),
            np.searchsorted(thresholds, np.nan_to_num(data["duration"], nan=0.0), side="right").astype(np.int64) + 1,
        ]
        dims = np.asarray([
            len(user_map) + 1,
            len(video_map) + 1,
            len(author_map) + 1,
            len(tab_map) + 1,
            len(thresholds) + 2,
        ], dtype=np.int64)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(dims)[:-1]))
        return np.stack(fields, axis=1) + offsets, dims

    x_train, field_dims = assemble(train)
    x_val, _ = assemble(val)
    user_train = np.asarray(train["user"], dtype=str)
    user_val = np.asarray(val["user"], dtype=str)
    return (
        x_train,
        train["label"],
        user_train,
        x_val,
        val["label"],
        user_val,
        field_dims,
        np.asarray(val["user"], dtype=str),
        np.asarray(val["video"], dtype=str),
    )


def build_top5_groups(users, labels, seed):
    n = len(labels)
    rng = np.random.default_rng(seed)
    random_key = rng.random(n)
    user_key = np.asarray([scalar_text(v) for v in users])
    order = np.lexsort((random_key, user_key))
    sorted_users = user_key[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n]))
    groups = []
    for start, end in zip(starts, ends):
        idx = order[start:end]
        complete = (idx.size // 5) * 5
        if complete == 0:
            continue
        blocks = idx[:complete].reshape(-1, 5)
        block_labels = labels[blocks]
        mixed = (block_labels.sum(axis=1) > 0.0) & (block_labels.sum(axis=1) < 5.0)
        if np.any(mixed):
            groups.append(blocks[mixed])
    if not groups:
        return np.empty((0, 5), dtype=np.int64)
    return np.concatenate(groups, axis=0).astype(np.int64, copy=False)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.2):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        self.cross_weight = nn.Parameter(torch.empty(width))
        self.cross_bias = nn.Parameter(torch.zeros(width))
        self.hidden = nn.Linear(width, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width + hidden_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear_embedding.weight)
        nn.init.normal_(self.cross_weight, std=0.01)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        emb = self.embedding(x)
        x0 = emb.flatten(1)
        cross_scale = torch.sum(x0 * self.cross_weight, dim=1, keepdim=True)
        cross = x0 * cross_scale + self.cross_bias + x0
        hidden = self.dropout(F.relu(self.hidden(self.dropout(cross))))
        dense_logit = self.output(torch.cat((cross, hidden), dim=1)).squeeze(1)
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return dense_logit + first_order + self.global_bias


def embedding_row_penalty(model, x):
    unique_rows = torch.unique(x)
    return model.embedding(unique_rows).pow(2).mean() + model.linear_embedding(unique_rows).pow(2).mean()


def train_logloss_phase(model, optimizer, x, y, batch_size, rng, device):
    model.train()
    order = rng.permutation(len(y))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        xb = torch.as_tensor(x[idx], dtype=torch.long, device=device)
        yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss = loss + 1e-6 * embedding_row_penalty(model, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def delta_ndcg_loss(scores, labels):
    batch, width = scores.shape
    order = torch.argsort(scores.detach(), dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(width, device=scores.device).view(1, -1).expand(batch, -1)
    ranks.scatter_(1, order, rank_values)
    discounts_table = 1.0 / torch.log2(torch.arange(width, device=scores.device, dtype=torch.float32) + 2.0)
    discounts = discounts_table[ranks]
    positive_count = labels.sum(dim=1).long().clamp(min=1, max=width)
    prefix = torch.cumsum(discounts_table, dim=0)
    idcg = prefix[positive_count - 1].view(-1, 1, 1)
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    pair_mask = label_diff > 0.5
    discount_delta = torch.abs(discounts.unsqueeze(2) - discounts.unsqueeze(1))
    weights = discount_delta / idcg
    weighted = F.softplus(-score_diff) * weights * pair_mask
    denominator = (weights * pair_mask).sum().clamp_min(1e-8)
    return weighted.sum() / denominator


def train_lambda_phase(model, optimizer, x, y, groups, group_batch_size, rng, device):
    if len(groups) == 0:
        return
    model.train()
    order = rng.permutation(len(groups))
    for start in range(0, len(order), group_batch_size):
        selected = groups[order[start:start + group_batch_size]]
        flat = selected.reshape(-1)
        xb = torch.as_tensor(x[flat], dtype=torch.long, device=device)
        yb = torch.as_tensor(y[flat], dtype=torch.float32, device=device).reshape(-1, 5)
        optimizer.zero_grad(set_to_none=True)
        scores = model(xb).reshape(-1, 5)
        loss = delta_ndcg_loss(scores, yb)
        loss = loss + 1e-6 * embedding_row_penalty(model, xb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def predict(model, x, device, batch_size=16384):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            result[start:start + len(xb)] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_evaluate(fast_path, users, labels, scores):
    if fast_path:
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

    seed_everything(args.seed)
    fast = load_fast(args.data_dir)
    fast_path = fast is not None
    data = fast if fast_path else load_csv_fallback(args.data_dir)
    x_train, y_train, user_train, x_val, y_val, user_val, field_dims, user_out, video_out = data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims, embed_dim=16, hidden_dim=128, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    groups = build_top5_groups(user_train, y_train, args.seed + 101)
    rng = np.random.default_rng(args.seed + 202)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    best_gauc = -math.inf
    best_state = None
    stale = 0
    for _ in range(epochs):
        train_logloss_phase(model, optimizer, x_train, y_train, 8192, rng, device)
        scores = predict(model, x_val, device)
        metrics = official_evaluate(fast_path, user_val, y_val, scores)
        gauc = float(metrics["GAUC"])
        if gauc > best_gauc:
            best_gauc = gauc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        train_lambda_phase(model, optimizer, x_train, y_train, groups, 1024, rng, device)
        scores = predict(model, x_val, device)
        metrics = official_evaluate(fast_path, user_val, y_val, scores)
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
    final_scores = predict(model, x_val, device)
    final_metrics = official_evaluate(fast_path, user_val, y_val, final_scores)

    os.makedirs(args.out_dir, exist_ok=True)
    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            writer.writerow([i, scalar_text(user_out[i]), scalar_text(video_out[i]), format(float(score), ".10g")])

    output_metrics = {
        "gauc": float(final_metrics["GAUC"]),
        "ndcg5": float(final_metrics["nDCG@5"]),
        "primary": float(final_metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
