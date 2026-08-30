import argparse
import csv
import json
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def parse_date_value(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "")
    try:
        return datetime.strptime(text[:8], "%Y%m%d").toordinal()
    except Exception:
        return 0


def load_npz(data_dir):
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    xtr = np.asarray(train["X"], dtype=np.int64)
    xva = np.asarray(val["X"], dtype=np.int64)
    ytr = np.asarray(train["y"], dtype=np.float32)
    yva = np.asarray(val["y"], dtype=np.float32)
    train_users = np.asarray(train["user"])
    val_users = np.asarray(val["user"])
    dates = np.asarray(train["date"])
    field_dims = np.asarray(train["field_dims"], dtype=np.int64).reshape(-1)
    video_offset = int(field_dims[0])
    val_videos = xva[:, 1] - video_offset
    return xtr, ytr, train_users, dates, xva, yva, val_users, val_videos, field_dims, True


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "user_id": row["user_id"],
                    "video_id": row["video_id"],
                    "tab": row["tab"],
                    "duration_ms": float(row["duration_ms"] or 0.0),
                    "date": row["date"],
                    "long_view": float(row["long_view"]),
                }
            )
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))

    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9))) if len(durations) else np.asarray([], dtype=np.float64)

    user_values = sorted({row["user_id"] for row in train_rows})
    video_values = sorted({row["video_id"] for row in train_rows})
    tab_values = sorted({row["tab"] for row in train_rows})
    user_map = {value: i + 1 for i, value in enumerate(user_values)}
    video_map = {value: i + 1 for i, value in enumerate(video_values)}
    tab_map = {value: i + 1 for i, value in enumerate(tab_values)}

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 2, len(tab_map) + 1, len(cuts) + 2],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = 1 + offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            bucket = int(np.searchsorted(cuts, row["duration_ms"], side="right")) + 1
            x[i, 4] = bucket + offsets[4]
        return x

    xtr = encode(train_rows)
    xva = encode(val_rows)
    ytr = np.asarray([row["long_view"] for row in train_rows], dtype=np.float32)
    yva = np.asarray([row["long_view"] for row in val_rows], dtype=np.float32)
    train_users = np.asarray([row["user_id"] for row in train_rows])
    val_users = np.asarray([row["user_id"] for row in val_rows])
    dates = np.asarray([row["date"] for row in train_rows])
    val_videos = np.asarray([row["video_id"] for row in val_rows])
    return xtr, ytr, train_users, dates, xva, yva, val_users, val_videos, field_dims, False


def recency_weights(dates, half_life=7.0):
    ordinals = np.asarray([parse_date_value(value) for value in dates], dtype=np.float64)
    valid = ordinals > 0
    if not valid.any():
        return np.ones(len(ordinals), dtype=np.float32)
    latest = float(ordinals[valid].max())
    ages = np.maximum(0.0, latest - ordinals)
    weights = np.exp2(-ages / half_life)
    weights[~valid] = 1.0
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def make_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if len(positives) and len(negatives):
            positive_parts.append(positives)
            negative_parts.append(rng.choice(negatives, size=len(positives), replace=True))
    if not positive_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class RegularizedDCN(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, dropout=0.30):
        super().__init__()
        total_features = int(np.sum(field_dims))
        width = len(field_dims) * embedding_dim
        self.embedding = nn.Embedding(total_features, embedding_dim)
        self.linear_embedding = nn.Embedding(total_features, 1)
        self.cross_w = nn.Linear(width, 1, bias=False)
        self.cross_b = nn.Parameter(torch.zeros(width))
        self.cross_out = nn.Linear(width, 1)
        self.mlp = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, x):
        embedded = self.embedding(x)
        x0 = embedded.flatten(1)
        crossed = x0 * self.cross_w(x0) + self.cross_b + x0
        linear = self.linear_embedding(x).sum(dim=1).squeeze(-1)
        return linear + self.cross_out(crossed).squeeze(-1) + self.mlp(x0).squeeze(-1)

    def accessed_row_l2(self, x):
        main_penalty = self.embedding(x).pow(2).sum(dim=-1).mean()
        linear_penalty = self.linear_embedding(x).pow(2).mean()
        return main_penalty + linear_penalty


def predict(model, x, device):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), 131072):
            end = min(start + 131072, len(x))
            batch = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            output[start:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return output


def evaluate_metrics(is_npz, users, labels, scores):
    if is_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def train_model(xtr, ytr, users, dates, xva, yva, val_users, field_dims, is_npz, seed, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedDCN(field_dims, embedding_dim=16, dropout=0.30).to(device)

    embedding_params = [model.embedding.weight, model.linear_embedding.weight]
    embedding_ids = {id(parameter) for parameter in embedding_params}
    dense_params = [parameter for parameter in model.parameters() if id(parameter) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=0.002,
    )

    bce = nn.BCEWithLogitsLoss(reduction="none")
    weights = recency_weights(dates, half_life=7.0)
    pair_pos, pair_neg = make_pairs(users, ytr, seed + 17)
    rng = np.random.default_rng(seed)
    batch_size = 16384
    pair_batch_size = 4096
    best_gauc = -1.0
    best_state = None
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(xtr))
        for start in range(0, len(order), batch_size):
            point_indices = order[start:start + batch_size]
            point_count = len(point_indices)

            if len(pair_pos):
                chosen = rng.integers(0, len(pair_pos), size=min(pair_batch_size, len(pair_pos)))
                positive_indices = pair_pos[chosen]
                negative_indices = pair_neg[chosen]
                combined_x = np.concatenate(
                    [xtr[point_indices], xtr[positive_indices], xtr[negative_indices]], axis=0
                )
            else:
                positive_indices = np.empty(0, dtype=np.int64)
                negative_indices = np.empty(0, dtype=np.int64)
                combined_x = xtr[point_indices]

            combined_tensor = torch.as_tensor(combined_x, dtype=torch.long, device=device)
            logits = model(combined_tensor)
            point_logits = logits[:point_count]
            point_labels = torch.as_tensor(ytr[point_indices], dtype=torch.float32, device=device)
            point_weights = torch.as_tensor(weights[point_indices], dtype=torch.float32, device=device)
            point_loss = (bce(point_logits, point_labels) * point_weights).sum() / point_weights.sum().clamp_min(1e-8)

            if len(positive_indices):
                pair_count = len(positive_indices)
                positive_logits = logits[point_count:point_count + pair_count]
                negative_logits = logits[point_count + pair_count:point_count + 2 * pair_count]
                pair_weights = torch.as_tensor(weights[positive_indices], dtype=torch.float32, device=device)
                pair_terms = torch.nn.functional.softplus(-(positive_logits - negative_logits))
                pair_loss = (pair_terms * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
            else:
                pair_loss = point_loss

            row_l2 = model.accessed_row_l2(combined_tensor)
            loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-5 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, xva, device)
        metrics = evaluate_metrics(is_npz, val_users, yva, scores)
        if metrics["gauc"] > best_gauc + 1e-7:
            best_gauc = metrics["gauc"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        for group in optimizer.param_groups:
            group["lr"] *= 0.5

        if stale_epochs >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def scalar_value(value):
    return value.item() if isinstance(value, np.generic) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        data = load_npz(args.data_dir)
    else:
        data = load_csv(args.data_dir)

    xtr, ytr, train_users, dates, xva, yva, val_users, val_videos, field_dims, is_npz = data
    epochs = 4
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    model, device = train_model(
        xtr,
        ytr,
        train_users,
        dates,
        xva,
        yva,
        val_users,
        field_dims,
        is_npz,
        args.seed,
        epochs,
    )
    scores = predict(model, xva, device)
    metrics = evaluate_metrics(is_npz, val_users, yva, scores)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user, video, score) in enumerate(zip(val_users, val_videos, scores)):
            writer.writerow([row_id, scalar_value(user), scalar_value(video), float(score)])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
