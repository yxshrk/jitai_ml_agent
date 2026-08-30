import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scalar_text(value):
    if value is None or value == "":
        return "__MISSING__"
    return str(value)


def load_csv_data(data_dir):
    train_path = Path(data_dir) / "train.csv"
    val_path = Path(data_dir) / "val.csv"

    train_rows = []
    with train_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_rows.append(row)

    train_duration = np.asarray(
        [float(row.get("duration_ms") or 0.0) for row in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    field_names = ("user_id", "video_id", "author_id", "tab")
    vocabularies = []
    for name in field_names:
        values = sorted({scalar_text(row.get(name)) for row in train_rows})
        vocabularies.append({value: i + 1 for i, value in enumerate(values)})

    field_dims = [len(vocab) + 1 for vocab in vocabularies] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def encode_rows(rows, need_label):
        n = len(rows)
        x = np.empty((n, 5), dtype=np.int64)
        y = np.empty(n, dtype=np.float32) if need_label else None
        users = np.empty(n, dtype=np.int64)
        raw_users = []
        raw_videos = []
        for i, row in enumerate(rows):
            encoded = []
            for name, vocab in zip(field_names, vocabularies):
                encoded.append(vocab.get(scalar_text(row.get(name)), 0))
            duration = float(row.get("duration_ms") or 0.0)
            duration_bucket = int(np.searchsorted(quantiles, duration, side="right"))
            local = np.asarray(encoded + [duration_bucket], dtype=np.int64)
            x[i] = local + offsets
            users[i] = encoded[0]
            raw_users.append(row.get("user_id", ""))
            raw_videos.append(row.get("video_id", ""))
            if need_label:
                y[i] = float(row["long_view"])
        return x, y, users, raw_users, raw_videos

    train_x, train_y, train_user, _, _ = encode_rows(train_rows, True)
    del train_rows

    val_rows = []
    with val_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val_rows.append(row)
    val_x, val_y, val_user, raw_users, raw_videos = encode_rows(val_rows, True)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "pred_users": raw_users,
        "pred_videos": raw_videos,
        "npz": False,
    }


def load_data(data_dir):
    train_npz = Path(data_dir) / "train.npz"
    val_npz = Path(data_dir) / "val.npz"
    if not (train_npz.exists() and val_npz.exists()):
        return load_csv_data(data_dir)

    with np.load(train_npz, allow_pickle=False) as train:
        train_x = np.asarray(train["X"], dtype=np.int64)
        train_y = np.asarray(train["y"], dtype=np.float32).reshape(-1)
        train_user = np.asarray(train["user"]).reshape(-1)
        field_dims = np.asarray(train["field_dims"], dtype=np.int64).reshape(-1)

    with np.load(val_npz, allow_pickle=False) as val:
        val_x = np.asarray(val["X"], dtype=np.int64)
        val_y = np.asarray(val["y"], dtype=np.float32).reshape(-1)
        val_user = np.asarray(val["user"]).reshape(-1)
        if "field_dims" in val:
            val_dims = np.asarray(val["field_dims"], dtype=np.int64).reshape(-1)
            if val_dims.shape == field_dims.shape:
                field_dims = np.maximum(field_dims, val_dims)

    offsets = np.cumsum(np.concatenate(([0], field_dims[:-1]))).astype(np.int64)
    pred_users = val_user.tolist()
    pred_videos = (val_x[:, 1] - offsets[1]).tolist()
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "val_x": val_x,
        "val_y": val_y,
        "val_user": val_user,
        "field_dims": field_dims,
        "pred_users": pred_users,
        "pred_videos": pred_videos,
        "npz": True,
    }


def build_pairs(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.concatenate(([True], sorted_users[1:] != sorted_users[:-1], [True]))
    )
    positive_parts = []
    negative_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        group_labels = labels[indices] >= 0.5
        positives = indices[group_labels]
        negatives = indices[~group_labels]
        if positives.size == 0 or negatives.size == 0:
            continue
        count = max(positives.size, negatives.size)
        positive_parts.append(np.resize(positives, count))
        negative_parts.append(np.resize(negatives, count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dim=128, dropout=0.30):
        super().__init__()
        total_dim = int(np.sum(field_dims))
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        input_dim = len(field_dims) * embedding_dim
        self.input_dropout = nn.Dropout(dropout)
        self.cross_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(2)]
        )
        self.cross_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(2)]
        )
        for weight in self.cross_weights:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + hidden_dim // 2, 1)

    def forward(self, x, return_embedding=False):
        embedded = self.embedding(x).flatten(1)
        x0 = self.input_dropout(embedded)
        crossed = x0
        for weight, bias in zip(self.cross_weights, self.cross_biases):
            projection = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = x0 * projection + bias + crossed
        deep = self.mlp(x0)
        logits = self.output(torch.cat((crossed, deep), dim=1)).squeeze(1)
        if return_embedding:
            return logits, embedded
        return logits


def predict(model, x, device, batch_size=32768):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = model(batch).detach().cpu().numpy()
    return result


def evaluator_for(npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics.get("primary")),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_dir)

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_user = data["train_user"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    val_user = data["val_user"]

    pair_pos, pair_neg = build_pairs(train_user, train_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(data["field_dims"]).to(device)

    embedding_params = list(model.embedding.parameters())
    embedding_ids = {id(p) for p in embedding_params}
    dense_params = [p for p in model.parameters() if id(p) not in embedding_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": embedding_params, "weight_decay": 0.0},
            {"params": dense_params, "weight_decay": 1e-3},
        ],
        lr=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=2e-4, min_lr=2e-6
    )

    epochs = 16
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    batch_size = 8192
    rng = np.random.default_rng(args.seed)
    evaluate = evaluator_for(data["npz"])
    best_gauc = -np.inf
    best_state = None
    stale = 0

    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            xb = torch.as_tensor(train_x[batch_indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[batch_indices], dtype=torch.float32, device=device)
            logits, used_embedding = model(xb, return_embedding=True)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if pair_pos.size:
                pair_count = max(1, len(batch_indices) // 2)
                sampled = rng.integers(0, pair_pos.size, size=pair_count)
                pos_x = torch.as_tensor(train_x[pair_pos[sampled]], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(train_x[pair_neg[sampled]], dtype=torch.long, device=device)
                pos_logits, pos_embedding = model(pos_x, return_embedding=True)
                neg_logits, neg_embedding = model(neg_x, return_embedding=True)
                pair_loss = F.softplus(-(pos_logits - neg_logits)).mean()
                row_l2 = (
                    used_embedding.square().sum(dim=1).mean()
                    + pos_embedding.square().sum(dim=1).mean()
                    + neg_embedding.square().sum(dim=1).mean()
                ) / 3.0
            else:
                pair_loss = point_loss.new_zeros(())
                row_l2 = used_embedding.square().sum(dim=1).mean()

            loss = 0.5 * point_loss + 0.5 * pair_loss + 1e-3 * row_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, val_x, device)
        epoch_metrics = normalize_metrics(evaluate(val_user, val_y, val_scores))
        gauc = epoch_metrics["gauc"]
        scheduler.step(gauc)
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 5:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_scores = predict(model, val_x, device)
    metrics = normalize_metrics(evaluate(val_user, val_y, final_scores))

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(
            zip(data["pred_users"], data["pred_videos"], final_scores)
        ):
            writer.writerow([i, user_id, video_id, format(float(score), ".9g")])

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
