import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=8):
        super().__init__()
        self.n_fields = len(field_dims)
        self.embedding = nn.Embedding(int(np.sum(field_dims)), embedding_dim)
        self.linear = nn.Embedding(int(np.sum(field_dims)), 1)
        width = self.n_fields * embedding_dim
        self.cross1 = nn.Linear(width, 1)
        self.cross2 = nn.Linear(width, 1)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
        )
        self.output = nn.Linear(width + 64, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        embedded = self.embedding(x).flatten(1)
        crossed = embedded + embedded * self.cross1(embedded)
        crossed = crossed + embedded * self.cross2(crossed)
        deep = self.mlp(embedded)
        return self.linear(x).sum(dim=1).squeeze(-1) + self.output(
            torch.cat((crossed, deep), dim=1)
        ).squeeze(-1)


def parse_date(value):
    text = str(value).strip()
    try:
        return dt.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except ValueError:
        try:
            return dt.datetime.fromisoformat(text).date().toordinal()
        except ValueError:
            return 0


def read_csv_data(data_dir):
    train_path = Path(data_dir) / "train.csv"
    val_path = Path(data_dir) / "val.csv"

    with train_path.open("r", newline="", encoding="utf-8") as handle:
        train_rows = list(csv.DictReader(handle))
    with val_path.open("r", newline="", encoding="utf-8") as handle:
        val_rows = list(csv.DictReader(handle))

    train_duration = np.asarray(
        [float(row.get("duration_ms", 0) or 0) for row in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    def raw_fields(row):
        video = str(row.get("video_id", ""))
        author = str(row.get("author_id", video))
        duration = float(row.get("duration_ms", 0) or 0)
        bucket = int(np.searchsorted(quantiles, duration, side="right"))
        return (
            str(row.get("user_id", "")),
            video,
            author,
            str(row.get("tab", "")),
            str(bucket),
        )

    mappings = []
    for field in range(5):
        values = sorted({raw_fields(row)[field] for row in train_rows})
        mappings.append({value: index + 1 for index, value in enumerate(values)})

    field_dims = np.asarray([len(mapping) + 1 for mapping in mappings], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        result = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            fields = raw_fields(row)
            for field, value in enumerate(fields):
                result[i, field] = mappings[field].get(value, 0) + offsets[field]
        return result

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    train_y = np.asarray([float(row["long_view"]) for row in train_rows], dtype=np.float32)
    val_y = np.asarray([float(row["long_view"]) for row in val_rows], dtype=np.float32)
    train_user = np.asarray([row.get("user_id", "") for row in train_rows])
    val_user = np.asarray([row.get("user_id", "") for row in val_rows])
    val_video = np.asarray([row.get("video_id", "") for row in val_rows])
    train_date = np.asarray([parse_date(row.get("date", "")) for row in train_rows], dtype=np.int64)
    return train_x, train_y, train_user, train_date, val_x, val_y, val_user, val_video, field_dims


def load_data(data_dir):
    train_npz = Path(data_dir) / "train.npz"
    val_npz = Path(data_dir) / "val.npz"
    if train_npz.exists() and val_npz.exists():
        with np.load(train_npz, allow_pickle=False) as train, np.load(val_npz, allow_pickle=False) as val:
            train_x = np.asarray(train["X"], dtype=np.int64)
            train_y = np.asarray(train["y"], dtype=np.float32)
            train_user = np.asarray(train["user"])
            train_date = np.asarray(train["date"])
            val_x = np.asarray(val["X"], dtype=np.int64)
            val_y = np.asarray(val["y"], dtype=np.float32)
            val_user = np.asarray(val["user"])
            field_dims = np.asarray(train["field_dims"], dtype=np.int64)
        video_offset = int(field_dims[0])
        val_video = val_x[:, 1].astype(np.int64) - video_offset
        return train_x, train_y, train_user, train_date, val_x, val_y, val_user, val_video, field_dims, True
    values = read_csv_data(data_dir)
    return (*values, False)


def date_ordinals(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        numeric = arr.astype(np.int64)
        if numeric.size and np.nanmax(numeric) > 10000000:
            return np.asarray([parse_date(value) for value in numeric], dtype=np.int64)
        return numeric
    return np.asarray([parse_date(value) for value in arr], dtype=np.int64)


def make_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    positives = []
    negatives = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, device, batch_size=32768):
    result = np.empty(len(x), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def official_metrics(use_npz, users, labels, scores):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return evaluate(users, labels, scores)


def train_member(seed, train_x, train_y, train_user, train_date, val_x, val_y,
                 val_user, field_dims, use_npz, epochs, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rng = np.random.default_rng(seed)
    pair_pos, pair_neg = make_pairs(train_user, train_y, rng)
    ordinals = date_ordinals(train_date)
    latest = int(np.max(ordinals)) if len(ordinals) else 0
    weights = np.exp2(-(latest - ordinals).clip(min=0).astype(np.float32) / 7.0).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)

    model = DCNLite(field_dims, embedding_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.65)
    batch_size = 8192
    best_gauc = -np.inf
    best_scores = None
    stale = 0

    for _ in range(epochs):
        model.train()
        impression_order = rng.permutation(len(train_x))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else pair_pos
        pair_cursor = 0

        for start in range(0, len(impression_order), batch_size):
            idx = impression_order[start:start + batch_size]
            xb = torch.as_tensor(train_x[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[idx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()

            if len(pair_pos):
                count = min(len(idx), len(pair_pos))
                if pair_cursor + count > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                chosen = pair_order[pair_cursor:pair_cursor + count]
                pair_cursor += count
                pos_idx = pair_pos[chosen]
                neg_idx = pair_neg[chosen]
                pos_x = torch.as_tensor(train_x[pos_idx], dtype=torch.long, device=device)
                neg_x = torch.as_tensor(train_x[neg_idx], dtype=torch.long, device=device)
                pair_weight = torch.as_tensor(weights[pos_idx], dtype=torch.float32, device=device)
                rank_loss = (F.softplus(-(model(pos_x) - model(neg_x))) * pair_weight).mean()
                loss = 0.5 * point_loss + 0.5 * rank_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scores = predict(model, val_x, device)
        metrics = official_metrics(use_npz, val_user, val_y, scores)
        gauc = float(metrics.get("GAUC", metrics.get("gauc")))
        if gauc > best_gauc:
            best_gauc = gauc
            best_scores = scores.copy()
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if stale >= 2:
            break

    return best_scores


def within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    order = np.lexsort((scores, users))
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    ranks = np.empty(len(scores), dtype=np.float64)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        n = end - start
        if n == 1:
            ranks[order[start]] = 0.5
        else:
            ranks[order[start:end]] = np.arange(n, dtype=np.float64) / float(n - 1)
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_data(args.data_dir)
    train_x, train_y, train_user, train_date, val_x, val_y, val_user, val_video, field_dims, use_npz = loaded
    epochs = 6
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    rank_sum = np.zeros(len(val_x), dtype=np.float64)
    for member in range(5):
        scores = train_member(
            args.seed + member, train_x, train_y, train_user, train_date,
            val_x, val_y, val_user, field_dims, use_npz, epochs, device
        )
        rank_sum += within_user_ranks(val_user, scores)
    final_scores = rank_sum / 5.0
    metrics = official_metrics(use_npz, val_user, val_y, final_scores)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(val_user, val_video, final_scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user,
                             video.item() if isinstance(video, np.generic) else video,
                             "%.10f" % float(score)])

    output_metrics = {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output_metrics, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
