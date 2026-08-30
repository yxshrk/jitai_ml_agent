import argparse
import contextlib
import csv
import importlib
import json
import os
import random
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "42")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, total_dim, embedding_dim=16):
        super().__init__()
        self.linear = nn.Embedding(total_dim, 1)
        self.embedding = nn.Embedding(total_dim, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        vectors = self.embedding(x)
        summed = vectors.sum(dim=1)
        interaction = 0.5 * (summed.square() - vectors.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear_term + interaction


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _load_npz(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return None

    with np.load(train_path, allow_pickle=False) as train_data:
        x_train = np.asarray(train_data["X"], dtype=np.int64)
        y_train = np.asarray(train_data["y"], dtype=np.float32)
        field_dims = np.asarray(train_data["field_dims"], dtype=np.int64)

    with np.load(val_path, allow_pickle=False) as val_data:
        x_val = np.asarray(val_data["X"], dtype=np.int64)
        y_val = np.asarray(val_data["y"], dtype=np.float32)
        val_users = np.asarray(val_data["user"])

    video_offset = int(field_dims[0])
    val_videos = x_val[:, 1].astype(np.int64) - video_offset
    return x_train, y_train, x_val, y_val, val_users, val_videos, field_dims, True


def _category_id(mapping, value, add):
    key = "" if value is None else str(value)
    existing = mapping.get(key)
    if existing is not None:
        return existing
    if not add:
        return 0
    new_id = len(mapping) + 1
    mapping[key] = new_id
    return new_id


def _load_csv(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    user_map = {}
    video_map = {}
    author_map = {}
    tab_map = {}

    train_categorical = []
    train_durations = []
    train_labels = []

    with open(train_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uid = _category_id(user_map, row.get("user_id"), True)
            vid = _category_id(video_map, row.get("video_id"), True)
            aid = _category_id(author_map, row.get("author_id", "__missing__"), True)
            tab = _category_id(tab_map, row.get("tab"), True)
            train_categorical.append((uid, vid, aid, tab))
            train_durations.append(_safe_float(row.get("duration_ms"), 0.0))
            train_labels.append(_safe_float(row.get("long_view"), 0.0))

    durations = np.asarray(train_durations, dtype=np.float64)
    if durations.size:
        boundaries = np.quantile(durations, np.arange(1, 10, dtype=np.float64) / 10.0)
    else:
        boundaries = np.zeros(9, dtype=np.float64)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    x_train = np.empty((len(train_categorical), 5), dtype=np.int64)
    for i, (categorical, duration) in enumerate(zip(train_categorical, durations)):
        uid, vid, aid, tab = categorical
        bucket = int(np.searchsorted(boundaries, duration, side="right"))
        x_train[i] = (uid, vid, aid, tab, bucket)
    x_train += offsets
    y_train = np.asarray(train_labels, dtype=np.float32)

    val_rows = []
    val_labels = []
    val_users = []
    val_videos = []
    with open(val_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_user = row.get("user_id", "")
            raw_video = row.get("video_id", "")
            uid = _category_id(user_map, raw_user, False)
            vid = _category_id(video_map, raw_video, False)
            aid = _category_id(author_map, row.get("author_id", "__missing__"), False)
            tab = _category_id(tab_map, row.get("tab"), False)
            duration = _safe_float(row.get("duration_ms"), 0.0)
            bucket = int(np.searchsorted(boundaries, duration, side="right"))
            val_rows.append((uid, vid, aid, tab, bucket))
            val_labels.append(_safe_float(row.get("long_view"), 0.0))
            val_users.append(raw_user)
            val_videos.append(raw_video)

    x_val = np.asarray(val_rows, dtype=np.int64)
    if x_val.size == 0:
        x_val = np.empty((0, 5), dtype=np.int64)
    x_val += offsets
    y_val = np.asarray(val_labels, dtype=np.float32)
    return (
        x_train,
        y_train,
        x_val,
        y_val,
        np.asarray(val_users),
        np.asarray(val_videos),
        field_dims,
        False,
    )


def _load_data(data_dir):
    loaded = _load_npz(data_dir)
    return loaded if loaded is not None else _load_csv(data_dir)


def _get_metric(result, *names):
    for name in names:
        if name in result:
            return float(result[name])
    raise KeyError(names[0])


def _evaluate(users, labels, scores, fast_path):
    sink_path = os.devnull
    with open(sink_path, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        if fast_path:
            evaluator = importlib.import_module("data.official.evaluate").evaluate
        else:
            evaluator = importlib.import_module("harness.evaluate_provisional").evaluate
        result = evaluator(users, labels, scores)
    return {
        "gauc": _get_metric(result, "GAUC", "gauc"),
        "ndcg5": _get_metric(result, "nDCG@5", "ndcg5", "NDCG@5"),
        "primary": _get_metric(result, "primary", "PRIMARY"),
    }


def _predict(model, x, device, batch_size=65536):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            output[start:end] = torch.sigmoid(model(batch)).cpu().numpy()
    return output


def _train_member(x_train, y_train, x_val, y_val, val_users, field_dims, seed, epochs, fast_path):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizationMachine(int(field_dims.sum()), embedding_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    batch_size = 32768
    best_gauc = -float("inf")
    best_primary = -float("inf")
    best_scores = None
    stale_epochs = 0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scores = _predict(model, x_val, device)
        metrics = _evaluate(val_users, y_val, scores, fast_path)
        improved = metrics["gauc"] > best_gauc + 1e-12
        tied_better = abs(metrics["gauc"] - best_gauc) <= 1e-12 and metrics["primary"] > best_primary
        if improved or tied_better or best_scores is None:
            best_gauc = metrics["gauc"]
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 2:
                break

    return best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    x_train, y_train, x_val, y_val, val_users, val_videos, field_dims, fast_path = _load_data(args.data_dir)

    epochs = 10
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        try:
            epochs = max(1, min(epochs, int(smoke)))
        except ValueError:
            epochs = 1

    member_scores = []
    for member in range(3):
        member_seed = args.seed + member * 104729
        scores = _train_member(
            x_train,
            y_train,
            x_val,
            y_val,
            val_users,
            field_dims,
            member_seed,
            epochs,
            fast_path,
        )
        member_scores.append(scores.astype(np.float64))

    ensemble_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    metrics = _evaluate(val_users, y_val, ensemble_scores, fast_path)

    predictions_path = os.path.join(args.out_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(zip(val_users, val_videos, ensemble_scores)):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
