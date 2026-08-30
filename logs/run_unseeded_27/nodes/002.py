import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import csv
import io
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize_offset_encoding(x, field_dims):
    x = np.asarray(x)
    dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(dims[:-1])))
    out = x.astype(np.int64, copy=True)
    for j, (dim, offset) in enumerate(zip(dims, offsets)):
        col = out[:, j]
        is_global = col.size == 0 or (int(col.min()) >= int(offset) and int(col.max()) < int(offset + dim))
        if not is_global:
            is_local = col.size == 0 or (int(col.min()) >= 0 and int(col.max()) < int(dim))
            if not is_local:
                raise ValueError("Feature indices are outside field dimensions")
            out[:, j] += offset
    return out.astype(np.int32), offsets


def load_npz_data(data_dir):
    with np.load(data_dir / "train.npz", allow_pickle=False) as tr:
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
        x_train_raw = np.asarray(tr["X"])
        y_train = np.asarray(tr["y"], dtype=np.float32)
    with np.load(data_dir / "val.npz", allow_pickle=False) as va:
        x_val_raw = np.asarray(va["X"])
        y_val = np.asarray(va["y"], dtype=np.float32)
        val_users = np.asarray(va["user"])
    x_train, offsets = normalize_offset_encoding(x_train_raw, field_dims)
    x_val, _ = normalize_offset_encoding(x_val_raw, field_dims)
    val_videos = x_val[:, 1].astype(np.int64) - int(offsets[1])
    return x_train, y_train, x_val, y_val, val_users, val_users, val_videos, field_dims


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def load_csv_data(data_dir):
    user_map = {}
    video_map = {}
    tab_map = {}
    train_user = []
    train_video = []
    train_tab = []
    train_duration = []
    train_y = []

    def fit_code(mapping, value):
        code = mapping.get(value)
        if code is None:
            code = len(mapping)
            mapping[value] = code
        return code

    with open(data_dir / "train.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_user.append(fit_code(user_map, row["user_id"]))
            train_video.append(fit_code(video_map, row["video_id"]))
            train_tab.append(fit_code(tab_map, row["tab"]))
            train_duration.append(max(0.0, safe_float(row["duration_ms"])))
            train_y.append(safe_float(row["long_view"]))

    train_duration_arr = np.asarray(train_duration, dtype=np.float64)
    if train_duration_arr.size:
        edges = np.quantile(train_duration_arr, np.linspace(0.0, 1.0, 11))
        inner_edges = edges[1:-1]
    else:
        inner_edges = np.zeros(9, dtype=np.float64)
    train_bucket = np.searchsorted(inner_edges, train_duration_arr, side="right").astype(np.int32)

    val_user = []
    val_video = []
    val_tab = []
    val_duration = []
    val_y = []
    val_user_raw = []
    val_video_raw = []
    unknown_user = len(user_map)
    unknown_video = len(video_map)
    unknown_tab = len(tab_map)

    with open(data_dir / "val.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_user = row["user_id"]
            raw_video = row["video_id"]
            val_user_raw.append(raw_user)
            val_video_raw.append(raw_video)
            val_user.append(user_map.get(raw_user, unknown_user))
            val_video.append(video_map.get(raw_video, unknown_video))
            val_tab.append(tab_map.get(row["tab"], unknown_tab))
            val_duration.append(max(0.0, safe_float(row["duration_ms"])))
            val_y.append(safe_float(row["long_view"]))

    val_duration_arr = np.asarray(val_duration, dtype=np.float64)
    val_bucket = np.searchsorted(inner_edges, val_duration_arr, side="right").astype(np.int32)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims[:-1])))

    x_train = np.column_stack((
        np.asarray(train_user, dtype=np.int64),
        np.asarray(train_video, dtype=np.int64),
        np.zeros(len(train_y), dtype=np.int64),
        np.asarray(train_tab, dtype=np.int64),
        train_bucket.astype(np.int64),
    ))
    x_val = np.column_stack((
        np.asarray(val_user, dtype=np.int64),
        np.asarray(val_video, dtype=np.int64),
        np.zeros(len(val_y), dtype=np.int64),
        np.asarray(val_tab, dtype=np.int64),
        val_bucket.astype(np.int64),
    ))
    x_train += offsets[None, :]
    x_val += offsets[None, :]
    return (
        x_train.astype(np.int32),
        np.asarray(train_y, dtype=np.float32),
        x_val.astype(np.int32),
        np.asarray(val_y, dtype=np.float32),
        np.asarray(val_user_raw),
        val_user_raw,
        val_video_raw,
        field_dims,
    )


class FactorizationMachine(nn.Module):
    def __init__(self, total_features, rank):
        super().__init__()
        self.linear = nn.Embedding(total_features, 1)
        self.embedding = nn.Embedding(total_features, rank)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        linear_term = self.linear(x).sum(dim=1).squeeze(-1)
        latent = self.embedding(x)
        summed = latent.sum(dim=1)
        interaction = 0.5 * (summed.square() - latent.square().sum(dim=1)).sum(dim=1)
        return self.bias + linear_term + interaction


def predict(model, x, device, batch_size=65536):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            output[start:end] = torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32)
    return output


def quiet_evaluate(evaluate_fn, users, labels, scores):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = evaluate_fn(users, labels, scores)
    gauc = result["GAUC"] if "GAUC" in result else result["gauc"]
    ndcg = result["nDCG@5"] if "nDCG@5" in result else result["ndcg5"]
    primary = result["primary"]
    return {"gauc": float(gauc), "ndcg5": float(ndcg), "primary": float(primary)}


def train_model(x_train, y_train, x_val, y_val, eval_users, field_dims, evaluate_fn, seed, epochs):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FactorizationMachine(int(np.sum(field_dims)), rank=20).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [model.embedding.weight, model.linear.weight], "weight_decay": 3e-4},
        {"params": [model.bias], "weight_decay": 0.0},
    ], lr=0.003)
    batch_size = 32768 if device.type == "cuda" else 16384
    rng = np.random.RandomState(seed)
    best_primary = -float("inf")
    best_scores = None
    stale = 0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            optimizer.step()
        scores = predict(model, x_val, device)
        metrics = quiet_evaluate(evaluate_fn, eval_users, y_val, scores)
        if metrics["primary"] > best_primary + 1e-8:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

    if best_scores is None:
        best_scores = predict(model, x_val, device)
    return best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast_path = (data_dir / "train.npz").is_file() and (data_dir / "val.npz").is_file()

    if fast_path:
        x_train, y_train, x_val, y_val, eval_users, output_users, output_videos, field_dims = load_npz_data(data_dir)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from data.official.evaluate import evaluate as evaluate_fn
    else:
        x_train, y_train, x_val, y_val, eval_users, output_users, output_videos, field_dims = load_csv_data(data_dir)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from harness.evaluate_provisional import evaluate as evaluate_fn

    epochs = 8
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    scores = train_model(
        x_train, y_train, x_val, y_val, eval_users, field_dims,
        evaluate_fn, args.seed, epochs,
    )
    metrics = quiet_evaluate(evaluate_fn, eval_users, y_val, scores)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(zip(output_users, output_videos, scores)):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10f")])

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
