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


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, hidden_dim=128, dropout=0.3, cross_layers=2):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        total_dim = int(np.sum(field_dims))
        input_dim = self.num_fields * embed_dim
        self.embedding = nn.Embedding(total_dim, embed_dim)
        self.linear_embedding = nn.Embedding(total_dim, 1)
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.cross_w = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(cross_layers)]
        )
        self.cross_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(cross_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rep_dim = input_dim + hidden_dim // 2
        self.main_head = nn.Linear(rep_dim, 1)
        self.ordinal_head = nn.Linear(rep_dim, 4)
        self.input_dropout = nn.Dropout(dropout)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for w in self.cross_w:
            nn.init.normal_(w, mean=0.0, std=0.01)

    def forward(self, x):
        embedded = self.embedding(x).reshape(x.shape[0], -1)
        x0 = self.input_dropout(embedded)
        crossed = x0
        for w, b in zip(self.cross_w, self.cross_b):
            scale = torch.sum(crossed * w, dim=1, keepdim=True)
            crossed = x0 * scale + b + crossed
        deep = self.mlp(x0)
        representation = torch.cat([crossed, deep], dim=1)
        linear = self.linear_embedding(x).squeeze(-1).sum(dim=1) + self.linear_bias
        logits = linear + self.main_head(representation).squeeze(1)
        ordinal_logits = self.ordinal_head(representation)
        return logits, ordinal_logits


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def encode_value(mapping, value, fit):
    if value in mapping:
        return mapping[value]
    if fit:
        code = len(mapping) + 1
        mapping[value] = code
        return code
    return 0


def read_csv_data(data_dir):
    user_map = {}
    video_map = {}
    tab_map = {}
    train_user_local = []
    train_video_local = []
    train_tab_local = []
    train_duration = []
    train_y = []
    train_play = []

    with open(Path(data_dir) / "train.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_user_local.append(encode_value(user_map, row["user_id"], True))
            train_video_local.append(encode_value(video_map, row["video_id"], True))
            train_tab_local.append(encode_value(tab_map, row["tab"], True))
            train_duration.append(parse_float(row["duration_ms"], 0.0))
            train_y.append(parse_float(row["long_view"], 0.0))
            train_play.append(parse_float(row["play_time_ms"], 0.0))

    duration_array = np.asarray(train_duration, dtype=np.float32)
    if duration_array.size:
        quantiles = np.quantile(duration_array, np.linspace(0.0, 1.0, 11))
        duration_edges = np.asarray(quantiles[1:-1], dtype=np.float32)
    else:
        duration_edges = np.zeros(9, dtype=np.float32)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])
    train_local = np.column_stack(
        [
            np.asarray(train_user_local, dtype=np.int64),
            np.asarray(train_video_local, dtype=np.int64),
            np.zeros(len(train_y), dtype=np.int64),
            np.asarray(train_tab_local, dtype=np.int64),
            np.searchsorted(duration_edges, duration_array, side="right").astype(np.int64),
        ]
    )
    train_x = train_local + offsets[None, :]

    val_user_local = []
    val_video_local = []
    val_tab_local = []
    val_duration = []
    val_y = []
    val_user_out = []
    val_video_out = []
    val_user_eval = []
    with open(Path(data_dir) / "val.csv", "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_text = row["user_id"]
            video_text = row["video_id"]
            val_user_local.append(encode_value(user_map, user_text, False))
            val_video_local.append(encode_value(video_map, video_text, False))
            val_tab_local.append(encode_value(tab_map, row["tab"], False))
            val_duration.append(parse_float(row["duration_ms"], 0.0))
            val_y.append(parse_float(row["long_view"], 0.0))
            val_user_out.append(user_text)
            val_video_out.append(video_text)
            try:
                val_user_eval.append(int(user_text))
            except ValueError:
                val_user_eval.append(user_text)

    val_duration_array = np.asarray(val_duration, dtype=np.float32)
    val_local = np.column_stack(
        [
            np.asarray(val_user_local, dtype=np.int64),
            np.asarray(val_video_local, dtype=np.int64),
            np.zeros(len(val_y), dtype=np.int64),
            np.asarray(val_tab_local, dtype=np.int64),
            np.searchsorted(duration_edges, val_duration_array, side="right").astype(np.int64),
        ]
    )
    val_x = val_local + offsets[None, :]
    return {
        "train_x": train_x.astype(np.int64, copy=False),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_user": np.asarray(train_user_local, dtype=np.int64),
        "train_play": np.asarray(train_play, dtype=np.float32),
        "train_duration": duration_array,
        "val_x": val_x.astype(np.int64, copy=False),
        "val_y": np.asarray(val_y, dtype=np.float32),
        "val_user_eval": np.asarray(val_user_eval),
        "val_user_out": val_user_out,
        "val_video_out": val_video_out,
        "field_dims": field_dims,
        "fast": False,
    }


def read_npz_data(data_dir):
    with np.load(Path(data_dir) / "train.npz", allow_pickle=False) as train_npz:
        train_x = np.asarray(train_npz["X"], dtype=np.int64)
        train_y = np.asarray(train_npz["y"], dtype=np.float32)
        train_user = np.asarray(train_npz["user"])
        train_play = np.asarray(train_npz["play_time_ms"], dtype=np.float32)
        train_duration = np.asarray(train_npz["duration_ms"], dtype=np.float32)
        field_dims = np.asarray(train_npz["field_dims"], dtype=np.int64).reshape(-1)
    with np.load(Path(data_dir) / "val.npz", allow_pickle=False) as val_npz:
        val_x = np.asarray(val_npz["X"], dtype=np.int64)
        val_y = np.asarray(val_npz["y"], dtype=np.float32)
        val_user = np.asarray(val_npz["user"])
    video_offset = int(field_dims[0])
    val_video = val_x[:, 1].astype(np.int64) - video_offset
    return {
        "train_x": train_x,
        "train_y": train_y,
        "train_user": train_user,
        "train_play": train_play,
        "train_duration": train_duration,
        "val_x": val_x,
        "val_y": val_y,
        "val_user_eval": val_user,
        "val_user_out": val_user,
        "val_video_out": val_video,
        "field_dims": field_dims,
        "fast": True,
    }


def make_user_pairs(users, labels, rng):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    positive_parts = []
    negative_parts = []
    for start, end in zip(starts, ends):
        indices = order[start:end]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if positives.size == 0 or negatives.size == 0:
            continue
        count = max(positives.size, negatives.size)
        positive_parts.append(rng.choice(positives, size=count, replace=positives.size < count))
        negative_parts.append(rng.choice(negatives, size=count, replace=negatives.size < count))
    if not positive_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positive_parts), np.concatenate(negative_parts)


def predict(model, x, device, batch_size):
    model.eval()
    outputs = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            batch_x = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            logits, _ = model(batch_x)
            outputs[start:end] = torch.sigmoid(logits).cpu().numpy()
    return outputs


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fast = (Path(args.data_dir) / "train.npz").is_file() and (Path(args.data_dir) / "val.npz").is_file()
    data = read_npz_data(args.data_dir) if fast else read_csv_data(args.data_dir)

    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    train_x = data["train_x"]
    train_y = data["train_y"]
    train_user = data["train_user"]
    train_play = np.maximum(data["train_play"], 0.0)
    train_duration = np.maximum(data["train_duration"], 1.0)
    denominator = np.maximum(np.minimum(train_duration, 18000.0), 1.0)
    watch_ratio = np.clip(train_play / denominator, 0.0, 2.0).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    pair_positive, pair_negative = make_user_pairs(train_user, train_y, rng)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(data["field_dims"], embed_dim=16, hidden_dim=128, dropout=0.3, cross_layers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    max_epochs = 10
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        max_epochs = min(max_epochs, max(1, int(smoke_epochs)))
    batch_size = 8192 if device.type == "cuda" else 2048
    prediction_batch_size = 32768 if device.type == "cuda" else 8192
    thresholds = torch.tensor([0.25, 0.50, 0.75, 1.00], dtype=torch.float32, device=device)
    best_gauc = -float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        permutation = rng.permutation(len(train_x))
        for start in range(0, len(permutation), batch_size):
            main_indices = permutation[start:start + batch_size]
            current_size = len(main_indices)
            pair_size = max(1, current_size // 2) if pair_positive.size else 0
            if pair_size:
                sampled_pairs = rng.integers(0, pair_positive.size, size=pair_size)
                pos_indices = pair_positive[sampled_pairs]
                neg_indices = pair_negative[sampled_pairs]
                joined_x = np.concatenate(
                    [train_x[main_indices], train_x[pos_indices], train_x[neg_indices]], axis=0
                )
            else:
                joined_x = train_x[main_indices]

            x_tensor = torch.as_tensor(joined_x, dtype=torch.long, device=device)
            y_tensor = torch.as_tensor(train_y[main_indices], dtype=torch.float32, device=device)
            ratio_tensor = torch.as_tensor(watch_ratio[main_indices], dtype=torch.float32, device=device)
            logits, ordinal_logits = model(x_tensor)
            main_logits = logits[:current_size]
            main_ordinal = ordinal_logits[:current_size]
            bce_loss = F.binary_cross_entropy_with_logits(main_logits, y_tensor)
            ordinal_targets = (ratio_tensor[:, None] >= thresholds[None, :]).to(torch.float32)
            ordinal_loss = F.binary_cross_entropy_with_logits(main_ordinal, ordinal_targets)
            if pair_size:
                pos_logits = logits[current_size:current_size + pair_size]
                neg_logits = logits[current_size + pair_size:current_size + 2 * pair_size]
                bpr_loss = F.softplus(-(pos_logits - neg_logits)).mean()
            else:
                bpr_loss = torch.zeros((), dtype=torch.float32, device=device)
            loss = 0.5 * bce_loss + 0.5 * bpr_loss + 0.3 * ordinal_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_scores = predict(model, data["val_x"], device, prediction_batch_size)
        epoch_metrics = evaluate(data["val_user_eval"], data["val_y"], val_scores)
        epoch_gauc = float(epoch_metrics["GAUC"])
        if epoch_gauc > best_gauc + 1.0e-12:
            best_gauc = epoch_gauc
            best_state = clone_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_scores = predict(model, data["val_x"], device, prediction_batch_size)
    final_metrics = evaluate(data["val_user_eval"], data["val_y"], final_scores)

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(
            zip(data["val_user_out"], data["val_video_out"], final_scores)
        ):
            writer.writerow([row_id, user_id, video_id, format(float(score), ".10g")])

    metrics_output = {
        "gauc": float(final_metrics["GAUC"]),
        "ndcg5": float(final_metrics["nDCG@5"]),
        "primary": float(final_metrics["primary"]),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
