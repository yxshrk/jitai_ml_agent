import argparse
import contextlib
import csv
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    torch.use_deterministic_algorithms(True)


def duration_buckets(train_duration, values):
    quantiles = np.quantile(train_duration.astype(np.float64), np.linspace(0.1, 0.9, 9))
    quantiles = np.unique(quantiles)
    return np.searchsorted(quantiles, values, side="right").astype(np.int64), len(quantiles) + 1


def make_encoder(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_values(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def offset_fields(local_x, field_dims):
    offsets = np.zeros(len(field_dims), dtype=np.int64)
    if len(field_dims) > 1:
        offsets[1:] = np.cumsum(np.asarray(field_dims[:-1], dtype=np.int64))
    return local_x.astype(np.int64) + offsets.reshape(1, -1)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_train(path):
        columns = {name: [] for name in ["user_id", "video_id", "author_id", "tab", "duration_ms", "long_view"]}
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            has_author = "author_id" in (reader.fieldnames or [])
            for row in reader:
                columns["user_id"].append(row["user_id"])
                columns["video_id"].append(row["video_id"])
                columns["author_id"].append(row["author_id"] if has_author else "__unknown_author__")
                columns["tab"].append(row["tab"])
                columns["duration_ms"].append(float(row["duration_ms"]))
                columns["long_view"].append(float(row["long_view"]))
        return columns

    def read_val(path):
        columns = {name: [] for name in ["user_id", "video_id", "author_id", "tab", "duration_ms", "long_view"]}
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            has_author = "author_id" in (reader.fieldnames or [])
            for row in reader:
                columns["user_id"].append(row["user_id"])
                columns["video_id"].append(row["video_id"])
                columns["author_id"].append(row["author_id"] if has_author else "__unknown_author__")
                columns["tab"].append(row["tab"])
                columns["duration_ms"].append(float(row["duration_ms"]))
                columns["long_view"].append(float(row["long_view"]))
        return columns

    train = read_train(train_path)
    val = read_val(val_path)
    user_map = make_encoder(train["user_id"])
    video_map = make_encoder(train["video_id"])
    author_map = make_encoder(train["author_id"])
    tab_map = make_encoder(train["tab"])

    train_duration = np.asarray(train["duration_ms"], dtype=np.float64)
    val_duration = np.asarray(val["duration_ms"], dtype=np.float64)
    train_dur, dur_dim_without_unknown = duration_buckets(train_duration, train_duration)
    quantiles = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    val_dur = np.searchsorted(quantiles, val_duration, side="right").astype(np.int64)

    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        dur_dim_without_unknown,
    ], dtype=np.int64)

    train_local = np.column_stack([
        encode_values(train["user_id"], user_map),
        encode_values(train["video_id"], video_map),
        encode_values(train["author_id"], author_map),
        encode_values(train["tab"], tab_map),
        train_dur,
    ])
    val_local = np.column_stack([
        encode_values(val["user_id"], user_map),
        encode_values(val["video_id"], video_map),
        encode_values(val["author_id"], author_map),
        encode_values(val["tab"], tab_map),
        val_dur,
    ])

    return {
        "train_x": offset_fields(train_local, field_dims),
        "train_y": np.asarray(train["long_view"], dtype=np.float32),
        "train_user": np.asarray(train["user_id"], dtype=object),
        "val_x": offset_fields(val_local, field_dims),
        "val_y": np.asarray(val["long_view"], dtype=np.float32),
        "val_user": np.asarray(val["user_id"], dtype=object),
        "val_video": np.asarray(val["video_id"], dtype=object),
        "field_dims": field_dims,
        "npz": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=False) as train, np.load(val_npz, allow_pickle=False) as val:
            field_dims = np.asarray(train["field_dims"] if "field_dims" in train else val["field_dims"], dtype=np.int64)
            train_x = np.asarray(train["X"], dtype=np.int64)
            val_x = np.asarray(val["X"], dtype=np.int64)
            video_offset = int(field_dims[0])
            val_video = val_x[:, 1].astype(np.int64) - video_offset
            return {
                "train_x": train_x,
                "train_y": np.asarray(train["y"], dtype=np.float32),
                "train_user": np.asarray(train["user"]),
                "val_x": val_x,
                "val_y": np.asarray(val["y"], dtype=np.float32),
                "val_user": np.asarray(val["user"]),
                "val_video": val_video,
                "field_dims": field_dims,
                "npz": True,
            }
    return load_csv_data(data_dir)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embedding_dim=16, hidden_dim=128, dropout=0.2):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = len(field_dims) * embedding_dim
        self.embedding = nn.Embedding(total, embedding_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.cross_out = nn.Linear(input_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)
        for weight in self.cross_w:
            nn.init.normal_(weight, std=0.01)

    def forward(self, x):
        emb = self.embedding(x).reshape(x.shape[0], -1)
        cross = emb
        for weight, bias in zip(self.cross_w, self.cross_b):
            scale = torch.sum(cross * weight, dim=1, keepdim=True)
            cross = emb * scale + bias + cross
        linear = self.linear(x).sum(dim=1).squeeze(1)
        return linear + self.cross_out(cross).squeeze(1) + self.mlp(emb).squeeze(1) + self.bias


def user_spans(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return order, np.asarray([0], dtype=np.int64)
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)])).astype(np.int64)
    return order, bounds


def make_top5_groups(order, bounds, rng):
    group_count = int(sum((int(bounds[i + 1] - bounds[i]) + 4) // 5 for i in range(len(bounds) - 1)))
    groups = np.full((group_count, 5), -1, dtype=np.int64)
    row = 0
    for i in range(len(bounds) - 1):
        indices = order[bounds[i]:bounds[i + 1]].copy()
        rng.shuffle(indices)
        for start in range(0, len(indices), 5):
            chunk = indices[start:start + 5]
            groups[row, :len(chunk)] = chunk
            row += 1
    rng.shuffle(groups)
    return groups


def delta_ndcg_bpr(scores, labels, valid):
    rank_scores = scores.detach().masked_fill(~valid, -1.0e9)
    ranked_positions = torch.argsort(rank_scores, dim=1, descending=True)
    ranks = torch.empty_like(ranked_positions)
    base = torch.arange(scores.shape[1], device=scores.device).view(1, -1).expand_as(ranked_positions)
    ranks.scatter_(1, ranked_positions, base)
    discounts = 1.0 / torch.log2(ranks.to(torch.float32) + 2.0)

    positives_per_group = ((labels > 0.5) & valid).sum(dim=1).clamp(max=5)
    ideal_discount = torch.tensor(
        [1.0 / np.log2(i + 2.0) for i in range(5)],
        dtype=scores.dtype,
        device=scores.device,
    )
    idcg = torch.zeros(scores.shape[0], dtype=scores.dtype, device=scores.device)
    for count in range(1, 6):
        idcg = torch.where(positives_per_group == count, ideal_discount[:count].sum(), idcg)

    positive = (labels > 0.5) & valid
    negative = (labels <= 0.5) & valid
    pair_mask = positive.unsqueeze(2) & negative.unsqueeze(1)
    deltas = torch.abs(discounts.unsqueeze(2) - discounts.unsqueeze(1))
    deltas = deltas / idcg.clamp_min(1.0e-8).view(-1, 1, 1)
    weights = deltas * pair_mask.to(scores.dtype)
    denominator = weights.sum()
    if denominator.detach().item() <= 0.0:
        return scores.sum() * 0.0
    differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    return (weights * F.softplus(-differences)).sum() / denominator


def predict(model, x, device, batch_size=8192):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def get_evaluator(use_npz):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def train_model(data, seed, epochs, evaluator):
    device = torch.device("cpu")
    model = DCNLite(data["field_dims"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    order, bounds = user_spans(data["train_user"])
    rng = np.random.default_rng(seed)
    best_primary = -np.inf
    best_state = None
    stale = 0
    group_batch_size = 512
    x = data["train_x"]
    y = data["train_y"]

    for _ in range(epochs):
        groups = make_top5_groups(order, bounds, rng)
        model.train()
        for start in range(0, len(groups), group_batch_size):
            group = groups[start:start + group_batch_size]
            valid_np = group >= 0
            safe_group = np.where(valid_np, group, 0)
            xb = torch.as_tensor(x[safe_group], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[safe_group], dtype=torch.float32, device=device)
            valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
            flat_scores = model(xb.reshape(-1, xb.shape[-1]))
            scores = flat_scores.reshape(xb.shape[0], 5)
            point_loss = F.binary_cross_entropy_with_logits(scores[valid], yb[valid])
            rank_loss = delta_ndcg_bpr(scores, yb, valid)
            accessed_l2 = model.embedding(xb[valid]).pow(2).mean()
            loss = 0.5 * point_loss + 0.5 * rank_loss + 1.0e-6 * accessed_l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        val_scores = predict(model, data["val_x"], device)
        metrics = evaluator(data["val_user"], data["val_y"], val_scores)
        primary = float(metrics["primary"])
        if primary > best_primary:
            best_primary = primary
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def write_predictions(path, users, videos, scores):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(users, videos, scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user,
                             video.item() if isinstance(video, np.generic) else video,
                             format(float(score), ".10g")])


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    data = load_data(args.data_dir)
    evaluator = get_evaluator(data["npz"])
    model = train_model(data, args.seed, epochs, evaluator)
    scores = predict(model, data["val_x"], torch.device("cpu"))
    metrics = evaluator(data["val_user"], data["val_y"], scores)

    write_predictions(
        os.path.join(args.out_dir, "predictions.csv"),
        data["val_user"],
        data["val_video"],
        scores,
    )
    output_metrics = {
        "gauc": float(metrics["GAUC"]),
        "ndcg5": float(metrics["nDCG@5"]),
        "primary": float(metrics["primary"]),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(output_metrics, handle, separators=(",", ":"))


if __name__ == "__main__":
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        main()
