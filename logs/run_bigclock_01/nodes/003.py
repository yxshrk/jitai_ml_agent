import argparse
import csv
import datetime
import json
import math
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_npz_split(path):
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    return out


def make_date_age(date_values):
    values = np.asarray(date_values).reshape(-1)
    if len(values) == 0:
        return np.zeros(0, dtype=np.float32)
    normalized = []
    for value in values:
        text = str(value.decode() if isinstance(value, bytes) else value)
        text = text.split(".")[0].replace("-", "").replace("/", "")
        try:
            if len(text) >= 8:
                day = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
            else:
                day = int(float(text))
        except Exception:
            day = 0
        normalized.append(day)
    arr = np.asarray(normalized, dtype=np.int64)
    positive = arr[arr > 0]
    if len(positive) == 0:
        return np.zeros(len(arr), dtype=np.float32)
    maximum = int(positive.max())
    arr = np.where(arr > 0, arr, maximum)
    return (maximum - arr).astype(np.float32)


def encode_csv(train_path, val_path):
    wanted = ["user_id", "video_id", "author_id", "tab", "duration_ms", "date", "long_view"]

    def read_file(path, training):
        result = {key: [] for key in wanted}
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            names = set(reader.fieldnames or [])
            for row in reader:
                result["user_id"].append(row.get("user_id", "0"))
                result["video_id"].append(row.get("video_id", "0"))
                result["author_id"].append(row.get("author_id", "__missing_author__") if "author_id" in names else "__missing_author__")
                result["tab"].append(row.get("tab", "0"))
                result["duration_ms"].append(float(row.get("duration_ms", "0") or 0.0))
                result["date"].append(row.get("date", "0"))
                result["long_view"].append(float(row.get("long_view", "0") or 0.0) if training or "long_view" in names else 0.0)
        return result

    train = read_file(train_path, True)
    val = read_file(val_path, False)
    train_duration = np.asarray(train["duration_ms"], dtype=np.float64)
    if len(train_duration):
        edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)
    train_bucket = np.searchsorted(edges, train_duration, side="right").astype(str).tolist()
    val_bucket = np.searchsorted(edges, np.asarray(val["duration_ms"], dtype=np.float64), side="right").astype(str).tolist()
    train_fields = [train["user_id"], train["video_id"], train["author_id"], train["tab"], train_bucket]
    val_fields = [val["user_id"], val["video_id"], val["author_id"], val["tab"], val_bucket]
    train_encoded = []
    val_encoded = []
    dims = []
    offset = 0
    for train_column, val_column in zip(train_fields, val_fields):
        mapping = {}
        for value in train_column:
            if value not in mapping:
                mapping[value] = len(mapping) + 1
        train_codes = np.asarray([mapping[value] for value in train_column], dtype=np.int64)
        val_codes = np.asarray([mapping.get(value, 0) for value in val_column], dtype=np.int64)
        dim = len(mapping) + 1
        train_encoded.append(train_codes + offset)
        val_encoded.append(val_codes + offset)
        dims.append(dim)
        offset += dim
    return {
        "train_X": np.stack(train_encoded, axis=1).astype(np.int64),
        "val_X": np.stack(val_encoded, axis=1).astype(np.int64),
        "train_y": np.asarray(train["long_view"], dtype=np.float32),
        "val_y": np.asarray(val["long_view"], dtype=np.float32),
        "train_user": np.asarray(train["user_id"]),
        "val_user": np.asarray(val["user_id"]),
        "val_video": np.asarray(val["video_id"]),
        "train_date": np.asarray(train["date"]),
        "field_dims": np.asarray(dims, dtype=np.int64),
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = load_npz_split(train_npz)
        va = load_npz_split(val_npz)
        field_dims = np.asarray(tr.get("field_dims", va.get("field_dims")), dtype=np.int64)
        val_x = np.asarray(va["X"], dtype=np.int64)
        video_offset = int(field_dims[0])
        if "video" in va:
            val_video = np.asarray(va["video"])
        elif "video_id" in va:
            val_video = np.asarray(va["video_id"])
        else:
            val_video = val_x[:, 1] - video_offset
        return {
            "train_X": np.asarray(tr["X"], dtype=np.int64),
            "val_X": val_x,
            "train_y": np.asarray(tr["y"], dtype=np.float32),
            "val_y": np.asarray(va["y"], dtype=np.float32),
            "train_user": np.asarray(tr["user"]),
            "val_user": np.asarray(va["user"]),
            "val_video": val_video,
            "train_date": np.asarray(tr.get("date", np.zeros(len(tr["y"]), dtype=np.int64))),
            "field_dims": field_dims,
        }
    return encode_csv(os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "val.csv"))


class DCNLite(nn.Module):
    def __init__(self, field_dims, rank, dropout):
        super().__init__()
        total = int(np.sum(field_dims))
        input_dim = int(len(field_dims) * rank)
        self.embedding = nn.Embedding(total, rank)
        self.linear_embedding = nn.Embedding(total, 1)
        self.cross_weight = nn.ParameterList([nn.Parameter(torch.empty(input_dim)) for _ in range(2)])
        self.cross_bias = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(2)])
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(input_dim + 64, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear_embedding.weight)
        for weight in self.cross_weight:
            nn.init.normal_(weight, std=0.01)

    def forward(self, x):
        embedded = self.embedding(x).reshape(x.shape[0], -1)
        crossed = embedded
        for weight, bias in zip(self.cross_weight, self.cross_bias):
            scale = torch.sum(crossed * weight, dim=1, keepdim=True)
            crossed = embedded * scale + bias + crossed
        deep = self.deep(embedded)
        first_order = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return self.output(torch.cat([crossed, deep], dim=1)).squeeze(1) + first_order + self.global_bias


def build_pairs(users, labels, seed, limit=None):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        pos = indices[labels[indices] > 0.5]
        neg = indices[labels[indices] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos = np.concatenate(positives).astype(np.int64)
    neg = np.concatenate(negatives).astype(np.int64)
    if limit is not None and len(pos) > limit:
        chosen = rng.choice(len(pos), size=limit, replace=False)
        pos = pos[chosen]
        neg = neg[chosen]
    return pos, neg


def official_metrics(user, labels, scores, fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(user, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, x, device, batch_size=16384):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.long, device=device)
            pieces.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(pieces).astype(np.float64) if pieces else np.zeros(0, dtype=np.float64)


def train_variant(data, config, seed, epochs, probe_limit, device, fast_path):
    seed_all(seed)
    model = DCNLite(data["field_dims"], 16, config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config["lr_gamma"])
    n = len(data["train_y"])
    rng = np.random.default_rng(seed + 17)
    if probe_limit is not None and n > probe_limit:
        train_indices = np.sort(rng.choice(n, size=probe_limit, replace=False)).astype(np.int64)
    else:
        train_indices = np.arange(n, dtype=np.int64)
    ages = make_date_age(data["train_date"])
    recency = np.power(0.5, ages / float(config["half_life"])).astype(np.float32)
    pair_limit = min(400000, max(10000, len(train_indices)))
    pair_pos, pair_neg = build_pairs(data["train_user"], data["train_y"], seed + 31, pair_limit)
    if probe_limit is not None and len(pair_pos) > probe_limit:
        chosen = rng.choice(len(pair_pos), size=probe_limit, replace=False)
        pair_pos = pair_pos[chosen]
        pair_neg = pair_neg[chosen]
    batch_size = 4096
    best_gauc = -1.0
    best_primary = -1.0
    best_scores = None
    best_state = None
    epoch_records = []
    patience = 3
    stale = 0
    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_indices)
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.zeros(0, dtype=np.int64)
        steps = max(1, math.ceil(len(shuffled) / batch_size))
        for step in range(steps):
            bidx = shuffled[step * batch_size:(step + 1) * batch_size]
            if len(bidx) == 0:
                continue
            xb = torch.as_tensor(data["train_X"][bidx], dtype=torch.long, device=device)
            yb = torch.as_tensor(data["train_y"][bidx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[bidx], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).sum() / wb.sum().clamp_min(1e-6)
            if len(pair_order):
                start = (step * batch_size) % len(pair_order)
                take = pair_order[start:min(start + len(bidx), len(pair_order))]
                if len(take) < len(bidx):
                    take = np.concatenate([take, pair_order[:len(bidx) - len(take)]])
                pi = pair_pos[take]
                ni = pair_neg[take]
                xp = torch.as_tensor(data["train_X"][pi], dtype=torch.long, device=device)
                xn = torch.as_tensor(data["train_X"][ni], dtype=torch.long, device=device)
                pair_weight = torch.as_tensor(0.5 * (recency[pi] + recency[ni]), dtype=torch.float32, device=device)
                pair_raw = F.softplus(-(model(xp) - model(xn)))
                pair_loss = (pair_raw * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, data["val_X"], device)
        metrics = official_metrics(data["val_user"], data["val_y"], scores, fast_path)
        epoch_records.append({"epoch": epoch + 1, **metrics})
        if metrics["gauc"] > best_gauc + 1e-10:
            best_gauc = metrics["gauc"]
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_scores, best_primary, best_gauc, epoch_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)
    data = load_data(args.data_dir)
    fast_path = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = max(1, int(smoke)) if smoke is not None else None
    probe_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    final_epochs = min(8, smoke_cap) if smoke_cap is not None else 8
    configs = [
        {"dropout": 0.20, "weight_decay": 3e-4, "lr_gamma": 0.50, "half_life": 7.0},
        {"dropout": 0.30, "weight_decay": 3e-4, "lr_gamma": 0.50, "half_life": 7.0},
        {"dropout": 0.25, "weight_decay": 1e-3, "lr_gamma": 0.50, "half_life": 7.0},
        {"dropout": 0.30, "weight_decay": 1e-3, "lr_gamma": 0.60, "half_life": 14.0},
    ]
    if smoke_cap is not None:
        configs = configs[:1]
    history = []
    best_config = None
    best_probe_primary = -1.0
    probe_limit = min(len(data["train_y"]), 300000)
    for index, config in enumerate(configs):
        _, _, primary, gauc, epochs_record = train_variant(
            data, config, args.seed + 101 * index, probe_epochs, probe_limit, device, fast_path
        )
        history.append({"phase": "probe", "config": config, "best_primary": float(primary), "best_gauc": float(gauc), "epochs": epochs_record})
        if primary > best_probe_primary:
            best_probe_primary = primary
            best_config = config
    model, scores, _, _, final_epoch_records = train_variant(
        data, best_config, args.seed + 1009, final_epochs, None, device, fast_path
    )
    if scores is None:
        scores = predict(model, data["val_X"], device)
    metrics = official_metrics(data["val_user"], data["val_y"], scores, fast_path)
    history.append({"phase": "final", "config": best_config, "best_primary": metrics["primary"], "best_gauc": metrics["gauc"], "epochs": final_epoch_records})
    prediction_path = os.path.join(args.out_dir, "predictions.csv")
    with open(prediction_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for index, (user, video, score) in enumerate(zip(data["val_user"], data["val_video"], scores)):
            writer.writerow([index, user.item() if hasattr(user, "item") else user, video.item() if hasattr(video, "item") else video, "{:.10f}".format(float(score))])
    output_metrics = {
        "gauc": metrics["gauc"],
        "ndcg5": metrics["ndcg5"],
        "primary": metrics["primary"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(output_metrics, handle, sort_keys=True)


if __name__ == "__main__":
    main()
