import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    train = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    val = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    x_train = np.asarray(train["X"], dtype=np.int64)
    y_train = np.asarray(train["y"], dtype=np.float32)
    x_val = np.asarray(val["X"], dtype=np.int64)
    y_val = np.asarray(val["y"], dtype=np.float32)
    user_train = np.asarray(train["user"])
    user_val = np.asarray(val["user"])
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    dates = np.asarray(train["date"]) if "date" in train.files else None
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video_val = x_val[:, 1] - offsets[1]
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "user_train": user_train,
        "user_val": user_val,
        "video_val": video_val,
        "field_dims": field_dims,
        "dates": dates,
        "npz": True,
    }


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def load_csv(data_dir):
    train_path = Path(data_dir) / "train.csv"
    val_path = Path(data_dir) / "val.csv"
    with train_path.open("r", newline="", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with val_path.open("r", newline="", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))

    durations = np.asarray([safe_int(r.get("duration_ms", 0)) for r in train_rows], dtype=np.float64)
    if len(durations):
        quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    else:
        quantiles = np.zeros(9, dtype=np.float64)

    field_names = ["user_id", "video_id", "author_proxy", "tab", "dur_bucket"]
    maps = {name: {} for name in field_names}

    def raw_fields(row):
        video = row.get("video_id", "")
        duration = safe_int(row.get("duration_ms", 0))
        bucket = int(np.searchsorted(quantiles, duration, side="right"))
        return [
            row.get("user_id", ""),
            video,
            video,
            row.get("tab", ""),
            str(bucket),
        ]

    for row in train_rows:
        values = raw_fields(row)
        for name, value in zip(field_names, values):
            mapping = maps[name]
            if value not in mapping:
                mapping[value] = len(mapping) + 1

    field_dims = np.asarray([len(maps[name]) + 1 for name in field_names], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, (name, value) in enumerate(zip(field_names, raw_fields(row))):
                x[i, j] = maps[name].get(value, 0) + offsets[j]
        return x

    x_train = encode(train_rows)
    x_val = encode(val_rows)
    y_train = np.asarray([float(r.get("long_view", 0.0)) for r in train_rows], dtype=np.float32)
    y_val = np.asarray([float(r.get("long_view", 0.0)) for r in val_rows], dtype=np.float32)
    user_train = np.asarray([r.get("user_id", "") for r in train_rows])
    user_val = np.asarray([r.get("user_id", "") for r in val_rows])
    video_val = np.asarray([r.get("video_id", "") for r in val_rows])
    dates = np.asarray([safe_int(r.get("date", 0)) for r in train_rows], dtype=np.int64)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "user_train": user_train,
        "user_val": user_val,
        "video_val": video_val,
        "field_dims": field_dims,
        "dates": dates,
        "npz": False,
    }


def recency_weights(dates):
    if dates is None or len(dates) == 0:
        return None
    try:
        d = np.asarray(dates).astype(np.int64)
    except Exception:
        return None
    valid = d > 0
    if not np.any(valid):
        return None
    parsed = np.zeros(len(d), dtype=np.int64)
    import datetime
    good_indices = np.flatnonzero(valid)
    ordinals = []
    kept = []
    for idx in good_indices:
        text = str(int(d[idx]))
        try:
            day = datetime.datetime.strptime(text, "%Y%m%d").date().toordinal()
            ordinals.append(day)
            kept.append(idx)
        except Exception:
            pass
    if not ordinals:
        return None
    parsed[np.asarray(kept, dtype=np.int64)] = np.asarray(ordinals, dtype=np.int64)
    maximum = int(np.max(parsed))
    weights = np.ones(len(d), dtype=np.float32)
    usable = parsed > 0
    weights[usable] = np.exp2((parsed[usable] - maximum) / 7.0).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-6)
    return weights


def build_pairs(users, labels, seed):
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        chosen = neg[rng.integers(0, len(neg), size=len(pos))]
        positives.append(pos)
        negatives.append(chosen)
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x0, x):
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.3):
        super().__init__()
        total = int(np.sum(field_dims))
        fields = len(field_dims)
        dim = fields * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear = nn.Embedding(total, 1)
        self.cross1 = CrossLayer(dim)
        self.cross2 = CrossLayer(dim)
        self.deep = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_out = nn.Linear(dim, 1)
        self.deep_out = nn.Linear(64, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        embedded = self.embedding(x).flatten(1)
        crossed = self.cross1(embedded, embedded)
        crossed = self.cross2(embedded, crossed)
        linear = self.linear(x).sum(dim=1).squeeze(1)
        return linear + self.cross_out(crossed).squeeze(1) + self.deep_out(self.deep(embedded)).squeeze(1) + self.bias


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self, model):
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for name, value in self.shadow.items():
            state[name] = value.detach().cpu().clone()
        return state


def predict(model, x, device, batch_size=16384):
    model.eval()
    output = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:stop], dtype=torch.long, device=device)
            output[start:stop] = torch.sigmoid(model(xb)).cpu().numpy()
    return output


def metric_values(npz_mode, users, labels, scores):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    gauc = float(result.get("GAUC", result.get("gauc")))
    ndcg = float(result.get("nDCG@5", result.get("ndcg5")))
    primary = float(result.get("primary", 0.5 * (gauc + ndcg)))
    return {"gauc": gauc, "ndcg5": ndcg, "primary": primary}


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    if (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists():
        data = load_npz(data_dir)
    else:
        data = load_csv(data_dir)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    weights = recency_weights(data["dates"])
    if weights is None:
        weights = np.ones(len(y_train), dtype=np.float32)

    pair_pos, pair_neg = build_pairs(data["user_train"], y_train, args.seed + 17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(data["field_dims"], embed_dim=16, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    ema = EMA(model, decay=0.995)

    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    batch_size = 4096
    rng = np.random.default_rng(args.seed)
    best_gauc = -math.inf
    best_state = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_train))
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0

        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            xb = torch.as_tensor(x_train[indices], dtype=torch.long, device=device)
            yb = torch.as_tensor(y_train[indices], dtype=torch.float32, device=device)
            wb = torch.as_tensor(weights[indices], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = torch.sum(point_loss * wb) / torch.clamp(torch.sum(wb), min=1e-6)

            if len(pair_order):
                count = len(indices)
                if pair_cursor + count > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                selected = pair_order[pair_cursor:pair_cursor + count]
                pair_cursor += len(selected)
                pidx = pair_pos[selected]
                nidx = pair_neg[selected]
                xp = torch.as_tensor(x_train[pidx], dtype=torch.long, device=device)
                xn = torch.as_tensor(x_train[nidx], dtype=torch.long, device=device)
                pw = torch.as_tensor((weights[pidx] + weights[nidx]) * 0.5, dtype=torch.float32, device=device)
                pair_loss_each = F.softplus(-(model(xp) - model(xn)))
                pair_loss = torch.sum(pair_loss_each * pw) / torch.clamp(torch.sum(pw), min=1e-6)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            ema.update(model)

        scheduler.step()
        raw_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(model))
        val_scores = predict(model, x_val, device)
        epoch_metrics = metric_values(data["npz"], data["user_val"], y_val, val_scores)
        if epoch_metrics["gauc"] > best_gauc + 1e-12:
            best_gauc = epoch_metrics["gauc"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        model.load_state_dict(raw_state)
        if stale >= 2:
            break

    if best_state is None:
        best_state = ema.state_dict(model)
    model.load_state_dict(best_state)
    final_scores = predict(model, x_val, device)
    metrics = metric_values(data["npz"], data["user_val"], y_val, final_scores)

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user_id, video_id, score) in enumerate(zip(data["user_val"], data["video_val"], final_scores)):
            if isinstance(user_id, np.generic):
                user_id = user_id.item()
            if isinstance(video_id, np.generic):
                video_id = video_id.item()
            writer.writerow([i, user_id, video_id, float(score)])

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
