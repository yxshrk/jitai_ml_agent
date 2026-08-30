import argparse
import csv
import datetime
import json
import os
import random
import warnings

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def date_ordinal(value):
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 8:
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    try:
        return int(float(s))
    except ValueError:
        return 0


def minute_of_day(value):
    try:
        x = int(float(value))
    except (TypeError, ValueError):
        return 0
    hour = max(0, min(23, x // 100))
    minute = max(0, min(59, x % 100))
    return hour * 60 + minute


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    train = {
        "X": np.asarray(tr["X"], dtype=np.int64),
        "y": np.asarray(tr["y"], dtype=np.float32),
        "user": np.asarray(tr["user"]),
        "hourmin": np.asarray(tr["hourmin"]),
        "date": np.asarray(tr["date"]),
        "duration": np.asarray(tr["duration_ms"], dtype=np.float32),
    }
    valid = {
        "X": np.asarray(va["X"], dtype=np.int64),
        "y": np.asarray(va["y"], dtype=np.float32),
        "user": np.asarray(va["user"]),
        "hourmin": np.asarray(va["hourmin"]),
        "date": np.asarray(va["date"]),
        "duration": np.asarray(va["duration_ms"], dtype=np.float32),
    }
    dims = np.asarray(tr["field_dims"], dtype=np.int64)
    video_offset = int(dims[0])
    valid["video"] = valid["X"][:, 1] - video_offset
    return train, valid, dims


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__unknown_author__"),
                "tab": r["tab"],
                "hourmin": r["hourmin"],
                "date": r["date"],
                "duration_ms": float(r["duration_ms"] or 0.0),
                "long_view": float(r["long_view"]) if training or "long_view" in r else 0.0,
            }
            rows.append(item)
    return rows


def make_mapping(values):
    mapping = {"__UNK__": 0}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def load_csv(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    dims = np.asarray([len(user_map), len(video_map), len(author_map), len(tab_map), 10], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def encode(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            X[i, 0] = user_map.get(r["user_id"], 0) + offsets[0]
            X[i, 1] = video_map.get(r["video_id"], 0) + offsets[1]
            X[i, 2] = author_map.get(r["author_id"], 0) + offsets[2]
            X[i, 3] = tab_map.get(r["tab"], 0) + offsets[3]
            X[i, 4] = int(np.searchsorted(quantiles, r["duration_ms"], side="right")) + offsets[4]
        out = {
            "X": X,
            "y": np.asarray([r["long_view"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user_id"] for r in rows]),
            "video": np.asarray([r["video_id"] for r in rows]),
            "hourmin": np.asarray([r["hourmin"] for r in rows]),
            "date": np.asarray([r["date"] for r in rows]),
            "duration": np.asarray([r["duration_ms"] for r in rows], dtype=np.float32),
        }
        return out

    return encode(train_rows), encode(val_rows), dims


def session_context(data, tab_offset, state):
    n = len(data["y"])
    context = np.empty((n, 6), dtype=np.int64)
    for i in range(n):
        user = data["user"][i].item() if isinstance(data["user"][i], np.generic) else data["user"][i]
        day = date_ordinal(data["date"][i])
        minute = minute_of_day(data["hourmin"][i])
        timestamp = day * 1440 + minute
        hour = minute // 60
        half_hour = minute // 30
        try:
            dow = datetime.date.fromordinal(day).weekday() if day > 100000 else day % 7
        except ValueError:
            dow = day % 7
        tab = max(0, int(data["X"][i, 3]) - tab_offset)
        previous = state.get(user)
        if previous is None or timestamp < previous[0]:
            gap_bucket = 0
            position = 0
        else:
            gap = timestamp - previous[0]
            if gap > 120:
                gap_bucket = 6
                position = 0
            else:
                if gap <= 1:
                    gap_bucket = 1
                elif gap <= 5:
                    gap_bucket = 2
                elif gap <= 15:
                    gap_bucket = 3
                elif gap <= 30:
                    gap_bucket = 4
                else:
                    gap_bucket = 5
                position = previous[1] + 1
        if position == 0:
            position_bucket = 0
        elif position <= 4:
            position_bucket = position
        elif position <= 7:
            position_bucket = 5
        elif position <= 15:
            position_bucket = 6
        else:
            position_bucket = 7
        context[i] = (hour, dow, half_hour, hour * 7 + dow, (tab % 64) * 24 + hour, gap_bucket * 8 + position_bucket)
        state[user] = (timestamp, position)
    return context


def add_context(train, valid, dims):
    tab_offset = int(np.sum(dims[:3]))
    state = {}
    tr_ctx = session_context(train, tab_offset, state)
    va_ctx = session_context(valid, tab_offset, state)
    tab_dim = min(int(dims[3]), 64)
    extra_dims = np.asarray([24, 7, 48, 168, tab_dim * 24, 56], dtype=np.int64)
    starts = int(np.sum(dims)) + np.concatenate(([0], np.cumsum(extra_dims)[:-1]))
    train["X"] = np.concatenate([train["X"], tr_ctx + starts], axis=1)
    valid["X"] = np.concatenate([valid["X"], va_ctx + starts], axis=1)
    return np.concatenate([dims, extra_dims])


class ContextDCN(nn.Module):
    def __init__(self, field_dims, k=24, dropout=0.21):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * k
        self.embedding = nn.Embedding(total, k)
        self.linear_embedding = nn.Embedding(total, 1)
        self.cross_weight = nn.Linear(width, width)
        self.cross_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(width + 64, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)

    def forward(self, x):
        emb = self.embedding(x).flatten(1)
        crossed = emb * self.cross_weight(emb) + emb
        crossed = self.cross_norm(crossed)
        deep = self.mlp(emb)
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return self.output(torch.cat([crossed, deep], dim=1)).squeeze(1) + linear


def official_metrics(users, labels, scores, npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(users, labels, scores)
    return {
        "gauc": float(result["GAUC"]),
        "ndcg5": float(result["nDCG@5"]),
        "primary": float(result["primary"]),
    }


def predict(model, X, device, batch_size=16384):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
            parts.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64) if parts else np.empty(0, dtype=np.float64)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    npz_mode = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_mode:
        train, valid, dims = load_npz(args.data_dir)
    else:
        train, valid, dims = load_csv(args.data_dir)
    dims = add_context(train, valid, dims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextDCN(dims, k=24, dropout=0.21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00168, weight_decay=0.000037)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    epochs = 2
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    train_days = np.asarray([date_ordinal(x) for x in train["date"]], dtype=np.float32)
    newest = float(np.max(train_days)) if len(train_days) else 0.0
    recency = np.exp2((train_days - newest) / 7.0).astype(np.float32)
    recency /= max(float(np.mean(recency)), 1e-8)

    rng = np.random.RandomState(args.seed)
    batch_size = 4096
    best_gauc = -1.0
    best_state = None
    for epoch in range(epochs):
        order = rng.permutation(len(train["y"]))
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(train["X"][idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(train["y"][idx], dtype=torch.float32, device=device)
            wb = torch.as_tensor(recency[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = (criterion(logits, yb) * wb).mean()
            loss.backward()
            optimizer.step()
        scores = predict(model, valid["X"], device)
        metrics = official_metrics(valid["user"], valid["y"], scores, npz_mode)
        if metrics["gauc"] > best_gauc:
            best_gauc = metrics["gauc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = predict(model, valid["X"], device)
    metrics = official_metrics(valid["user"], valid["y"], scores, npz_mode)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            user = valid["user"][i].item() if isinstance(valid["user"][i], np.generic) else valid["user"][i]
            video = valid["video"][i].item() if isinstance(valid["video"][i], np.generic) else valid["video"][i]
            writer.writerow([i, user, video, float(score)])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
