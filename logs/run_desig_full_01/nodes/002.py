import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def date_ordinals(values):
    vals = np.asarray(values).astype(np.int64)
    unique, inverse = np.unique(vals, return_inverse=True)
    ords = np.empty(len(unique), dtype=np.int64)
    for i, value in enumerate(unique):
        text = str(int(value))
        if len(text) == 8:
            try:
                ords[i] = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
                continue
            except ValueError:
                pass
        ords[i] = int(value)
    return ords[inverse]


def time_context(hourmin, dates):
    hm = np.asarray(hourmin).astype(np.int64)
    hour = np.clip(hm // 100, 0, 23).astype(np.int64)
    minute = np.clip(hm % 100, 0, 59).astype(np.int64)
    day_number = date_ordinals(dates)
    weekday = np.mod(day_number, 7).astype(np.int64)
    timestamp = day_number * 1440 + hour * 60 + minute
    return hour, weekday, timestamp


def add_session_features(X_train, X_val, train_users, val_users,
                         train_hourmin, val_hourmin, train_dates, val_dates,
                         base_total_dim):
    tr_hour, tr_weekday, tr_ts = time_context(train_hourmin, train_dates)
    va_hour, va_weekday, va_ts = time_context(val_hourmin, val_dates)
    tr_gap = np.empty(len(X_train), dtype=np.int64)
    va_gap = np.empty(len(X_val), dtype=np.int64)
    tr_pos = np.empty(len(X_train), dtype=np.int64)
    va_pos = np.empty(len(X_val), dtype=np.int64)
    state = {}

    def scan(users, timestamps, gap_out, pos_out):
        for i in range(len(users)):
            key = users[i].item() if isinstance(users[i], np.generic) else users[i]
            now = int(timestamps[i])
            previous = state.get(key)
            if previous is None:
                gap_category = 0
                position = 0
            else:
                gap = now - previous[0]
                if gap < 0 or gap > 30:
                    gap_category = 6
                    position = 0
                else:
                    position = previous[1] + 1
                    if gap <= 0:
                        gap_category = 1
                    elif gap <= 1:
                        gap_category = 2
                    elif gap <= 5:
                        gap_category = 3
                    elif gap <= 15:
                        gap_category = 4
                    else:
                        gap_category = 5
            if position == 0:
                position_category = 0
            elif position == 1:
                position_category = 1
            elif position <= 3:
                position_category = 2
            elif position <= 7:
                position_category = 3
            else:
                position_category = 4
            gap_out[i] = gap_category
            pos_out[i] = position_category
            state[key] = (now, position)

    scan(np.asarray(train_users), tr_ts, tr_gap, tr_pos)
    scan(np.asarray(val_users), va_ts, va_gap, va_pos)

    tr_hour_pos = tr_hour * 5 + tr_pos
    va_hour_pos = va_hour * 5 + va_pos
    tr_weekday_gap = tr_weekday * 7 + tr_gap
    va_weekday_gap = va_weekday * 7 + va_gap

    dimensions = [24, 7, 7, 5, 120, 49]
    tr_fields = [tr_hour, tr_weekday, tr_gap, tr_pos, tr_hour_pos, tr_weekday_gap]
    va_fields = [va_hour, va_weekday, va_gap, va_pos, va_hour_pos, va_weekday_gap]
    offset = int(base_total_dim)
    tr_added = []
    va_added = []
    for dim, tr_field, va_field in zip(dimensions, tr_fields, va_fields):
        tr_added.append((tr_field + offset)[:, None])
        va_added.append((va_field + offset)[:, None])
        offset += dim
    Xt = np.concatenate([X_train.astype(np.int64)] + tr_added, axis=1)
    Xv = np.concatenate([X_val.astype(np.int64)] + va_added, axis=1)
    return Xt, Xv, offset


def read_csv_file(path, validation):
    columns = {name: [] for name in ["user_id", "video_id", "tab", "hourmin", "date", "duration_ms"]}
    labels = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            columns["user_id"].append(row["user_id"])
            columns["video_id"].append(row["video_id"])
            columns["tab"].append(row["tab"])
            columns["hourmin"].append(int(float(row["hourmin"])))
            columns["date"].append(int(float(row["date"])))
            columns["duration_ms"].append(float(row["duration_ms"]))
            if validation:
                labels.append(float(row["long_view"]))
            else:
                labels.append(float(row["long_view"]))
    result = {key: np.asarray(value) for key, value in columns.items()}
    result["y"] = np.asarray(labels, dtype=np.float32)
    return result


def make_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping) + 1
    return mapping


def encode_with_mapping(values, mapping):
    return np.fromiter((mapping.get(value, 0) for value in values), dtype=np.int64, count=len(values))


def load_csv_data(data_dir):
    tr = read_csv_file(os.path.join(data_dir, "train.csv"), False)
    va = read_csv_file(os.path.join(data_dir, "val.csv"), True)
    user_map = make_mapping(tr["user_id"])
    video_map = make_mapping(tr["video_id"])
    tab_map = make_mapping(tr["tab"])
    quantiles = np.quantile(tr["duration_ms"], np.linspace(0.1, 0.9, 9))
    quantiles = np.maximum.accumulate(quantiles)

    tr_user = encode_with_mapping(tr["user_id"], user_map)
    va_user = encode_with_mapping(va["user_id"], user_map)
    tr_video = encode_with_mapping(tr["video_id"], video_map)
    va_video = encode_with_mapping(va["video_id"], video_map)
    tr_tab = encode_with_mapping(tr["tab"], tab_map)
    va_tab = encode_with_mapping(va["tab"], tab_map)
    tr_duration = np.searchsorted(quantiles, tr["duration_ms"], side="right").astype(np.int64)
    va_duration = np.searchsorted(quantiles, va["duration_ms"], side="right").astype(np.int64)
    tr_author = np.zeros(len(tr_user), dtype=np.int64)
    va_author = np.zeros(len(va_user), dtype=np.int64)

    dims = [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10]
    train_fields = [tr_user, tr_video, tr_author, tr_tab, tr_duration]
    val_fields = [va_user, va_video, va_author, va_tab, va_duration]
    offset = 0
    encoded_train = []
    encoded_val = []
    for dim, train_field, val_field in zip(dims, train_fields, val_fields):
        encoded_train.append((train_field + offset)[:, None])
        encoded_val.append((val_field + offset)[:, None])
        offset += dim
    return {
        "X_train": np.concatenate(encoded_train, axis=1),
        "X_val": np.concatenate(encoded_val, axis=1),
        "y_train": tr["y"],
        "y_val": va["y"],
        "train_users": tr["user_id"],
        "val_users": va["user_id"],
        "val_videos": va["video_id"],
        "train_hourmin": tr["hourmin"],
        "val_hourmin": va["hourmin"],
        "train_dates": tr["date"],
        "val_dates": va["date"],
        "base_total_dim": offset,
        "npz": False
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr_file = np.load(train_path)
        va_file = np.load(val_path)
        field_dims = tr_file["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "X_train": tr_file["X"].astype(np.int64),
            "X_val": va_file["X"].astype(np.int64),
            "y_train": tr_file["y"].astype(np.float32),
            "y_val": va_file["y"].astype(np.float32),
            "train_users": tr_file["user"],
            "val_users": va_file["user"],
            "val_videos": va_file["X"][:, 1].astype(np.int64) - video_offset,
            "train_hourmin": tr_file["hourmin"],
            "val_hourmin": va_file["hourmin"],
            "train_dates": tr_file["date"],
            "val_dates": va_file["date"],
            "base_total_dim": int(field_dims.sum()),
            "npz": True
        }
    return load_csv_data(data_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    data = load_data(args.data_dir)
    X_train, X_val, total_dim = add_session_features(
        data["X_train"], data["X_val"], data["train_users"], data["val_users"],
        data["train_hourmin"], data["val_hourmin"], data["train_dates"],
        data["val_dates"], data["base_total_dim"])

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(data["y_train"])
    Xv = torch.from_numpy(X_val)
    model = FM(total_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    if data["npz"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = np.concatenate([
                model(Xv[start:start + 65536]).numpy()
                for start in range(0, len(Xv), 65536)
            ])
        metrics = evaluate(data["val_users"], data["y_val"].astype(int), scores)
        primary = float(metrics["primary"])
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(data["val_users"], data["y_val"].astype(int), best_scores)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
            "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
            "primary": float(final_metrics["primary"])
        }, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["val_users"][i], data["val_videos"][i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
