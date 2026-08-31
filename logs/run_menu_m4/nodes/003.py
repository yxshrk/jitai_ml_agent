"""Gauge-fixed DeepFM with strictly causal pooled author history.

This reproduces the seq-deepfm-author-history package: the five official fields,
hour/weekday/random-tab context, and a mean-pooled embedding of each user's last
12 strictly previous authors. Complete-user-slate gauge-fixed BCE and the
parent's validation checkpoint rule are retained.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from collections import defaultdict, deque

import numpy as np
import torch


class CausalHistoryDeepFM(torch.nn.Module):
    def __init__(self, total_dim, num_fields=8, k=12, hidden=48, dropout=0.05):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(dropout)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear((num_fields + 1) * k, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x, history):
        field_emb = self.embedding_dropout(self.emb(x))
        summed = field_emb.sum(dim=1)
        fm = 0.5 * (summed.square() - field_emb.square().sum(dim=1)).sum(dim=1)
        linear = self.lin(x).sum(dim=(1, 2))

        mask = history.ge(0)
        safe_history = history.clamp_min(0)
        history_emb = self.embedding_dropout(self.emb(safe_history))
        history_emb = history_emb * mask.unsqueeze(-1).to(history_emb.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(history_emb.dtype)
        pooled_history = history_emb.sum(dim=1) / denominator

        deep_input = torch.cat((field_emb.flatten(start_dim=1), pooled_history), dim=1)
        deep_score = self.deep(deep_input).squeeze(1)
        return linear + fm + deep_score


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__unknown_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "hourmin": row.get("hourmin", "0"),
                "date": row.get("date", "19700101"),
                "long_view": float(row["long_view"]),
            })
    return rows


def make_mapping(values):
    return {value: i + 1 for i, value in enumerate(sorted(set(values)))}


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))

    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])

    train_durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=object)
        videos = []
        hourmin = np.empty(len(rows), dtype=np.int64)
        dates = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            raw_user = row["user_id"]
            users[i] = raw_user
            videos.append(row["video_id"])
            values = (
                user_map.get(raw_user, 0),
                video_map.get(row["video_id"], 0),
                author_map.get(row["author_id"], 0),
                tab_map.get(row["tab"], 0),
                int(np.searchsorted(quantiles, row["duration_ms"], side="right")),
            )
            x[i] = np.asarray(values, dtype=np.int64) + offsets
            y[i] = row["long_view"]
            try:
                hourmin[i] = int(float(row["hourmin"]))
            except (TypeError, ValueError):
                hourmin[i] = 0
            dates[i] = row["date"]
        return x, y, users, videos, hourmin, dates

    xt, yt, train_users, _, train_hourmin, train_dates = encode(train_rows)
    xv, yv, val_users, val_videos, val_hourmin, val_dates = encode(val_rows)
    return {
        "Xt": xt,
        "yt": yt,
        "train_user": train_users,
        "train_hourmin": train_hourmin,
        "train_date": train_dates,
        "Xv": xv,
        "yv": yv,
        "val_user": val_users,
        "val_video": val_videos,
        "val_hourmin": val_hourmin,
        "val_date": val_dates,
        "field_dims": field_dims,
        "fast_path": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz, allow_pickle=False) as tr, np.load(val_npz, allow_pickle=False) as va:
            return {
                "Xt": tr["X"].astype(np.int64),
                "yt": tr["y"].astype(np.float32),
                "train_user": np.asarray(tr["user"]),
                "train_hourmin": np.asarray(tr["hourmin"]),
                "train_date": np.asarray(tr["date"]),
                "Xv": va["X"].astype(np.int64),
                "yv": va["y"].astype(np.float32),
                "val_user": np.asarray(va["user"]),
                "val_video": ["0"] * len(va["y"]),
                "val_hourmin": np.asarray(va["hourmin"]),
                "val_date": np.asarray(va["date"]),
                "field_dims": tr["field_dims"].astype(np.int64),
                "fast_path": True,
            }
    return load_csv_data(data_dir)


def extract_hours(hourmin):
    values = np.asarray(hourmin)
    result = np.zeros(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            number = 0
        if 0 <= number <= 23:
            hour = number
        elif 0 <= number <= 2359 and number % 100 < 60:
            hour = number // 100
        else:
            hour = (number // 60) % 24
        result[i] = hour % 24
    return result


def extract_weekdays(dates):
    values = np.asarray(dates)
    result = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        if isinstance(value, bytes):
            key = value.decode("utf-8", errors="ignore")
        else:
            key = str(value)
        key = key.split(".")[0].replace("-", "").replace("/", "")
        if key not in cache:
            try:
                cache[key] = datetime.datetime.strptime(key[:8], "%Y%m%d").weekday()
            except (TypeError, ValueError):
                cache[key] = 0
        result[i] = cache[key]
    return result


def add_context_fields(x, hourmin, dates, field_dims):
    base_total = int(np.sum(field_dims))
    hours = extract_hours(hourmin)
    weekdays = extract_weekdays(dates)
    tab_offset = int(np.sum(field_dims[:3]))
    tab_local = x[:, 3].astype(np.int64) - tab_offset
    is_rand = (tab_local == 2).astype(np.int64)

    result = np.empty((len(x), 8), dtype=np.int64)
    result[:, :5] = x
    result[:, 5] = base_total + hours
    result[:, 6] = base_total + 24 + weekdays
    result[:, 7] = base_total + 24 + 7 + is_rand
    return result


def build_causal_histories(train_users, train_authors, val_users, val_authors, length=12):
    states = defaultdict(lambda: deque(maxlen=length))
    train_history = np.full((len(train_users), length), -1, dtype=np.int32)
    val_history = np.full((len(val_users), length), -1, dtype=np.int32)

    for i in range(len(train_users)):
        user = train_users[i].item() if isinstance(train_users[i], np.generic) else train_users[i]
        state = states[user]
        if state:
            values = list(state)
            train_history[i, length - len(values):] = values
        state.append(int(train_authors[i]))

    for i in range(len(val_users)):
        user = val_users[i].item() if isinstance(val_users[i], np.generic) else val_users[i]
        state = states[user]
        if state:
            values = list(state)
            val_history[i, length - len(values):] = values
        state.append(int(val_authors[i]))

    return train_history, val_history


def build_user_slates(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return [order[start:end] for start, end in zip(starts, ends)]


def make_complete_slate_batches(slates, rng, target_size):
    slate_order = rng.permutation(len(slates))
    batches = []
    current = []
    current_size = 0
    for slate_id in slate_order:
        slate = slates[int(slate_id)]
        if current and current_size + len(slate) > target_size:
            batches.append(current)
            current = []
            current_size = 0
        current.append(slate)
        current_size += len(slate)
        if current_size >= target_size:
            batches.append(current)
            current = []
            current_size = 0
    if current:
        batches.append(current)
    return batches


def centered_logits(raw_logits, lengths, global_bias):
    group_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=raw_logits.device),
        torch.as_tensor(lengths, device=raw_logits.device, dtype=torch.long),
    )
    sums = torch.zeros(len(lengths), device=raw_logits.device, dtype=raw_logits.dtype)
    sums.scatter_add_(0, group_ids, raw_logits)
    lengths_tensor = torch.as_tensor(lengths, device=raw_logits.device, dtype=raw_logits.dtype)
    means = sums / lengths_tensor
    return raw_logits - means[group_ids] + global_bias


def get_evaluator(fast_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_value(metrics, upper, lower):
    return metrics[upper] if upper in metrics else metrics[lower]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast_path"])

    field_dims = data["field_dims"]
    xt_context = add_context_fields(data["Xt"], data["train_hourmin"], data["train_date"], field_dims)
    xv_context = add_context_fields(data["Xv"], data["val_hourmin"], data["val_date"], field_dims)
    train_history, val_history = build_causal_histories(
        np.asarray(data["train_user"]),
        data["Xt"][:, 2],
        np.asarray(data["val_user"]),
        data["Xv"][:, 2],
        length=12,
    )

    xt = torch.from_numpy(xt_context)
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(xv_context)
    slates = build_user_slates(np.asarray(data["train_user"]))

    total_dim = int(np.sum(field_dims)) + 24 + 7 + 2
    model = CausalHistoryDeepFM(
        total_dim=total_dim,
        num_fields=8,
        k=12,
        hidden=48,
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00115, weight_decay=3e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(args.seed)

    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        batches = make_complete_slate_batches(slates, rng, 8192)
        total_loss = 0.0
        total_examples = 0

        for slate_batch in batches:
            indices = np.concatenate(slate_batch).astype(np.int64, copy=False)
            lengths = [len(slate) for slate in slate_batch]
            idx = torch.from_numpy(indices)
            xb = xt[idx].to(device, non_blocking=True)
            hb = torch.from_numpy(train_history[indices].astype(np.int64)).to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            raw = model(xb, hb)
            fixed = centered_logits(raw, lengths, model.global_bias)
            loss = criterion(fixed, yb)
            loss.backward()
            optimizer.step()

            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_examples += count

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(xv), 65536):
                end = min(start + 65536, len(xv))
                xb = xv[start:end].to(device, non_blocking=True)
                hb = torch.from_numpy(val_history[start:end].astype(np.int64)).to(device, non_blocking=True)
                score_parts.append(model(xb, hb).detach().cpu().numpy())
        scores = np.concatenate(score_parts)
        metrics = evaluate(data["val_user"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, total_examples), 5),
            "val_gauc": round(float(metric_value(metrics, "GAUC", "gauc")), 6),
            "val_ndcg5": round(float(metric_value(metrics, "nDCG@5", "ndcg5")), 6),
            "val_primary": round(primary, 6),
            "config": {
                "model": "gauge-fixed causal pooled-author-history DeepFM",
                "embedding_dim": 12,
                "hidden": 48,
                "history_length": 12,
                "fields": "five base plus hour weekday is_rand",
                "lr": 0.00115,
                "weight_decay": 3e-6,
                "dropout": 0.05,
            },
        })

        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    final_metrics = evaluate(data["val_user"], data["yv"].astype(int), best_scores)
    result = {
        "gauc": float(metric_value(final_metrics, "GAUC", "gauc")),
        "ndcg5": float(metric_value(final_metrics, "nDCG@5", "ndcg5")),
        "primary": float(final_metrics["primary"]),
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.6g}\n")


if __name__ == "__main__":
    main()
