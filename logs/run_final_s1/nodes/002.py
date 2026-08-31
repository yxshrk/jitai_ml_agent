import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch


GAP_EDGES = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.int64)
POS_EDGES = np.asarray([1, 2, 3, 5, 8, 16], dtype=np.int64)


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
        s = e.sum(dim=1)
        pair = 0.5 * (s * s - (e * e).sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


def parse_hourmin(value):
    s = str(value).strip()
    if not s:
        return 0
    if ":" in s:
        parts = s.split(":")
        try:
            return (int(parts[0]) % 24) * 60 + min(max(int(parts[1]), 0), 59)
        except (ValueError, IndexError):
            return 0
    try:
        v = int(float(s))
    except ValueError:
        return 0
    if 0 <= v <= 2359 and v % 100 < 60:
        return (v // 100) * 60 + v % 100
    if 0 <= v < 1440:
        return v
    return 0


def parse_date(value):
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        if len(digits) >= 8:
            d = datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
        else:
            d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(float(s)))
    except (ValueError, OverflowError):
        d = datetime.date(1970, 1, 1)
    return d.toordinal(), d.weekday()


def temporal_arrays(hourmin, dates):
    n = len(hourmin)
    minute = np.empty(n, dtype=np.int64)
    ordinal = np.empty(n, dtype=np.int64)
    weekday = np.empty(n, dtype=np.int64)
    date_cache = {}
    for i in range(n):
        minute[i] = parse_hourmin(hourmin[i])
        key = str(dates[i])
        parsed = date_cache.get(key)
        if parsed is None:
            parsed = parse_date(key)
            date_cache[key] = parsed
        ordinal[i], weekday[i] = parsed
    timestamp = ordinal * 1440 + minute
    hour = minute // 60
    return timestamp, hour.astype(np.int64), weekday


def session_context(user, timestamp, state=None):
    if state is None:
        state = {}
    n = len(user)
    gap_bucket = np.empty(n, dtype=np.int64)
    pos_bucket = np.empty(n, dtype=np.int64)
    for i in range(n):
        key = user[i].item() if isinstance(user[i], np.generic) else user[i]
        now = int(timestamp[i])
        previous = state.get(key)
        if previous is None:
            gap_bucket[i] = 8
            position = 1
        else:
            last_time, last_position = previous
            gap = now - last_time
            if gap < 0:
                gap_bucket[i] = 8
                position = 1
            else:
                gap_bucket[i] = int(np.searchsorted(GAP_EDGES, gap, side="right"))
                position = 1 if gap > 30 else last_position + 1
        pos_bucket[i] = int(np.searchsorted(POS_EDGES, position, side="left"))
        state[key] = (now, position)
    return gap_bucket, pos_bucket, state


def append_session_fields(X, field_dims, user, hourmin, dates, state=None):
    field_dims = np.asarray(field_dims, dtype=np.int64)
    tab_offset = int(field_dims[:3].sum())
    tab_dim = int(field_dims[3])
    tab = X[:, 3].astype(np.int64) - tab_offset
    tab = np.clip(tab, 0, tab_dim - 1)
    timestamp, hour, weekday = temporal_arrays(hourmin, dates)
    gap, position, state = session_context(user, timestamp, state)
    hour_tab = hour * tab_dim + tab
    weekday_tab = weekday * tab_dim + tab
    raw_fields = [gap, position, hour, weekday, hour_tab, weekday_tab]
    added_dims = [9, 7, 24, 7, 24 * tab_dim, 7 * tab_dim]
    out = np.empty((len(X), X.shape[1] + len(raw_fields)), dtype=np.int64)
    out[:, :X.shape[1]] = X.astype(np.int64)
    offset = int(field_dims.sum())
    for j, (values, dim) in enumerate(zip(raw_fields, added_dims)):
        out[:, X.shape[1] + j] = values + offset
        offset += dim
    return out, np.concatenate([field_dims, np.asarray(added_dims, dtype=np.int64)]), state


def categorical_maps(values):
    mapping = {}
    encoded = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        encoded[i] = mapping[key]
    return mapping, encoded


def apply_map(values, mapping):
    return np.asarray([mapping.get(str(v), 0) for v in values], dtype=np.int64)


def read_csv_split(path, require_label):
    columns = {k: [] for k in ["user_id", "video_id", "tab", "hourmin", "date", "duration_ms"]}
    labels = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for key in columns:
                columns[key].append(row.get(key, "0"))
            if require_label:
                labels.append(float(row["long_view"]))
    result = {key: np.asarray(value, dtype=object) for key, value in columns.items()}
    result["y"] = np.asarray(labels, dtype=np.float32) if require_label else None
    return result


def load_csv_data(data_dir):
    tr = read_csv_split(os.path.join(data_dir, "train.csv"), True)
    va = read_csv_split(os.path.join(data_dir, "val.csv"), True)
    user_map, tr_user = categorical_maps(tr["user_id"])
    video_map, tr_video = categorical_maps(tr["video_id"])
    tab_map, tr_tab = categorical_maps(tr["tab"])
    va_user = apply_map(va["user_id"], user_map)
    va_video = apply_map(va["video_id"], video_map)
    va_tab = apply_map(va["tab"], tab_map)
    tr_duration = np.asarray([float(x or 0) for x in tr["duration_ms"]], dtype=np.float64)
    va_duration = np.asarray([float(x or 0) for x in va["duration_ms"]], dtype=np.float64)
    quantiles = np.quantile(tr_duration, np.linspace(0.1, 0.9, 9))
    tr_dur = np.searchsorted(quantiles, tr_duration, side="right").astype(np.int64)
    va_dur = np.searchsorted(quantiles, va_duration, side="right").astype(np.int64)
    author_dim = 1
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, author_dim,
                             len(tab_map) + 1, 10], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(field_dims[:-1])])
    Xt = np.column_stack([tr_user, tr_video, np.zeros(len(tr_user), dtype=np.int64),
                          tr_tab, tr_dur]) + offsets
    Xv = np.column_stack([va_user, va_video, np.zeros(len(va_user), dtype=np.int64),
                          va_tab, va_dur]) + offsets
    grouping = {}
    next_group = 0
    train_group = np.empty(len(tr["user_id"]), dtype=np.int64)
    val_group = np.empty(len(va["user_id"]), dtype=np.int64)
    for target, values in [(train_group, tr["user_id"]), (val_group, va["user_id"])]:
        for i, value in enumerate(values):
            key = str(value)
            if key not in grouping:
                grouping[key] = next_group
                next_group += 1
            target[i] = grouping[key]
    Xt, expanded_dims, state = append_session_fields(
        Xt, field_dims, train_group, tr["hourmin"], tr["date"], None)
    Xv, _, _ = append_session_fields(
        Xv, field_dims, val_group, va["hourmin"], va["date"], state)
    return {
        "Xt": Xt, "yt": tr["y"], "Xv": Xv, "yv": va["y"],
        "eval_user": va["user_id"], "out_user": va["user_id"],
        "out_video": va["video_id"], "field_dims": expanded_dims,
        "official": False
    }


def load_npz_data(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = tr["field_dims"].astype(np.int64)
    Xt, expanded_dims, state = append_session_fields(
        tr["X"], field_dims, tr["user"], tr["hourmin"], tr["date"], None)
    Xv, _, _ = append_session_fields(
        va["X"], field_dims, va["user"], va["hourmin"], va["date"], state)
    video_offset = int(field_dims[0])
    video = va["X"][:, 1].astype(np.int64) - video_offset
    return {
        "Xt": Xt, "yt": tr["y"].astype(np.float32),
        "Xv": Xv, "yv": va["y"].astype(np.float32),
        "eval_user": va["user"], "out_user": va["user"],
        "out_video": video, "field_dims": expanded_dims, "official": True
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast_path = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                 os.path.exists(os.path.join(args.data_dir, "val.npz")))
    data = load_npz_data(args.data_dir) if fast_path else load_csv_data(args.data_dir)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if data["official"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    Xt = torch.from_numpy(data["Xt"].astype(np.int64))
    yt = torch.from_numpy(data["yt"].astype(np.float32))
    Xv = torch.from_numpy(data["Xv"].astype(np.int64))
    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_value = 0.0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(Xv), 65536):
                xb = Xv[start:start + 65536].to(device, non_blocking=True)
                score_parts.append(model(xb).detach().cpu().numpy())
        scores = np.concatenate(score_parts)
        metrics = evaluate(data["eval_user"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_value, 5),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6)
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = evaluate(data["eval_user"], data["yv"].astype(int), best_scores)
    output_metrics = {
        "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
        "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
        "primary": float(final_metrics["primary"]),
        "history": history,
        "features": {
            "gap_edges_minutes": GAP_EDGES.tolist(),
            "session_cut_minutes": 30,
            "position_edges": POS_EDGES.tolist(),
            "context": ["gap_bucket", "session_position_bucket", "hour", "weekday",
                        "hour_x_tab", "weekday_x_tab"]
        }
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["out_user"][i], data["out_video"][i], format(float(score), ".7g")])


if __name__ == "__main__":
    main()
