import argparse
import csv
import datetime
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        rows = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user_raw": row["user_id"],
                    "video_raw": row["video_id"],
                    "tab_raw": row.get("tab", "0"),
                    "hourmin": parse_int(row.get("hourmin", 0)),
                    "date": parse_int(row.get("date", 19700101), 19700101),
                    "duration_ms": max(parse_int(row.get("duration_ms", 0)), 0),
                    "y": float(row.get("long_view", 0.0)),
                }
                if training:
                    item["play_time_ms"] = max(parse_int(row.get("play_time_ms", 0)), 0)
                rows.append(item)
        return rows

    tr_rows = read_rows(train_path, True)
    va_rows = read_rows(val_path, False)

    def make_map(values):
        unique = sorted(set(values))
        return {v: i for i, v in enumerate(unique)}

    user_map = make_map([r["user_raw"] for r in tr_rows])
    video_map = make_map([r["video_raw"] for r in tr_rows])
    author_map = dict(video_map)
    tab_map = make_map([r["tab_raw"] for r in tr_rows])
    durations = np.asarray([r["duration_ms"] for r in tr_rows], dtype=np.float64)
    if len(durations):
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    field_dims = np.asarray([
        max(len(user_map), 1), max(len(video_map), 1), max(len(author_map), 1),
        max(len(tab_map), 1), 10
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows, training):
        n = len(rows)
        X = np.empty((n, 5), dtype=np.int32)
        y = np.empty(n, dtype=np.float32)
        user_eval = np.empty(n, dtype=np.int64)
        duration = np.empty(n, dtype=np.float32)
        hourmin = np.empty(n, dtype=np.int64)
        date = np.empty(n, dtype=np.int64)
        play = np.empty(n, dtype=np.float32) if training else None
        video_out = []
        for i, r in enumerate(rows):
            u = user_map.get(r["user_raw"], 0)
            v = video_map.get(r["video_raw"], 0)
            au = author_map.get(r["video_raw"], 0)
            tb = tab_map.get(r["tab_raw"], 0)
            db = int(np.searchsorted(edges, r["duration_ms"], side="right"))
            X[i] = np.asarray([u, v, au, tb, min(db, 9)], dtype=np.int64) + offsets
            y[i] = r["y"]
            user_eval[i] = u
            duration[i] = r["duration_ms"]
            hourmin[i] = r["hourmin"]
            date[i] = r["date"]
            video_out.append(r["video_raw"])
            if training:
                play[i] = r["play_time_ms"]
        result = {
            "X": X, "y": y, "user": user_eval, "duration_ms": duration,
            "hourmin": hourmin, "date": date, "video_out": np.asarray(video_out),
            "field_dims": field_dims
        }
        if training:
            result["play_time_ms"] = play
        return result

    return encode(tr_rows, True), encode(va_rows, False), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {
            "X": tr_file["X"].astype(np.int32, copy=False),
            "y": tr_file["y"].astype(np.float32, copy=False),
            "user": tr_file["user"],
            "play_time_ms": tr_file["play_time_ms"].astype(np.float32, copy=False),
            "duration_ms": tr_file["duration_ms"].astype(np.float32, copy=False),
            "hourmin": tr_file["hourmin"],
            "date": tr_file["date"],
            "field_dims": tr_file["field_dims"].astype(np.int64, copy=False),
        }
        offsets = np.concatenate(([0], np.cumsum(tr["field_dims"])[:-1])).astype(np.int64)
        video_local = va_file["X"][:, 1].astype(np.int64) - offsets[1]
        va = {
            "X": va_file["X"].astype(np.int32, copy=False),
            "y": va_file["y"].astype(np.float32, copy=False),
            "user": va_file["user"],
            "duration_ms": va_file["duration_ms"].astype(np.float32, copy=False),
            "hourmin": va_file["hourmin"],
            "date": va_file["date"],
            "video_out": video_local,
            "field_dims": tr["field_dims"],
        }
        return tr, va, True
    return load_csv_data(data_dir)


def date_ordinals(values):
    values = np.asarray(values).astype(np.int64, copy=False)
    out = np.empty(len(values), dtype=np.int64)
    cache = {}
    for v in np.unique(values):
        iv = int(v)
        try:
            d = datetime.date(iv // 10000, (iv // 100) % 100, iv % 100)
            ordinal = d.toordinal()
        except ValueError:
            ordinal = iv
        cache[iv] = ordinal
    for i, v in enumerate(values):
        out[i] = cache[int(v)]
    return out


def temporal_arrays(data):
    hm = np.asarray(data["hourmin"]).astype(np.int64, copy=False)
    hour = np.clip(hm // 100, 0, 23)
    minute = np.clip(hm % 100, 0, 59)
    ords = date_ordinals(data["date"])
    weekday = np.mod(ords + 6, 7)
    timestamp = ords * 1440 + hour * 60 + minute
    return hour.astype(np.int32), weekday.astype(np.int32), timestamp.astype(np.int64)


def build_causal_features(tr, va):
    total_base = int(np.sum(tr["field_dims"]))
    tab_offset = int(np.sum(tr["field_dims"][:3]))
    tr_hour, tr_weekday, tr_time = temporal_arrays(tr)
    va_hour, va_weekday, va_time = temporal_arrays(va)
    tr_tab_local = tr["X"][:, 3].astype(np.int64) - tab_offset
    va_tab_local = va["X"][:, 3].astype(np.int64) - tab_offset
    tr_rand = (tr_tab_local == 1).astype(np.int32)
    va_rand = (va_tab_local == 1).astype(np.int32)

    context_dims = np.asarray([24, 7, 2, 10, 10], dtype=np.int64)
    context_offsets = total_base + np.concatenate(([0], np.cumsum(context_dims)[:-1])).astype(np.int64)
    pad_id = total_base + int(np.sum(context_dims))

    tr_hist = np.full((len(tr["X"]), 12), pad_id, dtype=np.int32)
    va_hist = np.full((len(va["X"]), 12), pad_id, dtype=np.int32)
    tr_gap = np.empty(len(tr["X"]), dtype=np.int32)
    va_gap = np.empty(len(va["X"]), dtype=np.int32)
    tr_pos = np.empty(len(tr["X"]), dtype=np.int32)
    va_pos = np.empty(len(va["X"]), dtype=np.int32)
    gap_edges = np.asarray([1, 2, 5, 10, 30, 60, 180, 720, 1440], dtype=np.int64)

    histories = {}
    last_time = {}
    session_pos = {}

    def process(data, timestamps, histories_out, gap_out, pos_out):
        order = np.lexsort((np.arange(len(data["X"]), dtype=np.int64), timestamps, data["user"]))
        authors = data["X"][:, 2]
        for idx in order:
            key = int(data["user"][idx])
            hist = histories.get(key)
            if hist is None:
                hist = deque(maxlen=12)
                histories[key] = hist
            if hist:
                vals = list(hist)
                histories_out[idx, 12 - len(vals):] = vals
            now = int(timestamps[idx])
            previous = last_time.get(key)
            if previous is None:
                gap = 1000000
                position = 0
            else:
                gap = max(now - previous, 0)
                if gap > 30:
                    position = 0
                else:
                    position = session_pos.get(key, 0) + 1
            gap_out[idx] = min(int(np.searchsorted(gap_edges, gap, side="right")), 9)
            pos_out[idx] = min(position, 9)
            last_time[key] = now
            session_pos[key] = position
            hist.append(int(authors[idx]))

    process(tr, tr_time, tr_hist, tr_gap, tr_pos)
    process(va, va_time, va_hist, va_gap, va_pos)

    def combine(data, hour, weekday, rand, gap, pos):
        context = np.column_stack((hour, weekday, rand, gap, pos)).astype(np.int64)
        context += context_offsets.reshape(1, -1)
        return np.concatenate((data["X"].astype(np.int64), context), axis=1).astype(np.int32)

    tr_cat = combine(tr, tr_hour, tr_weekday, tr_rand, tr_gap, tr_pos)
    va_cat = combine(va, va_hour, va_weekday, va_rand, va_gap, va_pos)
    return tr_cat, tr_hist, va_cat, va_hist, pad_id + 1, pad_id


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, pad_id, fields=10, k=16):
        super().__init__()
        self.pad_id = pad_id
        self.emb = torch.nn.Embedding(total_dim, k, padding_idx=pad_id)
        self.linear = torch.nn.Embedding(total_dim, 1, padding_idx=pad_id)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.emb.weight[pad_id].zero_()
            self.linear.weight[pad_id].zero_()
        self.emb_dropout = torch.nn.Dropout(0.10)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear((fields + 1) * k, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
        )
        self.deep_head = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)

    def forward(self, x, history):
        current = self.emb_dropout(self.emb(x))
        h_raw = self.emb(history)
        mask = history.ne(self.pad_id).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        pooled = (h_raw * mask).sum(dim=1) / count
        pooled = self.emb_dropout(pooled)
        all_fields = torch.cat((current, pooled.unsqueeze(1)), dim=1)
        summed = all_fields.sum(dim=1)
        pair = 0.5 * (summed.square() - all_fields.square().sum(dim=1)).sum(dim=1)
        hist_linear = self.linear(history).squeeze(-1)
        hist_mask = history.ne(self.pad_id)
        hist_count = hist_mask.sum(dim=1).clamp_min(1)
        hist_linear = (hist_linear * hist_mask).sum(dim=1) / hist_count
        linear = self.linear(x).sum(dim=(1, 2)) + hist_linear
        representation = self.mlp(all_fields.flatten(1))
        main = self.bias + linear + pair + self.deep_head(representation).squeeze(1)
        watch = self.watch_head(representation).squeeze(1)
        return main, watch


def evaluate_scores(evaluate_fn, users, labels, scores):
    return evaluate_fn(users, labels.astype(int), scores)


def train_member(member_seed, epochs, device, tr, va, tr_cat, tr_hist, va_cat, va_hist,
                 total_dim, pad_id, evaluate_fn):
    seed_all(member_seed)
    model = SequenceDeepFM(total_dim, pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.55)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(tr_cat)
    batch_size = 8192 if device.type == "cuda" else 4096
    rng = np.random.RandomState(member_seed)
    y = tr["y"].astype(np.float32, copy=False)
    duration = tr["duration_ms"].astype(np.float32, copy=False)
    play = tr["play_time_ms"].astype(np.float32, copy=False)
    valid_watch = duration > 0
    observed_ratio = np.zeros(n, dtype=np.float32)
    observed_ratio[valid_watch] = np.clip(play[valid_watch] / duration[valid_watch], 0.0, 1.0)
    censored = valid_watch & (play >= duration)
    best_primary = -1.0
    best_scores = None
    patience = 0
    history_log = []

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        loss_sum = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            xb = torch.as_tensor(tr_cat[idx], dtype=torch.long, device=device)
            hb = torch.as_tensor(tr_hist[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=device)
            rb = torch.as_tensor(observed_ratio[idx], dtype=torch.float32, device=device)
            vb = torch.as_tensor(valid_watch[idx], dtype=torch.bool, device=device)
            cb = torch.as_tensor(censored[idx], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, watch_logits = model(xb, hb)
            main_loss = bce(logits, yb)
            prediction = torch.sigmoid(watch_logits)
            uncensored_mask = vb & (~cb)
            watch_terms = []
            if uncensored_mask.any():
                watch_terms.append((prediction[uncensored_mask] - rb[uncensored_mask]).square().mean())
            if cb.any():
                watch_terms.append(torch.relu(rb[cb] - prediction[cb]).square().mean())
            if watch_terms:
                watch_loss = torch.stack(watch_terms).mean()
                loss = main_loss + 0.08 * watch_loss
            else:
                loss = main_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            batches += 1
        scheduler.step()

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(va_cat), 65536):
                xb = torch.as_tensor(va_cat[start:start + 65536], dtype=torch.long, device=device)
                hb = torch.as_tensor(va_hist[start:start + 65536], dtype=torch.long, device=device)
                logits, _ = model(xb, hb)
                score_parts.append(logits.detach().cpu().numpy())
        scores = np.concatenate(score_parts).astype(np.float64)
        metrics = evaluate_scores(evaluate_fn, va["user"], va["y"], scores)
        primary = float(metrics["primary"])
        history_log.append({
            "seed": member_seed,
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(batches, 1), 6),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6)
        })
        if primary > best_primary + 1e-7:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, history_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    tr_cat, tr_hist, va_cat, va_hist, total_dim, pad_id = build_causal_features(tr, va)
    member_scores = []
    history = []
    for member_seed in (args.seed, args.seed + 1):
        scores, member_history = train_member(
            member_seed, epochs, device, tr, va, tr_cat, tr_hist, va_cat, va_hist,
            total_dim, pad_id, evaluate_fn
        )
        member_scores.append(scores)
        history.extend(member_history)

    ensemble_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    metrics = evaluate_scores(evaluate_fn, va["user"], va["y"], ensemble_scores)
    output = {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
        "history": history,
        "ensemble": {"method": "mean_logit", "seeds": [args.seed, args.seed + 1]}
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(ensemble_scores):
            writer.writerow([i, va["user"][i], va["video_out"][i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
