import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def date_ordinals(values):
    values = np.asarray(values)
    out = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        text = text.replace("-", "")
        if text not in cache:
            try:
                cache[text] = datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
            except Exception:
                cache[text] = 0
        out[i] = cache[text]
    return out


def parse_hourmin(values):
    arr = np.asarray(values)
    out = np.zeros(len(arr), dtype=np.int64)
    for i, value in enumerate(arr):
        try:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            number = int(float(value))
            hour = max(0, min(23, number // 100))
            minute = max(0, min(59, number % 100))
            out[i] = hour * 60 + minute
        except Exception:
            out[i] = 0
    return out


def recency_weights(dates, half_life):
    ords = date_ordinals(dates).astype(np.float32)
    valid = ords > 0
    if not np.any(valid):
        return np.ones(len(ords), dtype=np.float32)
    age = np.maximum(0.0, float(ords[valid].max()) - ords)
    weights = np.power(0.5, age / float(half_life)).astype(np.float32)
    weights[~valid] = 1.0
    return weights


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        columns = {
            "user": [], "video": [], "author": [], "tab": [], "duration": [],
            "y": [], "date": [], "hourmin": [], "play": []
        }
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                columns["user"].append(row.get("user_id", ""))
                columns["video"].append(row.get("video_id", ""))
                columns["author"].append(row.get("author_id", row.get("video_id", "")))
                columns["tab"].append(row.get("tab", ""))
                columns["duration"].append(float(row.get("duration_ms", 0) or 0))
                columns["y"].append(float(row.get("long_view", 0) or 0))
                columns["date"].append(row.get("date", ""))
                columns["hourmin"].append(row.get("hourmin", 0))
                if training:
                    columns["play"].append(float(row.get("play_time_ms", 0) or 0))
        return columns

    tr = read_rows(train_path, True)
    va = read_rows(val_path, False)
    durations = np.asarray(tr["duration"], dtype=np.float64)
    quantiles = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9))) if len(durations) else np.array([])
    train_fields = [tr["user"], tr["video"], tr["author"], tr["tab"]]
    val_fields = [va["user"], va["video"], va["author"], va["tab"]]
    train_fields.append(np.searchsorted(quantiles, durations, side="right").astype(str).tolist())
    val_fields.append(np.searchsorted(quantiles, np.asarray(va["duration"]), side="right").astype(str).tolist())
    encoded_train = []
    encoded_val = []
    field_dims = []
    offset = 0
    for train_values, val_values in zip(train_fields, val_fields):
        mapping = {}
        for value in train_values:
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping)
        unknown = len(mapping)
        dim = unknown + 1
        encoded_train.append(np.asarray([mapping[str(v)] + offset for v in train_values], dtype=np.int64))
        encoded_val.append(np.asarray([mapping.get(str(v), unknown) + offset for v in val_values], dtype=np.int64))
        field_dims.append(dim)
        offset += dim
    return {
        "Xt": np.stack(encoded_train, axis=1),
        "Xv": np.stack(encoded_val, axis=1),
        "yt": np.asarray(tr["y"], dtype=np.float32),
        "yv": np.asarray(va["y"], dtype=np.float32),
        "train_user": np.asarray(encoded_train[0]),
        "val_user": np.asarray(encoded_val[0]),
        "train_date": np.asarray(tr["date"]),
        "val_date": np.asarray(va["date"]),
        "train_hourmin": np.asarray(tr["hourmin"]),
        "val_hourmin": np.asarray(va["hourmin"]),
        "train_play": np.asarray(tr["play"], dtype=np.float32),
        "train_duration": np.asarray(tr["duration"], dtype=np.float32),
        "val_duration": np.asarray(va["duration"], dtype=np.float32),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
        "val_video_output": np.asarray(va["video"]),
        "fast": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        tr = np.load(train_path, allow_pickle=False)
        va = np.load(val_path, allow_pickle=False)
        return {
            "Xt": tr["X"].astype(np.int64, copy=False),
            "Xv": va["X"].astype(np.int64, copy=False),
            "yt": tr["y"].astype(np.float32, copy=False),
            "yv": va["y"].astype(np.float32, copy=False),
            "train_user": np.asarray(tr["user"]),
            "val_user": np.asarray(va["user"]),
            "train_date": np.asarray(tr["date"]),
            "val_date": np.asarray(va["date"]),
            "train_hourmin": np.asarray(tr["hourmin"]),
            "val_hourmin": np.asarray(va["hourmin"]),
            "train_play": np.asarray(tr["play_time_ms"], dtype=np.float32),
            "train_duration": np.asarray(tr["duration_ms"], dtype=np.float32),
            "val_duration": np.asarray(va["duration_ms"], dtype=np.float32),
            "field_dims": np.asarray(tr["field_dims"], dtype=np.int64),
            "val_video_output": va["X"][:, 1],
            "fast": True,
        }
    return load_csv_data(data_dir)


def build_composite_features(data, history_length=12, session_minutes=30):
    Xt = data["Xt"]
    Xv = data["Xv"]
    field_dims = data["field_dims"]
    base_total = int(field_dims.sum())
    train_ord = date_ordinals(data["train_date"])
    val_ord = date_ordinals(data["val_date"])
    train_minute = parse_hourmin(data["train_hourmin"])
    val_minute = parse_hourmin(data["val_hourmin"])
    train_time = train_ord * 1440 + train_minute
    val_time = val_ord * 1440 + val_minute
    train_hour = train_minute // 60
    val_hour = val_minute // 60
    train_weekday = np.where(train_ord > 0, (train_ord + 6) % 7, 0)
    val_weekday = np.where(val_ord > 0, (val_ord + 6) % 7, 0)
    tab_offset = int(field_dims[:3].sum())
    train_tab_raw = Xt[:, 3] - tab_offset
    val_tab_raw = Xv[:, 3] - tab_offset
    train_rand = (train_tab_raw == 1).astype(np.int64)
    val_rand = (val_tab_raw == 1).astype(np.int64)

    max_user = int(max(np.max(Xt[:, 0]) if len(Xt) else 0, np.max(Xv[:, 0]) if len(Xv) else 0)) + 1
    ring = np.zeros((max_user, history_length), dtype=np.int64)
    ring_count = np.zeros(max_user, dtype=np.int16)
    ring_ptr = np.zeros(max_user, dtype=np.int16)
    last_time = np.full(max_user, -1, dtype=np.int64)
    session_pos = np.zeros(max_user, dtype=np.int32)
    hist_t = np.zeros((len(Xt), history_length), dtype=np.int64)
    hist_v = np.zeros((len(Xv), history_length), dtype=np.int64)
    gap_t = np.zeros(len(Xt), dtype=np.int64)
    gap_v = np.zeros(len(Xv), dtype=np.int64)
    pos_t = np.zeros(len(Xt), dtype=np.int64)
    pos_v = np.zeros(len(Xv), dtype=np.int64)
    gap_edges = np.asarray([1, 5, 15, 30, 120, 720], dtype=np.int64)

    def process(X, times, history, gaps, positions):
        order = np.lexsort((np.arange(len(X), dtype=np.int64), times, X[:, 0]))
        for idx in order:
            user = int(X[idx, 0])
            count = int(ring_count[user])
            ptr = int(ring_ptr[user])
            if count:
                if count < history_length:
                    history[idx, :count] = ring[user, :count]
                else:
                    history[idx] = np.concatenate((ring[user, ptr:], ring[user, :ptr]))
            previous = int(last_time[user])
            now = int(times[idx])
            if previous < 0 or now < previous:
                gap = 10 ** 9
                session_pos[user] = 0
            else:
                gap = now - previous
                if gap > session_minutes:
                    session_pos[user] = 0
                else:
                    session_pos[user] += 1
            gaps[idx] = int(np.searchsorted(gap_edges, gap, side="right"))
            positions[idx] = min(9, int(np.floor(np.log2(1 + int(session_pos[user])))))
            author = int(X[idx, 2])
            ring[user, ptr] = author
            ring_ptr[user] = (ptr + 1) % history_length
            ring_count[user] = min(history_length, count + 1)
            last_time[user] = now

    process(Xt, train_time, hist_t, gap_t, pos_t)
    process(Xv, val_time, hist_v, gap_v, pos_v)

    context_dims = [24, 7, 2, 7, 10]
    offsets = np.cumsum([base_total] + context_dims[:-1]).astype(np.int64)
    context_t = np.stack([train_hour, train_weekday, train_rand, gap_t, pos_t], axis=1).astype(np.int64)
    context_v = np.stack([val_hour, val_weekday, val_rand, gap_v, pos_v], axis=1).astype(np.int64)
    context_t += offsets[None, :]
    context_v += offsets[None, :]
    full_t = np.concatenate([Xt, context_t], axis=1)
    full_v = np.concatenate([Xv, context_v], axis=1)
    return full_t, full_v, hist_t, hist_v, base_total + int(sum(context_dims))


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    positive = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[positive[idx]]
        neg = idx[~positive[idx]]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def metric_dict(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, current_fields=10, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.current_fields = current_fields
        self.dropout = float(dropout)
        self.emb = torch.nn.Embedding(total_dim, k, padding_idx=None)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        input_dim = (current_fields + 1) * k
        self.deep1 = torch.nn.Linear(input_dim, hidden)
        self.deep2 = torch.nn.Linear(hidden, hidden // 2)
        self.main_out = torch.nn.Linear(hidden // 2, 1, bias=False)
        self.watch_out = torch.nn.Linear(hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for layer in (self.deep1, self.deep2, self.main_out, self.watch_out):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x, history):
        current = self.emb(x)
        mask = (history != 0).float().unsqueeze(-1)
        hist_emb = self.emb(history)
        pooled = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        fields = torch.cat([current, pooled.unsqueeze(1)], dim=1)
        dropped = F.dropout(fields, p=self.dropout, training=self.training)
        summed = dropped.sum(dim=1)
        fm = 0.5 * (summed.square() - dropped.square().sum(dim=1)).sum(dim=1)
        linear = self.linear(x).sum(dim=(1, 2))
        deep = dropped.flatten(1)
        deep = F.relu(self.deep1(deep))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        deep = F.relu(self.deep2(deep))
        deep = F.dropout(deep, p=self.dropout, training=self.training)
        logit = self.bias + linear + fm + self.main_out(deep).squeeze(1)
        watch = self.watch_out(deep).squeeze(1)
        return logit, watch


class ParentReference(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16):
        super().__init__()
        d = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.deep1 = torch.nn.Linear(d, 128)
        self.deep2 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(dim=1)
        fm = 0.5 * (summed.square() - e.square().sum(dim=1)).sum(dim=1)
        deep = F.relu(self.deep1(e.flatten(1)))
        deep = F.relu(self.deep2(deep))
        return self.linear(x).sum(dim=(1, 2)) + fm + self.out(deep).squeeze(1)


def predict_composite(model, X, H, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device, non_blocking=True)
            hb = torch.from_numpy(H[start:start + batch_size]).to(device, non_blocking=True)
            logits, _ = model(xb, hb)
            parts.append(logits.cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_composite(config, seed, epochs, Xt, Ht, yt, watch_target, censored,
                    Xv, Hv, val_users, val_y, weights, pair_pos, pair_neg,
                    total_dim, evaluator, device, select_each_epoch=False):
    seed_all(seed)
    model = SequenceDeepFM(total_dim, dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    rng = np.random.RandomState(seed + 19)
    bs = 8192 if device.type == "cuda" else 4096
    best_primary = -1.0
    best_scores = None
    checkpoints = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(yt))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        pair_cursor = 0
        last_loss = 0.0
        for start in range(0, len(order), bs):
            idx_np = order[start:start + bs]
            xb = torch.from_numpy(Xt[idx_np]).to(device, non_blocking=True)
            hb = torch.from_numpy(Ht[idx_np]).to(device, non_blocking=True)
            yb = torch.from_numpy(yt[idx_np]).to(device, non_blocking=True)
            wb = torch.from_numpy(weights[idx_np]).to(device, non_blocking=True)
            target_b = torch.from_numpy(watch_target[idx_np]).to(device, non_blocking=True)
            censored_b = torch.from_numpy(censored[idx_np]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, watch_pred = model(xb, hb)
            point_losses = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-6)
            exact_watch = F.smooth_l1_loss(watch_pred, target_b, reduction="none")
            lower_watch = F.relu(target_b - watch_pred).square()
            watch_losses = torch.where(censored_b, lower_watch, exact_watch)
            watch_loss = (watch_losses * wb).sum() / wb.sum().clamp_min(1e-6)
            rank_loss = torch.zeros((), device=device)
            if len(pair_order):
                wanted = max(1, len(idx_np) // 2)
                if pair_cursor + wanted > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                take = pair_order[pair_cursor:pair_cursor + wanted]
                pair_cursor += len(take)
                p = pair_pos[take]
                n = pair_neg[take]
                pair_x = np.concatenate([Xt[p], Xt[n]], axis=0)
                pair_h = np.concatenate([Ht[p], Ht[n]], axis=0)
                pair_logits, _ = model(torch.from_numpy(pair_x).to(device), torch.from_numpy(pair_h).to(device))
                p_logit, n_logit = pair_logits.chunk(2)
                rank_loss = F.softplus(-(p_logit - n_logit)).mean()
            mix = float(config["bpr_mix"])
            loss = (1.0 - mix) * point_loss + mix * rank_loss + float(config["aux_weight"]) * watch_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        scheduler.step()
        if select_each_epoch:
            scores = predict_composite(model, Xv, Hv, device)
            metrics = metric_dict(evaluator, val_users, val_y, scores)
            checkpoints.append({
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 7),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "gauc": round(metrics["gauc"], 7),
                "ndcg5": round(metrics["ndcg5"], 7),
                "primary": round(metrics["primary"], 7),
            })
            if metrics["primary"] > best_primary:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
    if best_scores is None:
        best_scores = predict_composite(model, Xv, Hv, device)
        best_primary = metric_dict(evaluator, val_users, val_y, best_scores)["primary"]
    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(best_primary), best_scores, checkpoints


def train_parent_reference(Xt, yt, Xv, base_total, seed, epochs, device):
    seed_all(seed)
    model = ParentReference(base_total).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    rng = np.random.RandomState(seed + 3)
    bs = 8192 if device.type == "cuda" else 4096
    for _ in range(epochs):
        model.train()
        for start in range(0, len(yt), bs):
            idx = rng.randint(0, len(yt), size=min(bs, len(yt) - start))
            xb = torch.from_numpy(Xt[idx, :5]).to(device)
            yb = torch.from_numpy(yt[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = torch.from_numpy(Xv[start:start + 65536, :5]).to(device)
            parts.append(model(xb).cpu().numpy())
    scores = np.concatenate(parts).astype(np.float64)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def random_configs(seed, count):
    rng = np.random.RandomState(seed + 811)
    configs = []
    for _ in range(count):
        configs.append({
            "dropout": float(rng.uniform(0.14, 0.38)),
            "weight_decay": float(np.exp(rng.uniform(np.log(2e-5), np.log(3e-3)))),
            "lr": float(np.exp(rng.uniform(np.log(3.5e-4), np.log(1.8e-3)))),
            "step_size": int(rng.choice([1, 2, 3])),
            "gamma": float(rng.uniform(0.35, 0.78)),
            "aux_weight": float(rng.uniform(0.035, 0.18)),
            "bpr_mix": float(rng.uniform(0.30, 0.58)),
            "half_life": float(np.exp(rng.uniform(np.log(4.0), np.log(18.0)))),
        })
    return configs


def refine_configs(winner, seed, count):
    rng = np.random.RandomState(seed + 1409)
    configs = [dict(winner)]
    while len(configs) < count:
        configs.append({
            "dropout": float(np.clip(winner["dropout"] + rng.uniform(-0.05, 0.05), 0.10, 0.44)),
            "weight_decay": float(np.clip(winner["weight_decay"] * np.exp(rng.uniform(-0.65, 0.65)), 1e-5, 6e-3)),
            "lr": float(np.clip(winner["lr"] * np.exp(rng.uniform(-0.30, 0.30)), 2.5e-4, 2.5e-3)),
            "step_size": int(np.clip(winner["step_size"] + rng.choice([-1, 0, 1]), 1, 4)),
            "gamma": float(np.clip(winner["gamma"] + rng.uniform(-0.10, 0.10), 0.25, 0.88)),
            "aux_weight": float(np.clip(winner["aux_weight"] * np.exp(rng.uniform(-0.45, 0.45)), 0.02, 0.25)),
            "bpr_mix": float(np.clip(winner["bpr_mix"] + rng.uniform(-0.08, 0.08), 0.20, 0.68)),
            "half_life": float(np.clip(winner["half_life"] * np.exp(rng.uniform(-0.30, 0.30)), 3.0, 24.0)),
        })
    return configs


def append_progress(path, record):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    seed_all(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    Xt, Xv, Ht, Hv, total_dim = build_composite_features(data)
    yt = data["yt"]
    val_y = data["yv"]
    val_users = data["val_user"]
    base_total = int(data["field_dims"].sum())
    play = np.maximum(0.0, data["train_play"].astype(np.float32))
    duration = np.maximum(1.0, data["train_duration"].astype(np.float32))
    observed = np.minimum(play, duration)
    watch_target = np.log1p(observed / 1000.0).astype(np.float32)
    censored = (play >= duration).astype(np.bool_)
    pair_pos, pair_neg = build_pairs(data["train_user"], yt, args.seed + 73)

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    coarse_epochs = 4
    refine_epochs = 7
    final_epochs = args.epochs
    coarse_count = 56 if device.type == "cuda" else 36
    refine_count = 24 if device.type == "cuda" else 16
    repeats = 2
    if smoke_cap is not None:
        coarse_epochs = min(coarse_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
        coarse_count = 1
        refine_count = 1
        repeats = 1

    weight_cache = {}
    def get_weights(half_life):
        key = round(float(half_life), 6)
        if key not in weight_cache:
            weight_cache[key] = recency_weights(data["train_date"], half_life)
        return weight_cache[key]

    history = []
    coarse_summaries = []
    for config_id, config in enumerate(random_configs(args.seed, coarse_count)):
        values = []
        for repeat in range(repeats):
            probe_seed = args.seed + 1000 + config_id * 11 + repeat
            primary, _, checkpoints = train_composite(
                config, probe_seed, coarse_epochs, Xt, Ht, yt, watch_target, censored,
                Xv, Hv, val_users, val_y, get_weights(config["half_life"]),
                pair_pos, pair_neg, total_dim, evaluator, device, False
            )
            record = {"stage": "coarse", "config_id": config_id, "repeat": repeat,
                      "seed": probe_seed, "config": config, "primary": primary,
                      "checkpoints": checkpoints}
            history.append(record)
            append_progress(progress_path, {k: record[k] for k in ("stage", "config_id", "repeat", "seed", "config", "primary")})
            values.append(primary)
        coarse_summaries.append({"config": config, "mean_primary": float(np.mean(values)),
                                 "std_primary": float(np.std(values))})

    coarse_winner = max(coarse_summaries, key=lambda item: item["mean_primary"])["config"]
    refine_summaries = []
    for config_id, config in enumerate(refine_configs(coarse_winner, args.seed, refine_count)):
        values = []
        for repeat in range(repeats):
            probe_seed = args.seed + 10000 + config_id * 13 + repeat
            primary, _, checkpoints = train_composite(
                config, probe_seed, refine_epochs, Xt, Ht, yt, watch_target, censored,
                Xv, Hv, val_users, val_y, get_weights(config["half_life"]),
                pair_pos, pair_neg, total_dim, evaluator, device, False
            )
            record = {"stage": "refine", "config_id": config_id, "repeat": repeat,
                      "seed": probe_seed, "config": config, "primary": primary,
                      "checkpoints": checkpoints}
            history.append(record)
            append_progress(progress_path, {k: record[k] for k in ("stage", "config_id", "repeat", "seed", "config", "primary")})
            values.append(primary)
        refine_summaries.append({"config": config, "mean_primary": float(np.mean(values)),
                                 "std_primary": float(np.std(values))})

    winner = max(refine_summaries, key=lambda item: item["mean_primary"])["config"]
    parent_epochs = 1 if smoke_cap is not None else 2
    parent_scores = train_parent_reference(data["Xt"], yt, data["Xv"], base_total,
                                           args.seed + 30001, parent_epochs, device)

    member_scores = []
    member_records = []
    for member_id in range(2):
        member_seed = args.seed + 40001 + member_id
        primary, scores, checkpoints = train_composite(
            winner, member_seed, final_epochs, Xt, Ht, yt, watch_target, censored,
            Xv, Hv, val_users, val_y, get_weights(winner["half_life"]),
            pair_pos, pair_neg, total_dim, evaluator, device, True
        )
        assert not np.allclose(scores, parent_scores), "ensemble member unexpectedly equals parent predictions"
        if member_scores:
            assert not np.allclose(scores, member_scores[0]), "ensemble members unexpectedly have identical scores"
        member_scores.append(scores)
        record = {"stage": "final_member", "member": member_id, "seed": member_seed,
                  "config": winner, "primary": primary, "checkpoints": checkpoints}
        member_records.append(record)
        history.append(record)
        append_progress(progress_path, {"stage": "final_member", "member": member_id,
                                        "seed": member_seed, "primary": primary, "config": winner})

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    assert not np.allclose(final_scores, parent_scores), "ensemble unexpectedly equals parent predictions"
    final_metrics = metric_dict(evaluator, val_users, val_y, final_scores)
    append_progress(progress_path, {"stage": "ensemble", "member_primaries": [r["primary"] for r in member_records],
                                    "primary": final_metrics["primary"]})

    metrics_output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winning_config": winner,
        "member_primaries": [float(r["primary"]) for r in member_records],
        "coarse_summaries": coarse_summaries,
        "refine_summaries": refine_summaries,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_output, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", encoding="utf-8") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(final_scores):
            fh.write(f"{row_id},{val_users[row_id]},{data['val_video_output'][row_id]},{score:.9g}\n")


if __name__ == "__main__":
    main()
