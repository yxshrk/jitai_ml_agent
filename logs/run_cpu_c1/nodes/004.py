"""Sequence DeepFM with a searched aggressive regularization schedule package.

Uses the NPZ fast path when available, otherwise reads only train.csv and val.csv.
The sole change from the accepted parent is a jointly tuned package of embedding
and MLP dropout, accessed-row embedding L2, dense AdamW weight decay, rapid step
LR decay, and validation-GAUC checkpoint selection.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CONTEXT_DIMS = [24, 7, 2, 9, 8]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(m):
    return (float(m.get("GAUC", m.get("gauc"))),
            float(m.get("nDCG@5", m.get("ndcg5"))),
            float(m["primary"]))


def parse_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_csv_fallback(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def read_rows(path, training):
        result = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                item = {
                    "user": row.get("user_id", "0"),
                    "video": row.get("video_id", "0"),
                    "tab": row.get("tab", "0"),
                    "hourmin": parse_int(row.get("hourmin", 0)),
                    "date": parse_int(row.get("date", 19700101), 19700101),
                    "duration": max(0, parse_int(row.get("duration_ms", 0))),
                    "y": float(row.get("long_view", 0.0)),
                }
                if training:
                    item["play"] = max(0, parse_int(row.get("play_time_ms", 0)))
                result.append(item)
        return result

    tr_rows = read_rows(train_path, True)
    va_rows = read_rows(val_path, False)
    durations = np.asarray([r["duration"] for r in tr_rows], dtype=np.float64)
    if len(durations):
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    user_map = {v: i + 1 for i, v in enumerate(sorted({r["user"] for r in tr_rows}))}
    video_map = {v: i + 1 for i, v in enumerate(sorted({r["video"] for r in tr_rows}))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted({r["tab"] for r in tr_rows}))}
    field_dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                             len(tab_map) + 1, len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def make(rows, training):
        n = len(rows)
        x = np.zeros((n, 5), dtype=np.int64)
        raw_users = np.empty(n, dtype=object)
        raw_videos = np.empty(n, dtype=object)
        for i, r in enumerate(rows):
            local = [user_map.get(r["user"], 0), video_map.get(r["video"], 0), 0,
                     tab_map.get(r["tab"], 0),
                     int(np.searchsorted(edges, r["duration"], side="right")) + 1]
            x[i] = np.asarray(local, dtype=np.int64) + offsets
            raw_users[i] = r["user"]
            raw_videos[i] = r["video"]
        out = {
            "X": x,
            "y": np.asarray([r["y"] for r in rows], dtype=np.float32),
            "user": raw_users,
            "video_out": raw_videos,
            "duration_ms": np.asarray([r["duration"] for r in rows], dtype=np.float32),
            "hourmin": np.asarray([r["hourmin"] for r in rows], dtype=np.int64),
            "date": np.asarray([r["date"] for r in rows], dtype=np.int64),
            "field_dims": field_dims,
        }
        if training:
            out["play_time_ms"] = np.asarray([r["play"] for r in rows], dtype=np.float32)
        return out

    return make(tr_rows, True), make(va_rows, False), False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: np.asarray(tr_file[k]) for k in tr_file.files}
        va = {k: np.asarray(va_file[k]) for k in va_file.files}
        tr_file.close()
        va_file.close()
        tr["video_out"] = np.zeros(len(tr["y"]), dtype=np.int64)
        va["video_out"] = np.zeros(len(va["y"]), dtype=np.int64)
        return tr, va, True
    return load_csv_fallback(data_dir)


def date_to_day(values):
    vals = np.asarray(values)
    output = np.zeros(len(vals), dtype=np.int64)
    cache = {}
    for i, raw in enumerate(vals):
        key = str(raw)
        if key not in cache:
            text = key.split(".")[0].replace("-", "")
            try:
                if len(text) >= 8:
                    d = datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
                    cache[key] = d.toordinal()
                else:
                    cache[key] = parse_int(raw)
            except (ValueError, TypeError):
                cache[key] = parse_int(raw)
        output[i] = cache[key]
    return output


def make_time_arrays(data):
    hm = np.asarray(data["hourmin"], dtype=np.int64)
    hour = np.clip(hm // 100, 0, 23).astype(np.int64)
    day = date_to_day(data["date"])
    weekday = np.mod(day, 7).astype(np.int64)
    minute = day * 1440 + np.clip(hm // 100, 0, 23) * 60 + np.clip(hm % 100, 0, 59)
    return hour, weekday, minute


def build_causal_features(tr, va):
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    official_total = int(field_dims.sum())
    official_offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)
    tab_local_train = np.asarray(tr["X"][:, 3], dtype=np.int64) - official_offsets[3]
    tab_local_val = np.asarray(va["X"][:, 3], dtype=np.int64) - official_offsets[3]
    tr_hour, tr_weekday, tr_minute = make_time_arrays(tr)
    va_hour, va_weekday, va_minute = make_time_arrays(va)

    histories = defaultdict(lambda: deque(maxlen=12))
    previous_time = {}
    session_position = {}

    def process(data, hour, weekday, minute, tab_local):
        n = len(data["y"])
        hist = np.full((n, 12), -1, dtype=np.int32)
        gap = np.zeros(n, dtype=np.int64)
        position = np.zeros(n, dtype=np.int64)
        order = np.lexsort((np.arange(n, dtype=np.int64), minute))
        users = np.asarray(data["user"])
        authors = np.asarray(data["X"][:, 2], dtype=np.int64)
        gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.int64)
        for idx in order:
            u = users[idx].item() if isinstance(users[idx], np.generic) else users[idx]
            h = histories[u]
            if h:
                seq = list(h)
                hist[idx, -len(seq):] = seq
            now = int(minute[idx])
            if u in previous_time:
                delta = max(0, now - previous_time[u])
                gap[idx] = 1 + int(np.searchsorted(gap_edges, delta, side="right"))
                if delta <= 30:
                    pos = session_position[u] + 1
                else:
                    pos = 0
            else:
                gap[idx] = 0
                pos = 0
            position[idx] = min(pos, 7)
            previous_time[u] = now
            session_position[u] = pos
            h.append(int(authors[idx]))
        is_rand = (tab_local == 1).astype(np.int64)
        context = np.column_stack((hour, weekday, is_rand, gap, position)).astype(np.int64)
        context_offsets = official_total + np.concatenate(([0], np.cumsum(CONTEXT_DIMS)[:-1]))
        context += context_offsets
        xcat = np.concatenate((np.asarray(data["X"], dtype=np.int64), context), axis=1)
        return xcat, hist

    tr_x, tr_h = process(tr, tr_hour, tr_weekday, tr_minute, tab_local_train)
    va_x, va_h = process(va, va_hour, va_weekday, va_minute, tab_local_val)
    total_categories = official_total + int(sum(CONTEXT_DIMS)) + 1
    pad_id = total_categories - 1
    tr_h[tr_h < 0] = pad_id
    va_h[va_h < 0] = pad_id
    return tr_x, tr_h.astype(np.int64), va_x, va_h.astype(np.int64), total_categories, pad_id


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_categories, pad_id, embedding_dim=16,
                 mlp_dropout=0.3, embedding_dropout=0.15):
        super().__init__()
        self.pad_id = int(pad_id)
        self.emb = torch.nn.Embedding(total_categories, embedding_dim, padding_idx=pad_id)
        self.linear = torch.nn.Embedding(total_categories, 1, padding_idx=pad_id)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        field_count = 11
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(field_count * embedding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
        )
        self.deep_out = torch.nn.Linear(64, 1)
        self.watch_head = torch.nn.Linear(64, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.emb.weight[self.pad_id].zero_()
            self.linear.weight[self.pad_id].zero_()

    def forward(self, x, history):
        regular = self.emb(x)
        hist_emb = self.emb(history)
        mask = (history != self.pad_id).unsqueeze(-1)
        count = mask.sum(1).clamp_min(1)
        pooled = (hist_emb * mask).sum(1) / count
        fields = torch.cat((regular, pooled.unsqueeze(1)), dim=1)
        fields = self.embedding_dropout(fields)
        summed = fields.sum(1)
        fm = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        hist_linear = self.linear(history)
        hist_linear = (hist_linear * mask).sum((1, 2)) / count.squeeze(-1)
        linear = self.linear(x).sum((1, 2)) + hist_linear
        hidden = self.deep(fields.flatten(1))
        main = self.bias + linear + fm + self.deep_out(hidden).squeeze(1)
        watch = self.watch_head(hidden).squeeze(1)
        return main, watch

    def accessed_row_l2(self, x, history):
        ids = torch.unique(torch.cat((x.reshape(-1), history.reshape(-1))))
        ids = ids[ids != self.pad_id]
        if ids.numel() == 0:
            return self.bias.sum() * 0.0
        emb_penalty = self.emb(ids).square().sum(1).mean()
        linear_penalty = self.linear(ids).square().sum(1).mean()
        return emb_penalty + linear_penalty


def predict(model, x, history, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            end = min(start + 65536, len(x))
            xb = torch.from_numpy(x[start:end]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(history[start:end]).to(device=device, dtype=torch.long)
            logits, _ = model(xb, hb)
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def make_optimizer(model, config):
    decay = []
    no_decay = []
    embedding_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("emb.") or name.startswith("linear."):
            embedding_params.append(parameter)
        elif parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    groups = [
        {"params": embedding_params, "weight_decay": 0.0},
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config["lr"])


def prepare_watch_targets(tr):
    play = np.maximum(np.asarray(tr["play_time_ms"], dtype=np.float32), 0.0)
    duration = np.maximum(np.asarray(tr["duration_ms"], dtype=np.float32), 0.0)
    censored = (duration > 0) & (play >= duration)
    observed_log = np.log1p(np.where(censored, duration, play).astype(np.float64))
    center = float(observed_log.mean())
    scale = float(observed_log.std())
    if scale < 1e-6:
        scale = 1.0
    target = ((observed_log - center) / scale).astype(np.float32)
    return target, censored.astype(np.float32)


def train_member(member_seed, epochs, config, phase, probe_id, tr, va,
                 tr_x, tr_h, va_x, va_h, total_categories, pad_id,
                 evaluator, device):
    seed_everything(member_seed)
    model = SequenceDeepFM(
        total_categories,
        pad_id,
        mlp_dropout=config["mlp_dropout"],
        embedding_dropout=config["embedding_dropout"],
    ).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["lr_gamma"])
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(tr["y"])
    batch_size = 8192
    y = np.asarray(tr["y"], dtype=np.float32)
    watch_target, censor_float = prepare_watch_targets(tr)
    rng = np.random.RandomState(member_seed)
    best_gauc = -1.0
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    patience = 0
    epoch_rows = []

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        loss_sum = 0.0
        seen = 0
        current_lr = float(optimizer.param_groups[0]["lr"])
        for start in range(0, n, batch_size):
            ids = permutation[start:start + batch_size]
            xb = torch.from_numpy(tr_x[ids]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(tr_h[ids]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(y[ids]).to(device=device, dtype=torch.float32)
            wb = torch.from_numpy(watch_target[ids]).to(device=device, dtype=torch.float32)
            cb = torch.from_numpy(censor_float[ids]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits, watch_pred = model(xb, hb)
            main_loss = bce(logits, yb)
            ordinary = (watch_pred - wb).square()
            lower_bound = torch.relu(wb - watch_pred).square()
            aux_loss = ((1.0 - cb) * ordinary + cb * lower_bound).mean()
            row_l2 = model.accessed_row_l2(xb, hb)
            loss = main_loss + 0.1 * aux_loss + config["embedding_l2"] * row_l2
            loss.backward()
            optimizer.step()
            count = len(ids)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count

        scores = predict(model, va_x, va_h, device)
        metrics = evaluator(va["user"], np.asarray(va["y"]).astype(int), scores)
        gauc, ndcg5, primary = metric_values(metrics)
        epoch_rows.append({
            "phase": phase,
            "probe_id": probe_id,
            "member_seed": member_seed,
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "val_gauc": round(gauc, 6),
            "val_ndcg5": round(ndcg5, 6),
            "val_primary": round(primary, 6),
        })
        if gauc > best_gauc + 1e-7:
            best_gauc = gauc
            best_primary = primary
            best_scores = scores.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
        scheduler.step()
        if patience >= 4:
            break

    return {
        "scores": best_scores,
        "best_gauc": float(best_gauc),
        "best_primary": float(best_primary),
        "best_epoch": int(best_epoch),
        "epochs": epoch_rows,
    }


def generate_configs(seed, count):
    anchors = [
        {"mlp_dropout": 0.30, "embedding_dropout": 0.15, "embedding_l2": 3e-4,
         "weight_decay": 3e-4, "lr": 8e-4, "lr_gamma": 0.65, "step_size": 1},
        {"mlp_dropout": 0.35, "embedding_dropout": 0.10, "embedding_l2": 1e-3,
         "weight_decay": 1e-4, "lr": 7e-4, "lr_gamma": 0.70, "step_size": 1},
        {"mlp_dropout": 0.40, "embedding_dropout": 0.20, "embedding_l2": 3e-4,
         "weight_decay": 1e-3, "lr": 1e-3, "lr_gamma": 0.55, "step_size": 2},
        {"mlp_dropout": 0.28, "embedding_dropout": 0.08, "embedding_l2": 3e-3,
         "weight_decay": 3e-5, "lr": 6e-4, "lr_gamma": 0.75, "step_size": 1},
        {"mlp_dropout": 0.45, "embedding_dropout": 0.25, "embedding_l2": 1e-4,
         "weight_decay": 3e-4, "lr": 1.2e-3, "lr_gamma": 0.50, "step_size": 2},
        {"mlp_dropout": 0.32, "embedding_dropout": 0.18, "embedding_l2": 1e-3,
         "weight_decay": 1e-3, "lr": 5e-4, "lr_gamma": 0.80, "step_size": 1},
    ]
    configs = [dict(x) for x in anchors[:count]]
    rng = np.random.RandomState(seed + 731)
    while len(configs) < count:
        configs.append({
            "mlp_dropout": float(rng.uniform(0.25, 0.50)),
            "embedding_dropout": float(rng.uniform(0.05, 0.30)),
            "embedding_l2": float(10.0 ** rng.uniform(-4.5, -2.4)),
            "weight_decay": float(10.0 ** rng.uniform(-5.0, -2.8)),
            "lr": float(10.0 ** rng.uniform(-3.35, -2.82)),
            "lr_gamma": float(rng.uniform(0.45, 0.82)),
            "step_size": int(rng.choice([1, 1, 1, 2, 2])),
        })
    return configs


def clean_config(config):
    return {
        "mlp_dropout": round(float(config["mlp_dropout"]), 8),
        "embedding_dropout": round(float(config["embedding_dropout"]), 8),
        "embedding_l2": float(config["embedding_l2"]),
        "weight_decay": float(config["weight_decay"]),
        "lr": float(config["lr"]),
        "lr_gamma": round(float(config["lr_gamma"]), 8),
        "step_size": int(config["step_size"]),
    }


def append_progress(path, payload):
    with open(path, "a") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_epochs = args.epochs
    probe_epochs = min(8, args.epochs)
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_active = smoke is not None
    if smoke_active:
        cap = max(1, int(smoke))
        final_epochs = min(final_epochs, cap)
        probe_epochs = min(probe_epochs, cap)

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    tr_x, tr_h, va_x, va_h, total_categories, pad_id = build_causal_features(tr, va)
    progress_path = os.path.join(args.out_dir, "progress.log")
    initial_count = 2 if smoke_active else 44
    refine_count = 1 if smoke_active else 6
    configs = generate_configs(args.seed, initial_count)
    probe_records = []
    all_epoch_history = []

    for probe_id, config in enumerate(configs):
        result = train_member(
            args.seed + 1000, probe_epochs, config, "search", probe_id,
            tr, va, tr_x, tr_h, va_x, va_h, total_categories, pad_id,
            evaluator, device)
        record = {
            "phase": "search",
            "probe_id": probe_id,
            "seed": args.seed + 1000,
            "config": clean_config(config),
            "best_gauc": result["best_gauc"],
            "best_primary": result["best_primary"],
            "best_epoch": result["best_epoch"],
        }
        probe_records.append(record)
        all_epoch_history.extend(result["epochs"])
        append_progress(progress_path, record)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked = sorted(probe_records, key=lambda r: (r["best_gauc"], r["best_primary"]), reverse=True)
    top_ids = [r["probe_id"] for r in ranked[:refine_count]]
    refinement = {}
    for probe_id in top_ids:
        config = configs[probe_id]
        result = train_member(
            args.seed + 2000, probe_epochs, config, "refine", probe_id,
            tr, va, tr_x, tr_h, va_x, va_h, total_categories, pad_id,
            evaluator, device)
        first = probe_records[probe_id]
        mean_gauc = 0.5 * (first["best_gauc"] + result["best_gauc"])
        mean_primary = 0.5 * (first["best_primary"] + result["best_primary"])
        record = {
            "phase": "refine",
            "probe_id": probe_id,
            "seed": args.seed + 2000,
            "config": clean_config(config),
            "best_gauc": result["best_gauc"],
            "best_primary": result["best_primary"],
            "best_epoch": result["best_epoch"],
            "two_seed_mean_gauc": mean_gauc,
            "two_seed_mean_primary": mean_primary,
        }
        refinement[probe_id] = record
        probe_records.append(record)
        all_epoch_history.extend(result["epochs"])
        append_progress(progress_path, record)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    winner_id = max(top_ids, key=lambda i: (
        refinement[i]["two_seed_mean_gauc"],
        refinement[i]["two_seed_mean_primary"]))
    winning_config = configs[winner_id]

    member_scores = []
    member_summaries = []
    for member_index, member_seed in enumerate([args.seed, args.seed + 1]):
        result = train_member(
            member_seed, final_epochs, winning_config, "final", member_index,
            tr, va, tr_x, tr_h, va_x, va_h, total_categories, pad_id,
            evaluator, device)
        member_scores.append(result["scores"])
        all_epoch_history.extend(result["epochs"])
        summary = {
            "seed": member_seed,
            "best_gauc": result["best_gauc"],
            "best_primary": result["best_primary"],
            "best_epoch": result["best_epoch"],
        }
        member_summaries.append(summary)
        append_progress(progress_path, {"phase": "final", **summary})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
    metrics = evaluator(va["user"], np.asarray(va["y"]).astype(int), final_scores)
    gauc, ndcg5, primary = metric_values(metrics)
    payload = {
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "history": all_epoch_history,
        "probes": probe_records,
        "selected_probe_id": winner_id,
        "selected_config": clean_config(winning_config),
        "selection_metric": "two_seed_mean_validation_gauc",
        "members": member_summaries,
        "ensemble": "mean_logit",
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(final_scores):
            user = va["user"][i]
            video = va["video_out"][i]
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, format(float(score), ".9g")])


if __name__ == "__main__":
    main()
