"""Full Sequence DeepFM composite with causal history, session context, watch-time auxiliary, and logit ensemble."""
import argparse
import collections
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate as official_evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc", 0.0))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5", 0.0))),
        "primary": float(result["primary"]),
    }


def parse_date_value(value):
    text = str(value).strip()
    try:
        if len(text) >= 8 and text[:8].isdigit():
            return datetime.datetime.strptime(text[:8], "%Y%m%d").date()
        return datetime.date.fromisoformat(text[:10])
    except Exception:
        return None


def temporal_arrays(date_values, hourmin_values):
    dates = np.asarray(date_values).astype(str)
    hourmins = np.asarray(hourmin_values)
    unique = np.unique(dates)
    parsed = {value: parse_date_value(value) for value in unique}
    if all(parsed[value] is not None for value in unique):
        day_number = np.asarray([parsed[value].toordinal() for value in dates], dtype=np.int64)
        weekday = np.asarray([parsed[value].weekday() for value in dates], dtype=np.int64)
    else:
        mapping = {value: i for i, value in enumerate(sorted(unique))}
        day_number = np.asarray([mapping[value] for value in dates], dtype=np.int64)
        weekday = day_number % 7
    hm = np.asarray(hourmins)
    hours = np.zeros(len(hm), dtype=np.int64)
    minutes = np.zeros(len(hm), dtype=np.int64)
    for i, raw in enumerate(hm):
        try:
            value = int(float(raw))
        except Exception:
            value = 0
        value = max(0, value)
        hours[i] = min(23, value // 100)
        minutes[i] = min(59, value % 100)
    timestamp_minutes = day_number * 1440 + hours * 60 + minutes
    return day_number, weekday, hours, timestamp_minutes


def load_csv_split(data_dir):
    def read_rows(path, validation):
        result = []
        with open(path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                item = {
                    "user": row["user_id"],
                    "video": row["video_id"],
                    "tab": row["tab"],
                    "duration": float(row["duration_ms"]),
                    "date": row["date"],
                    "hourmin": row["hourmin"],
                    "y": float(row["long_view"]),
                }
                if not validation:
                    item["play"] = float(row["play_time_ms"])
                    item["click"] = float(row["click"])
                result.append(item)
        return result

    train_rows = read_rows(os.path.join(data_dir, "train.csv"), False)
    val_rows = read_rows(os.path.join(data_dir, "val.csv"), True)
    user_map = {v: i + 1 for i, v in enumerate(sorted({r["user"] for r in train_rows}))}
    video_map = {v: i + 1 for i, v in enumerate(sorted({r["video"] for r in train_rows}))}
    tab_map = {v: i + 1 for i, v in enumerate(sorted({r["tab"] for r in train_rows}))}
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, len(cuts) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1]))

    def encode(rows, validation):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        x[:, 0] = [user_map.get(r["user"], 0) for r in rows]
        x[:, 1] = [video_map.get(r["video"], 0) for r in rows]
        x[:, 2] = 0
        x[:, 3] = [tab_map.get(r["tab"], 0) for r in rows]
        x[:, 4] = np.searchsorted(cuts, [r["duration"] for r in rows], side="right")
        x += offsets
        output = {
            "X": x.astype(np.int32),
            "y": np.asarray([r["y"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user"] for r in rows]),
            "video": np.asarray([r["video"] for r in rows]),
            "date": np.asarray([r["date"] for r in rows]),
            "hourmin": np.asarray([r["hourmin"] for r in rows]),
            "duration_ms": np.asarray([r["duration"] for r in rows], dtype=np.float32),
            "field_dims": dims,
        }
        if not validation:
            output["play_time_ms"] = np.asarray([r["play"] for r in rows], dtype=np.float32)
            output["click"] = np.asarray([r["click"] for r in rows], dtype=np.float32)
        return output

    return encode(train_rows, False), encode(val_rows, True), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_file = np.load(train_path)
        val_file = np.load(val_path)
        train = {key: train_file[key] for key in train_file.files}
        val = {key: val_file[key] for key in val_file.files}
        train_file.close()
        val_file.close()
        video_offset = int(np.asarray(train["field_dims"])[0])
        val["video"] = np.asarray(val["X"][:, 1], dtype=np.int64) - video_offset
        return train, val, True
    return load_csv_split(data_dir)


def build_sequence_features(train, val, history_length=12):
    train_x = np.asarray(train["X"], dtype=np.int64)
    val_x = np.asarray(val["X"], dtype=np.int64)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    base_total = int(field_dims.sum())
    tab_offset = int(field_dims[:3].sum())
    author_offset = int(field_dims[:2].sum())

    all_users = np.concatenate([np.asarray(train["user"]).astype(str),
                                np.asarray(val["user"]).astype(str)])
    all_authors = np.concatenate([train_x[:, 2], val_x[:, 2]]).astype(np.int64)
    all_dates = np.concatenate([np.asarray(train["date"]), np.asarray(val["date"])])
    all_hourmin = np.concatenate([np.asarray(train["hourmin"]), np.asarray(val["hourmin"])])
    day_number, weekday, hour, timestamp = temporal_arrays(all_dates, all_hourmin)
    all_tab = np.concatenate([train_x[:, 3], val_x[:, 3]]) - tab_offset
    is_rand = (all_tab == 1).astype(np.int64)

    count = len(all_users)
    row_order = np.arange(count, dtype=np.int64)
    chronological = np.lexsort((row_order, timestamp, all_users))
    histories = np.full((count, history_length), author_offset, dtype=np.int32)
    gap_bucket = np.zeros(count, dtype=np.int64)
    session_position = np.zeros(count, dtype=np.int64)
    author_queues = {}
    previous_time = {}
    positions = {}
    gap_edges = np.asarray([1, 5, 15, 30, 60, 180, 720], dtype=np.int64)

    for index in chronological:
        user = all_users[index]
        queue = author_queues.get(user)
        if queue is None:
            queue = collections.deque(maxlen=history_length)
            author_queues[user] = queue
        if queue:
            values = list(queue)
            histories[index, history_length - len(values):] = values
        if user not in previous_time:
            gap_bucket[index] = 0
            positions[user] = 0
        else:
            gap = max(0, int(timestamp[index] - previous_time[user]))
            gap_bucket[index] = 1 + int(np.searchsorted(gap_edges, gap, side="right"))
            if gap > 30:
                positions[user] = 0
            else:
                positions[user] = positions[user] + 1
        session_position[index] = min(15, positions[user])
        previous_time[user] = int(timestamp[index])
        author = int(all_authors[index])
        if author != author_offset:
            queue.append(author)

    context_dims = np.asarray([24, 7, 2, 9, 16], dtype=np.int64)
    context_offsets = base_total + np.concatenate(([0], np.cumsum(context_dims)[:-1]))
    context = np.stack([hour, weekday, is_rand, gap_bucket, session_position], axis=1)
    context = context + context_offsets.reshape(1, -1)
    full_x = np.concatenate([np.concatenate([train_x, val_x], axis=0), context], axis=1)
    n_train = len(train_x)
    train_age = day_number[:n_train].max() - day_number[:n_train]
    return {
        "train_x": full_x[:n_train].astype(np.int64),
        "val_x": full_x[n_train:].astype(np.int64),
        "train_history": histories[:n_train].astype(np.int64),
        "val_history": histories[n_train:].astype(np.int64),
        "total_dim": int(base_total + context_dims.sum()),
        "author_padding": author_offset,
        "train_age": train_age.astype(np.float32),
    }


class SequenceDeepFM(torch.nn.Module):
    def __init__(self, total_dim, author_padding, fields=10, embedding_dim=16,
                 hidden=128, dropout=0.25):
        super().__init__()
        self.author_padding = int(author_padding)
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.embedding_dropout = torch.nn.Dropout(dropout)
        width = (fields + 1) * embedding_dim
        second = max(32, hidden // 2)
        self.deep1 = torch.nn.Linear(width, hidden)
        self.deep2 = torch.nn.Linear(hidden, second)
        self.deep_out = torch.nn.Linear(second, 1)
        self.watch_out = torch.nn.Linear(second, 1)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.deep_out.weight, std=0.01)
        torch.nn.init.zeros_(self.deep_out.bias)
        torch.nn.init.normal_(self.watch_out.weight, std=0.01)
        torch.nn.init.zeros_(self.watch_out.bias)

    def forward(self, x, history):
        field_embeddings = self.embedding_dropout(self.embedding(x))
        history_embeddings = self.embedding(history)
        history_mask = (history != self.author_padding).float().unsqueeze(-1)
        history_sum = (history_embeddings * history_mask).sum(dim=1)
        history_count = history_mask.sum(dim=1).clamp_min(1.0)
        history_mean = self.embedding_dropout(history_sum / history_count)
        combined = torch.cat([field_embeddings, history_mean.unsqueeze(1)], dim=1)
        summed = combined.sum(dim=1)
        fm = 0.5 * (summed.square() - combined.square().sum(dim=1)).sum(dim=1)
        linear = self.bias + self.linear(x).sum(dim=(1, 2))
        hidden = self.dropout(torch.relu(self.deep1(combined.flatten(1))))
        hidden = self.dropout(torch.relu(self.deep2(hidden)))
        main_logit = linear + fm + self.deep_out(hidden).squeeze(1)
        watch_prediction = torch.nn.functional.softplus(self.watch_out(hidden).squeeze(1))
        return main_logit, watch_prediction


def predict(model, x, history, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(DEVICE, non_blocking=True)
            hb = history[start:start + batch_size].to(DEVICE, non_blocking=True)
            logits, _ = model(xb, hb)
            chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def train_model(train_x, train_history, labels, watch_target, watch_censored,
                val_x, val_history, val_users, val_labels, train_age, total_dim,
                author_padding, config, epochs, seed):
    seed_all(seed)
    model = SequenceDeepFM(
        total_dim=total_dim,
        author_padding=author_padding,
        embedding_dim=16,
        hidden=int(config["hidden"]),
        dropout=float(config["dropout"]),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = 16384 if DEVICE.type == "cuda" else 8192
    eval_batch_size = 65536 if DEVICE.type == "cuda" else 32768
    recency = torch.exp(-math.log(2.0) * train_age / float(config["half_life"]))
    count = len(labels)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    curve = []

    for epoch in range(epochs):
        lr = float(config["lr"]) * float(config["gamma"]) ** (epoch // int(config["step_size"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        permutation = torch.randperm(count)
        loss_value = 0.0
        for start in range(0, count, batch_size):
            indices = permutation[start:start + batch_size]
            xb = train_x[indices].to(DEVICE, non_blocking=True)
            hb = train_history[indices].to(DEVICE, non_blocking=True)
            yb = labels[indices].to(DEVICE, non_blocking=True)
            wb = recency[indices].to(DEVICE, non_blocking=True)
            target = watch_target[indices].to(DEVICE, non_blocking=True)
            censored = watch_censored[indices].to(DEVICE, non_blocking=True)
            logits, watch_prediction = model(xb, hb)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none")
            point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
            residual = watch_prediction - target
            uncensored_loss = residual.square()
            censored_loss = torch.relu(target - watch_prediction).square()
            auxiliary = torch.where(censored, censored_loss, uncensored_loss)
            auxiliary = (auxiliary * wb).sum() / wb.sum().clamp_min(1e-8)
            loss = point_loss + float(config["aux_weight"]) * auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())

        scores = predict(model, val_x, val_history, eval_batch_size)
        metrics = metric_values(official_evaluate(val_users, val_labels, scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_value, 6),
            "lr": lr,
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-10:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_metrics = metrics

    del model, optimizer
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def coarse_configs(seed, count):
    rng = np.random.default_rng(seed + 4109)
    configs = []
    for _ in range(count):
        configs.append({
            "hidden": int(rng.choice([64, 96, 128, 160])),
            "dropout": float(rng.uniform(0.12, 0.40)),
            "weight_decay": float(10.0 ** rng.uniform(-4.7, -2.8)),
            "lr": float(10.0 ** rng.uniform(-3.45, -2.72)),
            "gamma": float(rng.choice([0.32, 0.42, 0.52, 0.64, 0.74])),
            "step_size": int(rng.choice([1, 2, 3])),
            "half_life": float(rng.choice([3.5, 5.0, 7.0, 10.0, 14.0])),
            "aux_weight": float(rng.choice([0.03, 0.05, 0.08, 0.12, 0.18])),
        })
    return configs


def refined_configs(winner, seed, count):
    rng = np.random.default_rng(seed + 6211)
    results = [dict(winner)]
    hidden_values = np.asarray([64, 96, 128, 160])
    for _ in range(count - 1):
        hidden_index = int(np.argmin(np.abs(hidden_values - int(winner["hidden"]))))
        hidden_index = int(np.clip(hidden_index + rng.choice([-1, 0, 0, 1]), 0,
                                   len(hidden_values) - 1))
        results.append({
            "hidden": int(hidden_values[hidden_index]),
            "dropout": float(np.clip(float(winner["dropout"]) + rng.normal(0, 0.035), 0.1, 0.45)),
            "weight_decay": float(np.clip(float(winner["weight_decay"]) * math.exp(rng.normal(0, 0.45)), 1e-5, 3e-3)),
            "lr": float(np.clip(float(winner["lr"]) * math.exp(rng.normal(0, 0.22)), 3e-4, 2.5e-3)),
            "gamma": float(np.clip(float(winner["gamma"]) + rng.normal(0, 0.06), 0.25, 0.82)),
            "step_size": int(np.clip(int(winner["step_size"]) + rng.choice([-1, 0, 0, 1]), 1, 3)),
            "half_life": float(np.clip(float(winner["half_life"]) * math.exp(rng.normal(0, 0.18)), 3.0, 16.0)),
            "aux_weight": float(np.clip(float(winner["aux_weight"]) * math.exp(rng.normal(0, 0.30)), 0.02, 0.22)),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_all(args.seed)

    train, val, fast_path = load_data(args.data_dir)
    evaluator = official_evaluate
    if not fast_path:
        from harness.evaluate_provisional import evaluate as provisional_evaluate
        evaluator = provisional_evaluate

    sequence = build_sequence_features(train, val, history_length=12)
    train_x = torch.from_numpy(sequence["train_x"])
    val_x = torch.from_numpy(sequence["val_x"])
    train_history = torch.from_numpy(sequence["train_history"])
    val_history = torch.from_numpy(sequence["val_history"])
    labels = torch.from_numpy(np.asarray(train["y"], dtype=np.float32))
    val_labels = np.asarray(val["y"], dtype=np.int64)
    val_users = np.asarray(val["user"])
    train_age = torch.from_numpy(sequence["train_age"])

    play = np.asarray(train["play_time_ms"], dtype=np.float32)
    duration = np.asarray(train["duration_ms"], dtype=np.float32)
    clipped_watch = np.minimum(np.maximum(play, 0.0), np.maximum(duration, 0.0))
    watch_target = torch.from_numpy(np.log1p(clipped_watch / 1000.0).astype(np.float32))
    watch_censored = torch.from_numpy((play >= np.maximum(duration, 1.0)).astype(np.bool_))

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke = int(smoke_text) if smoke_text is not None else None
    coarse_count, refine_count, ensemble_count = (48, 24, 3) if smoke is None else (2, 1, 1)
    coarse_epochs = 3 if smoke is None else min(3, smoke)
    refine_epochs = 5 if smoke is None else min(5, smoke)
    final_epochs = max(1, args.epochs)
    if smoke is not None:
        final_epochs = min(final_epochs, smoke)

    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    coarse_results = []
    for probe, config in enumerate(coarse_configs(args.seed, coarse_count), 1):
        metrics, scores, curve = train_model(
            train_x, train_history, labels, watch_target, watch_censored,
            val_x, val_history, val_users, val_labels, train_age,
            sequence["total_dim"], sequence["author_padding"], config,
            coarse_epochs, args.seed + 700,
        )
        entry = {"stage": "coarse", "probe": probe, "config": config,
                 **metrics, "curve": curve}
        history.append(entry)
        coarse_results.append((metrics["primary"], config))
        with open(progress_path, "a") as handle:
            handle.write(json.dumps({"stage": "coarse", "probe": probe,
                                     "config": config, "primary": metrics["primary"]}) + "\n")
        del scores

    coarse_results.sort(key=lambda item: item[0], reverse=True)
    refine_results = []
    for probe, config in enumerate(refined_configs(coarse_results[0][1], args.seed, refine_count), 1):
        metrics, scores, curve = train_model(
            train_x, train_history, labels, watch_target, watch_censored,
            val_x, val_history, val_users, val_labels, train_age,
            sequence["total_dim"], sequence["author_padding"], config,
            refine_epochs, args.seed + 1100,
        )
        entry = {"stage": "refine", "probe": probe, "config": config,
                 **metrics, "curve": curve}
        history.append(entry)
        refine_results.append((metrics["primary"], config))
        with open(progress_path, "a") as handle:
            handle.write(json.dumps({"stage": "refine", "probe": probe,
                                     "config": config, "primary": metrics["primary"]}) + "\n")
        del scores

    refine_results.sort(key=lambda item: item[0], reverse=True)
    winner = refine_results[0][1]
    member_scores = []
    member_metrics = []
    member_seeds = [args.seed + i for i in range(ensemble_count)]
    for run_seed in member_seeds:
        metrics, scores, curve = train_model(
            train_x, train_history, labels, watch_target, watch_censored,
            val_x, val_history, val_users, val_labels, train_age,
            sequence["total_dim"], sequence["author_padding"], winner,
            final_epochs, run_seed,
        )
        member_scores.append(scores)
        member_metrics.append(metrics)
        history.append({"stage": "final", "seed": run_seed, "config": winner,
                        **metrics, "curve": curve})
        with open(progress_path, "a") as handle:
            handle.write(json.dumps({"stage": "final", "seed": run_seed,
                                     "config": winner, "primary": metrics["primary"]}) + "\n")

    if len(member_scores) == 1:
        chosen_scores = member_scores[0]
        selected = "single"
    else:
        chosen_scores = np.mean(np.stack(member_scores, axis=0), axis=0)
        selected = "mean_logit_ensemble"
        ensemble_metrics = metric_values(evaluator(val_users, val_labels, chosen_scores))
        history.append({"stage": "ensemble", "method": "mean_logit",
                        "seeds": member_seeds, **ensemble_metrics})

    final_metrics = metric_values(evaluator(val_users, val_labels, chosen_scores))
    payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected": selected,
        "winning_config": winner,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(payload, handle)

    videos = np.asarray(val.get("video", np.zeros(len(chosen_scores), dtype=np.int64)))
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, score in enumerate(chosen_scores):
            writer.writerow([row_id, val_users[row_id], videos[row_id], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
