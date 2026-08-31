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
    def __init__(self, total_dim, num_current_fields=8, embedding_dim=12,
                 hidden_dim=48, dropout=0.05):
        super().__init__()
        self.num_current_fields = num_current_fields
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear((num_current_fields + 1) * embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        for layer in self.deep:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x, history):
        current = self.embedding(x)
        mask = history.ge(0)
        safe_history = history.clamp_min(0)
        history_embeddings = self.embedding(safe_history)
        history_embeddings = history_embeddings * mask.unsqueeze(-1).to(history_embeddings.dtype)
        denominator = mask.sum(1, keepdim=True).clamp_min(1).to(history_embeddings.dtype)
        pooled_history = history_embeddings.sum(1) / denominator
        pooled_history = pooled_history * mask.any(1, keepdim=True).to(pooled_history.dtype)

        fields = torch.cat((current, pooled_history.unsqueeze(1)), dim=1)
        summed = fields.sum(1)
        fm = 0.5 * (summed.square() - fields.square().sum(1)).sum(1)
        linear = self.linear(x).sum((1, 2))
        deep = self.deep(fields.flatten(1)).squeeze(1)
        return self.bias + linear + fm + deep


def weekday_from_dates(values):
    result = np.zeros(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        key = int(value)
        if key not in cache:
            text = str(key)
            try:
                cache[key] = datetime.date(
                    int(text[:4]), int(text[4:6]), int(text[6:8])
                ).weekday()
            except (ValueError, IndexError):
                cache[key] = 0
        result[i] = cache[key]
    return result


def hour_from_hourmin(values):
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return values.copy()
    if int(values.max()) <= 2359:
        hours = values // 100
    else:
        hours = values // 60
    return np.clip(hours, 0, 23).astype(np.int64)


def add_context_fields(x, field_dims, hourmin, dates):
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    tab_raw = x[:, 3].astype(np.int64) - offsets[3]
    hour = hour_from_hourmin(hourmin)
    weekday = weekday_from_dates(dates)
    is_rand = (tab_raw == 1).astype(np.int64)
    base_total = int(np.sum(field_dims))
    context = np.column_stack((
        hour + base_total,
        weekday + base_total + 24,
        is_rand + base_total + 31,
    )).astype(np.int64)
    return np.concatenate((x.astype(np.int64), context), axis=1), base_total + 33


def build_causal_histories(train_users, train_authors, val_users, val_authors, length=12):
    state = defaultdict(lambda: deque(maxlen=length))
    train_history = np.full((len(train_users), length), -1, dtype=np.int32)
    val_history = np.full((len(val_users), length), -1, dtype=np.int32)

    for i in range(len(train_users)):
        value = train_users[i]
        user = value.item() if isinstance(value, np.generic) else value
        prior = state[user]
        if prior:
            train_history[i, :len(prior)] = np.fromiter(
                prior, dtype=np.int32, count=len(prior)
            )
        prior.append(int(train_authors[i]))

    for i in range(len(val_users)):
        value = val_users[i]
        user = value.item() if isinstance(value, np.generic) else value
        prior = state[user]
        if prior:
            val_history[i, :len(prior)] = np.fromiter(
                prior, dtype=np.int32, count=len(prior)
            )
        prior.append(int(val_authors[i]))

    return train_history, val_history


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "hourmin": int(float(row["hourmin"])),
                "date": int(float(row["date"])),
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })
    return rows


def prepare_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))
    fields = ("user_id", "video_id", "author_id", "tab")
    mappings = {}
    dimensions = []
    for field in fields:
        mapping = {}
        for row in train_rows:
            value = row[field]
            if value not in mapping:
                mapping[value] = len(mapping)
        mappings[field] = mapping
        dimensions.append(len(mapping) + 1)

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    dimensions.append(10)
    offsets = np.concatenate(([0], np.cumsum(dimensions[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int32)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                mapping = mappings[field]
                x[i, j] = offsets[j] + mapping.get(row[field], len(mapping))
            x[i, 4] = offsets[4] + int(np.searchsorted(
                quantiles, row["duration_ms"], side="right"
            ))
        return {
            "X": x,
            "y": np.asarray([r["long_view"] for r in rows], dtype=np.float32),
            "user": np.asarray([r["user_id"] for r in rows]),
            "video": np.asarray([r["video_id"] for r in rows]),
            "hourmin": np.asarray([r["hourmin"] for r in rows], dtype=np.int32),
            "date": np.asarray([r["date"] for r in rows], dtype=np.int32),
            "field_dims": np.asarray(dimensions, dtype=np.int64),
        }

    return encode(train_rows), encode(val_rows)


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        with np.load(train_path) as source:
            train = {
                key: source[key]
                for key in ("X", "y", "user", "hourmin", "date", "field_dims")
            }
        with np.load(val_path) as source:
            val = {
                key: source[key]
                for key in ("X", "y", "user", "hourmin", "date")
            }
        field_dims = np.asarray(train["field_dims"], dtype=np.int64)
        video_offset = int(field_dims[0])
        val["video"] = val["X"][:, 1].astype(np.int64) - video_offset
        return train, val, True
    train, val = prepare_csv_data(data_dir)
    return train, val, False


def predict(model, x, history, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            xb = torch.from_numpy(x[start:stop]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(history[start:stop]).to(device=device, dtype=torch.long)
            parts.append(model(xb, hb).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def make_frequency_weights(train_x, total_dim, power):
    weights = np.ones(total_dim, dtype=np.float32)
    if power <= 0.0:
        return weights
    for column in (0, 2):
        ids = np.asarray(train_x[:, column], dtype=np.int64)
        unique_ids, counts = np.unique(ids, return_counts=True)
        raw = np.power(counts.astype(np.float64), -float(power))
        occurrence_mean = float(np.sum(counts * raw) / np.sum(counts))
        raw = raw / max(occurrence_mean, 1e-12)
        raw = np.clip(raw, 0.2, 25.0)
        clipped_mean = float(np.sum(counts * raw) / np.sum(counts))
        raw = raw / max(clipped_mean, 1e-12)
        weights[unique_ids] = raw.astype(np.float32)
    return weights


def build_tail_pool(encoded_users, labels):
    encoded_users = np.asarray(encoded_users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)
    order = np.argsort(encoded_users, kind="stable")
    sorted_users = encoded_users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        rows = order[left:right]
        pos = rows[labels[rows] > 0.5]
        neg = rows[labels[rows] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos.astype(np.int64, copy=False))
            negatives.append(neg.astype(np.int64, copy=False))
    return positives, negatives


def sample_tail_rows(positives, negatives, rng, group_count=96,
                     positive_count=2, candidate_count=24):
    pool_size = len(positives)
    chosen = rng.integers(0, pool_size, size=group_count)
    pos_rows = np.empty((group_count, positive_count), dtype=np.int64)
    neg_rows = np.empty((group_count, candidate_count), dtype=np.int64)
    neg_mask = np.zeros((group_count, candidate_count), dtype=np.bool_)
    for i, group in enumerate(chosen):
        pos = positives[int(group)]
        neg = negatives[int(group)]
        pos_rows[i] = pos[rng.integers(0, len(pos), size=positive_count)]
        take = min(candidate_count, len(neg))
        if take == len(neg):
            selected = neg
        else:
            selected = neg[rng.choice(len(neg), size=take, replace=False)]
        neg_rows[i, :take] = selected
        neg_rows[i, take:] = selected[0]
        neg_mask[i, :take] = True
    return pos_rows, neg_rows, neg_mask


def smooth_top_tail_loss(model, train_x, train_history, positives, negatives,
                         rng, device, top_m=8, tau=0.25, margin=0.25):
    pos_rows, neg_rows, neg_mask = sample_tail_rows(
        positives, negatives, rng, group_count=96,
        positive_count=2, candidate_count=24
    )
    groups, pos_count = pos_rows.shape
    candidate_count = neg_rows.shape[1]
    all_rows = np.concatenate((pos_rows.reshape(-1), neg_rows.reshape(-1)))
    xb = torch.from_numpy(train_x[all_rows]).to(device=device, dtype=torch.long)
    hb = torch.from_numpy(train_history[all_rows]).to(device=device, dtype=torch.long)
    logits = model(xb, hb)
    split = groups * pos_count
    positive_logits = logits[:split].reshape(groups, pos_count)
    negative_logits = logits[split:].reshape(groups, candidate_count)
    mask = torch.from_numpy(neg_mask).to(device=device)
    negative_logits = negative_logits.masked_fill(~mask, -1.0e9)
    selected_count = min(int(top_m), candidate_count)
    top_values = torch.topk(negative_logits, k=selected_count, dim=1).values
    valid_top = top_values > -1.0e8
    scaled = (top_values / float(tau)).masked_fill(~valid_top, -1.0e9)
    tail_weights = torch.softmax(scaled, dim=1)
    tail_score = (tail_weights * top_values.masked_fill(~valid_top, 0.0)).sum(1)
    return torch.nn.functional.softplus(
        float(margin) + tail_score.unsqueeze(1) - positive_logits
    ).mean()


def reset_random_state(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_variant(train_x, train_history, labels, val_x, val_history,
                  val_users, val_labels, total_dim, device, evaluate,
                  seed, epochs, reg_lambda, frequency_power,
                  frequency_weights, retain_scores, tail_enabled,
                  tail_pool, tail_weight=0.10, top_m=8, tau=0.25,
                  margin=0.25):
    reset_random_state(seed)
    model = CausalHistoryDeepFM(
        total_dim=total_dim,
        num_current_fields=8,
        embedding_dim=12,
        hidden_dim=48,
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00115, weight_decay=3e-6)
    criterion = torch.nn.BCEWithLogitsLoss()
    weight_tensor = torch.from_numpy(frequency_weights).to(
        device=device, dtype=torch.float32
    )
    n = len(labels)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tail_rng = np.random.default_rng(seed + 104729)
    positives, negatives = tail_pool
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    trajectory = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator).numpy()
        loss_sum = 0.0
        bce_sum = 0.0
        reg_sum = 0.0
        tail_sum = 0.0
        tail_coefficient_sum = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(train_x[indices]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(train_history[indices]).to(device=device, dtype=torch.long)
            yb = torch.from_numpy(labels[indices]).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, hb)
            bce = criterion(logits, yb)
            if reg_lambda > 0.0:
                user_ids = xb[:, 0]
                author_ids = xb[:, 2]
                user_norm = model.embedding(user_ids).square().sum(1)
                author_norm = model.embedding(author_ids).square().sum(1)
                adaptive_reg = 0.5 * (
                    user_norm * weight_tensor[user_ids] +
                    author_norm * weight_tensor[author_ids]
                ).mean()
            else:
                adaptive_reg = bce.new_zeros(())

            progress = float(epoch) + float(start) / max(float(n), 1.0)
            ramp = min(1.0, max(0.0, (progress - 0.5) / 2.0))
            coefficient = float(tail_weight) * ramp if tail_enabled else 0.0
            if coefficient > 0.0 and positives:
                tail_loss = smooth_top_tail_loss(
                    model=model,
                    train_x=train_x,
                    train_history=train_history,
                    positives=positives,
                    negatives=negatives,
                    rng=tail_rng,
                    device=device,
                    top_m=top_m,
                    tau=tau,
                    margin=margin,
                )
            else:
                tail_loss = bce.new_zeros(())

            loss = bce + float(reg_lambda) * adaptive_reg + coefficient * tail_loss
            loss.backward()
            optimizer.step()
            count = len(indices)
            loss_sum += float(loss.detach().cpu()) * count
            bce_sum += float(bce.detach().cpu()) * count
            reg_sum += float(adaptive_reg.detach().cpu()) * count
            tail_sum += float(tail_loss.detach().cpu()) * count
            tail_coefficient_sum += coefficient * count
            seen += count

        scores = predict(model, val_x, val_history, device)
        metrics = evaluate(val_users, val_labels, scores)
        primary = float(metrics["primary"])
        trajectory.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "train_bce": round(bce_sum / max(seen, 1), 6),
            "adaptive_penalty": round(reg_sum / max(seen, 1), 6),
            "tail_loss": round(tail_sum / max(seen, 1), 6),
            "mean_tail_weight": round(tail_coefficient_sum / max(seen, 1), 6),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_ndcg5": round(float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best_primary + 1e-8:
            best_primary = primary
            best_epoch = epoch + 1
            if retain_scores:
                best_scores = scores.copy()

    if retain_scores and best_scores is None:
        best_scores = predict(model, val_x, val_history, device)
    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_primary, best_epoch, best_scores, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    seed = int(args.seed)
    reset_random_state(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = int(args.epochs)
    smoke_epochs_text = os.environ.get("SMOKE_EPOCHS")
    smoke_mode = smoke_epochs_text is not None
    if smoke_mode:
        epochs = min(epochs, max(1, int(smoke_epochs_text)))

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w"):
        pass

    train, val, fast_path = load_data(args.data_dir)
    field_dims = np.asarray(train["field_dims"], dtype=np.int64)
    train_x, total_dim = add_context_fields(
        np.asarray(train["X"]), field_dims, train["hourmin"], train["date"]
    )
    val_x, _ = add_context_fields(
        np.asarray(val["X"]), field_dims, val["hourmin"], val["date"]
    )

    train_author = np.asarray(train["X"][:, 2], dtype=np.int64)
    val_author = np.asarray(val["X"][:, 2], dtype=np.int64)
    train_history, val_history = build_causal_histories(
        np.asarray(train["user"]), train_author,
        np.asarray(val["user"]), val_author,
        length=12,
    )

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    labels = np.asarray(train["y"], dtype=np.float32)
    val_labels = np.asarray(val["y"], dtype=np.int64)
    val_users = np.asarray(val["user"])
    tail_pool = build_tail_pool(train_x[:, 0], labels)

    if smoke_mode:
        probe_configs = [
            {"reg_lambda": 0.0, "frequency_power": 0.0, "tail_enabled": False},
            {"reg_lambda": 0.0, "frequency_power": 0.0, "tail_enabled": True},
        ]
    else:
        lambdas = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
        powers = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
        probe_configs = []
        control_pairs = (
            (0.0, 0.0),
            (0.0003, 0.5),
            (0.001, 0.75),
            (0.003, 1.0),
            (0.01, 1.0),
            (0.03, 1.25),
            (0.1, 1.5),
            (0.3, 1.5),
        )
        for reg_lambda, frequency_power in control_pairs:
            probe_configs.append({
                "reg_lambda": reg_lambda,
                "frequency_power": frequency_power,
                "tail_enabled": False,
            })
        probe_configs.append({
            "reg_lambda": 0.0,
            "frequency_power": 0.0,
            "tail_enabled": True,
        })
        for frequency_power in powers:
            for reg_lambda in lambdas:
                probe_configs.append({
                    "reg_lambda": reg_lambda,
                    "frequency_power": frequency_power,
                    "tail_enabled": True,
                })

    weight_cache = {}
    probe_history = []
    winning_primary = -1.0
    winning_config = None
    winning_epoch = 0

    for probe_index, config in enumerate(probe_configs):
        reg_lambda = float(config["reg_lambda"])
        frequency_power = float(config["frequency_power"])
        tail_enabled = bool(config["tail_enabled"])
        if frequency_power not in weight_cache:
            weight_cache[frequency_power] = make_frequency_weights(
                train_x, total_dim, frequency_power
            )
        primary, best_epoch, _, trajectory = train_variant(
            train_x=train_x,
            train_history=train_history,
            labels=labels,
            val_x=val_x,
            val_history=val_history,
            val_users=val_users,
            val_labels=val_labels,
            total_dim=total_dim,
            device=device,
            evaluate=evaluate,
            seed=seed,
            epochs=epochs,
            reg_lambda=reg_lambda,
            frequency_power=frequency_power,
            frequency_weights=weight_cache[frequency_power],
            retain_scores=False,
            tail_enabled=tail_enabled,
            tail_pool=tail_pool,
            tail_weight=0.10,
            top_m=8,
            tau=0.25,
            margin=0.25,
        )
        record = {
            "probe": probe_index + 1,
            "config": {
                "adaptive_reg_lambda": reg_lambda,
                "frequency_power": frequency_power,
                "tail_enabled": tail_enabled,
                "tail_weight": 0.10 if tail_enabled else 0.0,
                "top_m": 8,
                "tau": 0.25,
                "margin": 0.25,
            },
            "best_epoch": best_epoch,
            "primary": float(primary),
            "trajectory": trajectory,
        }
        probe_history.append(record)
        with open(progress_path, "a") as handle:
            handle.write(json.dumps({
                "probe": probe_index + 1,
                "adaptive_reg_lambda": reg_lambda,
                "frequency_power": frequency_power,
                "tail_enabled": tail_enabled,
                "best_epoch": best_epoch,
                "primary": float(primary),
            }, sort_keys=True) + "\n")
        if primary > winning_primary + 1e-8:
            winning_primary = primary
            winning_config = dict(config)
            winning_epoch = best_epoch

    winning_lambda = float(winning_config["reg_lambda"])
    winning_power = float(winning_config["frequency_power"])
    winning_tail_enabled = bool(winning_config["tail_enabled"])
    final_primary, final_best_epoch, best_scores, final_trajectory = train_variant(
        train_x=train_x,
        train_history=train_history,
        labels=labels,
        val_x=val_x,
        val_history=val_history,
        val_users=val_users,
        val_labels=val_labels,
        total_dim=total_dim,
        device=device,
        evaluate=evaluate,
        seed=seed,
        epochs=epochs,
        reg_lambda=winning_lambda,
        frequency_power=winning_power,
        frequency_weights=weight_cache[winning_power],
        retain_scores=True,
        tail_enabled=winning_tail_enabled,
        tail_pool=tail_pool,
        tail_weight=0.10,
        top_m=8,
        tau=0.25,
        margin=0.25,
    )

    final_metrics = evaluate(val_users, val_labels, best_scores)
    gauc = float(final_metrics.get("GAUC", final_metrics.get("gauc")))
    ndcg5 = float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5")))
    primary = float(final_metrics["primary"])

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "history": probe_history,
            "final_history": final_trajectory,
            "configuration": {
                "method": "smooth_top_negative_tail_rider",
                "base_model": "frequency_regularized_causal_pooled_author_history_deepfm",
                "diagnosis": "metric_mismatch_with_unusable_learning_curve_telemetry",
                "adaptive_reg_lambda": winning_lambda,
                "frequency_power": winning_power,
                "tail_enabled": winning_tail_enabled,
                "tail_weight": 0.10 if winning_tail_enabled else 0.0,
                "tail_warmup_epochs": 0.5,
                "tail_ramp_epochs": 2.0,
                "top_m": 8,
                "negative_candidate_count": 24,
                "positive_samples_per_user": 2,
                "tail_temperature": 0.25,
                "tail_margin": 0.25,
                "probe_selected_epoch": winning_epoch,
                "final_best_epoch": final_best_epoch,
                "probe_best_primary": winning_primary,
                "final_best_primary": final_primary,
                "embedding_dim": 12,
                "hidden_dim": 48,
                "history_length": 12,
                "learning_rate": 0.00115,
                "weight_decay": 3e-6,
                "dropout": 0.05,
                "epochs": epochs,
                "num_probes": len(probe_configs),
                "eligible_tail_users": len(tail_pool[0]),
                "seed": seed,
            },
        }, handle)

    users = np.asarray(val["user"])
    videos = np.asarray(val["video"])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            handle.write(f"{i},{users[i]},{videos[i]},{score:.8g}\n")


if __name__ == "__main__":
    main()
