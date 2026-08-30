import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.30):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_drop = torch.nn.Dropout(dropout)
        dim = fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(dim))
        self.cross_out = torch.nn.Linear(dim, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x0 = self.emb_drop(self.emb(x)).flatten(1)
        cross_scale = torch.sum(x0 * self.cross_w, dim=1, keepdim=True)
        xl = x0 + x0 * cross_scale + self.cross_b
        return self.bias + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def load_npz(data_dir):
    train = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    val = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = train["field_dims"].astype(np.int64)
    xv = val["X"].astype(np.int64)
    return {
        "Xt": train["X"].astype(np.int64),
        "yt": train["y"].astype(np.float32),
        "ut": train["user"],
        "train_date": train["date"],
        "Xv": xv,
        "yv": val["y"].astype(np.int64),
        "uv": val["user"],
        "val_date": val["date"],
        "video_out": xv[:, 1] - int(field_dims[0]),
        "field_dims": field_dims,
    }


def quantile_edges(values, buckets=10):
    quantiles = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    return np.unique(np.quantile(values.astype(np.float64), quantiles))


def load_csv_data(data_dir):
    train_rows = []
    durations = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as handle:
        for row in csv.DictReader(handle):
            record = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            }
            train_rows.append(record)
            durations.append(record["duration_ms"])

    edges = quantile_edges(np.asarray(durations, dtype=np.float64), 10)
    vocab = [{}, {}, {"__author_unknown__": 0}, {}, {}]

    def token(row, field):
        if field == 0:
            return row["user_id"]
        if field == 1:
            return row["video_id"]
        if field == 2:
            return "__author_unknown__"
        if field == 3:
            return row["tab"]
        return str(int(np.searchsorted(edges, row["duration_ms"], side="right")))

    for row in train_rows:
        for field in (0, 1, 3, 4):
            value = token(row, field)
            if value not in vocab[field]:
                vocab[field][value] = len(vocab[field])

    field_dims = np.asarray([len(values) + 1 for values in vocab], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(row):
        encoded = np.empty(5, dtype=np.int64)
        for field in range(5):
            value = token(row, field)
            encoded[field] = offsets[field] + vocab[field].get(value, len(vocab[field]))
        return encoded

    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as handle:
        for row in csv.DictReader(handle):
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })

    return {
        "Xt": np.stack([encode(row) for row in train_rows]),
        "yt": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "ut": np.asarray([row["user_id"] for row in train_rows]),
        "train_date": np.asarray([row["date"] for row in train_rows]),
        "Xv": np.stack([encode(row) for row in val_rows]),
        "yv": np.asarray([row["long_view"] for row in val_rows], dtype=np.int64),
        "uv": np.asarray([row["user_id"] for row in val_rows]),
        "val_date": np.asarray([row["date"] for row in val_rows]),
        "video_out": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
    }


def date_ordinal(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 8:
        digits = digits[:8]
        try:
            return datetime.date(
                int(digits[:4]), int(digits[4:6]), int(digits[6:8])
            ).toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except ValueError:
        return 0


def recency_weights(train_dates, val_dates, half_life=7.0):
    train_unique, train_inverse = np.unique(train_dates, return_inverse=True)
    train_ordinals = np.asarray(
        [date_ordinal(value) for value in train_unique], dtype=np.float64
    )
    val_unique = np.unique(val_dates)
    val_ordinals = np.asarray(
        [date_ordinal(value) for value in val_unique], dtype=np.float64
    )
    valid_val = val_ordinals[val_ordinals > 0]
    valid_train = train_ordinals[train_ordinals > 0]
    if len(valid_val):
        boundary = float(np.min(valid_val))
    elif len(valid_train):
        boundary = float(np.max(valid_train) + 1.0)
    else:
        return np.ones(len(train_dates), dtype=np.float32)
    ages = np.maximum(0.0, boundary - train_ordinals)
    unique_weights = np.exp(-math.log(2.0) * ages / half_life)
    weights = unique_weights[train_inverse].astype(np.float32)
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        return np.ones(len(train_dates), dtype=np.float32)
    weights /= mean_weight
    return weights


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_users[1:] != sorted_users[:-1], True]
    )
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        positive = indices[labels[indices] > 0.5]
        negative = indices[labels[indices] <= 0.5]
        if len(positive) and len(negative):
            positives.append(positive)
            negatives.append(negative[rng.randint(0, len(negative), size=len(positive))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return (
        np.concatenate(positives).astype(np.int64),
        np.concatenate(negatives).astype(np.int64),
    )


def predict(model, features, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device, non_blocking=True)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_one(weighting, seed, epochs, data, evaluator, device, half_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DCNLite(
        total_dim=int(data["field_dims"].sum()),
        fields=data["Xt"].shape[1],
        k=16,
        hidden=128,
        dropout=0.30,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    if weighting == "recency_7d":
        sample_weight = torch.from_numpy(data["recency"])
    else:
        sample_weight = torch.ones(len(yt), dtype=torch.float32)

    pair_pos, pair_neg = data["pairs"]
    pair_pos_t = torch.from_numpy(pair_pos)
    pair_neg_t = torch.from_numpy(pair_neg)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17011)

    batch_size = 8192
    n = len(yt)
    steps = int(math.ceil(n / batch_size))
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_epoch = 0.0
    curve = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        checkpoints = {steps - 1}
        if half_epoch and steps > 1:
            checkpoints.add(max(0, int(math.ceil(steps / 2.0)) - 1))
        running_loss = 0.0
        seen = 0

        for step, start in enumerate(range(0, n, batch_size)):
            indices = permutation[start:start + batch_size]
            xb = xt[indices].to(device, non_blocking=True)
            yb = yt[indices].to(device, non_blocking=True)
            wb = sample_weight[indices].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = (bce(logits, yb) * wb).mean()

            if len(pair_pos_t):
                chosen = torch.randint(
                    len(pair_pos_t), (len(indices),), generator=generator
                )
                positive_indices = pair_pos_t[chosen]
                negative_indices = pair_neg_t[chosen]
                pair_x = torch.cat(
                    (xt[positive_indices], xt[negative_indices]), dim=0
                ).to(device, non_blocking=True)
                pair_logits = model(pair_x)
                pair_loss = torch.nn.functional.softplus(
                    -(pair_logits[:len(indices)] - pair_logits[len(indices):])
                ).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss

            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)

            if step in checkpoints:
                scores = predict(model, xv, device)
                metrics = metric_values(evaluator(data["uv"], data["yv"], scores))
                fraction = 0.5 if step != steps - 1 else 1.0
                epoch_value = epoch + fraction
                curve.append({
                    "epoch": float(epoch_value),
                    "train_loss": round(running_loss / max(seen, 1), 7),
                    "val_gauc": round(metrics["gauc"], 7),
                    "val_ndcg5": round(metrics["ndcg5"], 7),
                    "val_primary": round(metrics["primary"], 7),
                })
                if metrics["primary"] > best_primary + 1e-12:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
                    best_metrics = metrics
                    best_epoch = float(epoch_value)
                model.train()
        scheduler.step()

    result = {
        "best_epoch": best_epoch,
        "best_primary": float(best_primary),
        "best_gauc": float(best_metrics["gauc"]),
        "best_ndcg5": float(best_metrics["ndcg5"]),
        "curve": curve,
    }
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, best_scores


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz"))
        and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        data = load_csv_data(args.data_dir)

    data["recency"] = recency_weights(
        data["train_date"], data["val_date"], half_life=7.0
    )
    data["pairs"] = build_pairs(data["ut"], data["yt"], args.seed)

    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    epochs = max(1, args.epochs)
    if smoke_cap is not None:
        epochs = min(epochs, max(1, smoke_cap))

    if smoke_cap is not None:
        repeats = 1
    elif device.type == "cuda":
        repeats = 80
    else:
        repeats = 48

    progress_path = os.path.join(args.out_dir, "progress.log")
    history = []
    condition_scores = {"uniform": [], "recency_7d": []}
    condition_epochs = {"uniform": [], "recency_7d": []}
    paired_deltas = []

    for repeat in range(repeats):
        seed = args.seed + 2017 * repeat
        repeat_scores = {}
        for weighting in ("uniform", "recency_7d"):
            result, _ = train_one(
                weighting, seed, epochs, data, evaluator, device, half_epoch=False
            )
            condition_scores[weighting].append(result["best_primary"])
            condition_epochs[weighting].append(result["best_epoch"])
            repeat_scores[weighting] = result["best_primary"]
            record = {
                "phase": "paired_recency_ablation",
                "architecture": "dcn_lite",
                "loss": "hybrid_0.5_bce_0.5_bpr",
                "regularization": "dropout_0.30_adamw_1e-3_step_gamma_0.5",
                "weighting": weighting,
                "seed": seed,
                "epochs": epochs,
                "best_epoch": result["best_epoch"],
                "gauc": result["best_gauc"],
                "ndcg5": result["best_ndcg5"],
                "primary": result["best_primary"],
            }
            history.append(record)
            append_progress(progress_path, record)
        paired_deltas.append(repeat_scores["recency_7d"] - repeat_scores["uniform"])

    summaries = []
    for weighting in ("uniform", "recency_7d"):
        scores = np.asarray(condition_scores[weighting], dtype=np.float64)
        summaries.append({
            "weighting": weighting,
            "mean_primary": float(scores.mean()),
            "std_primary": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "standard_error": (
                float(scores.std(ddof=1) / math.sqrt(len(scores)))
                if len(scores) > 1 else 0.0
            ),
            "mean_best_epoch": float(np.mean(condition_epochs[weighting])),
            "scores": [float(value) for value in scores],
        })

    delta_array = np.asarray(paired_deltas, dtype=np.float64)
    mean_delta = float(delta_array.mean())
    delta_std = float(delta_array.std(ddof=1)) if len(delta_array) > 1 else 0.0
    delta_se = delta_std / math.sqrt(len(delta_array)) if len(delta_array) > 1 else 0.0
    winner = "recency_7d" if mean_delta >= 0.0 else "uniform"

    final_result, best_scores = train_one(
        winner, args.seed, epochs, data, evaluator, device, half_epoch=True
    )
    final_metrics = metric_values(evaluator(data["uv"], data["yv"], best_scores))
    final_record = {
        "phase": "final",
        "architecture": "dcn_lite",
        "loss": "hybrid_0.5_bce_0.5_bpr",
        "regularization": "dropout_0.30_adamw_1e-3_step_gamma_0.5",
        "weighting": winner,
        "seed": args.seed,
        "epochs": epochs,
        "best_epoch": final_result["best_epoch"],
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
    }
    history.append(final_record)
    append_progress(progress_path, final_record)

    recency = data["recency"].astype(np.float64)
    effective_sample_size = float(
        recency.sum() ** 2 / max(float(np.square(recency).sum()), 1e-12)
    )
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winner": winner,
        "hypothesis_test": {
            "mean_paired_primary_delta_recency_minus_uniform": mean_delta,
            "paired_delta_std": delta_std,
            "paired_delta_standard_error": delta_se,
            "paired_repeats": repeats,
            "recency_half_life_days": 7.0,
            "recency_weight_mean": float(recency.mean()),
            "recency_weight_min": float(recency.min()),
            "recency_weight_max": float(recency.max()),
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": effective_sample_size / max(len(recency), 1),
        },
        "condition_summary": summaries,
        "final_curve": final_result["curve"],
        "history": history,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for index, score in enumerate(best_scores):
            handle.write(
                f"{index},{data['uv'][index]},{data['video_out'][index]},"
                f"{float(score):.8g}\n"
            )


if __name__ == "__main__":
    main()
