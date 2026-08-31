"""Frequency-regularized FM with same-context BPR negative stratification.

The parent FM, five input fields, adaptive embedding penalty, optimizer, batching,
and validation checkpointing are retained. The sole conceptual intervention is a
hybrid BCE/BPR objective whose BPR negatives are drawn partly from the positive's
same user/date/hour or user/date/tab context, with same-day and within-user
fallbacks. A paired search includes uniform-negative controls and context fractions.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

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

    def forward(self, x, return_embeddings=False):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        logits = self.bias + self.lin(x).sum((1, 2)) + pair
        if return_embeddings:
            return logits, e
        return logits


def _safe_hour(value):
    try:
        return int(float(value)) // 100
    except (TypeError, ValueError):
        return -1


def _read_csv(path, need_label):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row.get("date", ""),
                "hour": _safe_hour(row.get("hourmin", -1)),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def _build_csv_arrays(data_dir):
    train_rows = _read_csv(os.path.join(data_dir, "train.csv"), True)
    val_rows = _read_csv(os.path.join(data_dir, "val.csv"), True)
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))

    fields = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    field_dims = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        mapping = {value: i for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims.append(10)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                mapping = mappings[j]
                x[i, j] = offsets[j] + mapping.get(row[field], len(mapping))
            bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            x[i, 4] = offsets[4] + bucket
        return x

    train = {
        "X": encode(train_rows),
        "y": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in train_rows]),
        "video": np.asarray([r["video_id"] for r in train_rows]),
        "date": np.asarray([r["date"] for r in train_rows]),
        "hour": np.asarray([r["hour"] for r in train_rows], dtype=np.int64),
        "field_dims": field_dims,
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "video": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
    }
    return train, val, False


def _load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr_file, np.load(val_npz) as va_file:
            field_dims = tr_file["field_dims"].astype(np.int64)
            video_offset = int(field_dims[0])
            hourmin = np.asarray(tr_file["hourmin"])
            hour = np.asarray([_safe_hour(v) for v in hourmin], dtype=np.int64)
            train = {
                "X": tr_file["X"].astype(np.int64),
                "y": tr_file["y"].astype(np.float32),
                "user": np.asarray(tr_file["user"]),
                "video": tr_file["X"][:, 1].astype(np.int64) - video_offset,
                "date": np.asarray(tr_file["date"]),
                "hour": hour,
                "field_dims": field_dims,
            }
            val = {
                "X": va_file["X"].astype(np.int64),
                "y": va_file["y"].astype(np.float32),
                "user": np.asarray(va_file["user"]),
                "video": va_file["X"][:, 1].astype(np.int64) - video_offset,
                "field_dims": field_dims,
            }
        return train, val, True
    return _build_csv_arrays(data_dir)


def _make_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
        return evaluate
    from harness.evaluate_provisional import evaluate
    return evaluate


def _metric_values(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def _frequency_weights(x, field_dims, alpha):
    total_dim = int(field_dims.sum())
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))
    weights = np.ones(total_dim, dtype=np.float32)
    for field in range(3):
        start = int(offsets[field])
        end = start + int(field_dims[field])
        local = x[:, field] - start
        counts = np.bincount(local, minlength=int(field_dims[field])).astype(np.float64)
        present = counts > 0
        if not np.any(present):
            continue
        reference = float(np.median(counts[present]))
        raw = np.ones_like(counts)
        raw[present] = np.power(reference / counts[present], alpha)
        raw = np.clip(raw, 0.2, 8.0)
        occurrence_mean = float(np.sum(raw[present] * counts[present]) / np.sum(counts[present]))
        raw /= max(occurrence_mean, 1e-12)
        weights[start:end] = raw.astype(np.float32)
    return weights


def _scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _override_negative_bank(bank, positive_indices, y, key_arrays, rng):
    negative_groups = defaultdict(list)
    negative_indices = np.flatnonzero(y < 0.5)
    for idx in negative_indices:
        key = tuple(_scalar(arr[idx]) for arr in key_arrays)
        negative_groups[key].append(int(idx))

    positive_groups = defaultdict(list)
    for slot, idx in enumerate(positive_indices):
        key = tuple(_scalar(arr[idx]) for arr in key_arrays)
        if key in negative_groups:
            positive_groups[key].append(slot)

    epochs = bank.shape[0]
    overridden = 0
    for key, slots_list in positive_groups.items():
        negatives = np.asarray(negative_groups[key], dtype=np.int64)
        slots = np.asarray(slots_list, dtype=np.int64)
        draws = rng.integers(0, len(negatives), size=(epochs, len(slots)))
        bank[:, slots] = negatives[draws]
        overridden += len(slots)
    return overridden


def _build_pair_banks(train, epochs, seed):
    y = train["y"]
    users = train["user"]
    dates = train["date"]
    hours = train["hour"]
    tabs = train["X"][:, 3]
    rng = np.random.default_rng(seed + 17011)

    negative_by_user = defaultdict(list)
    for idx in np.flatnonzero(y < 0.5):
        negative_by_user[_scalar(users[idx])].append(int(idx))

    eligible_positive = []
    for idx in np.flatnonzero(y >= 0.5):
        if _scalar(users[idx]) in negative_by_user:
            eligible_positive.append(int(idx))
    positive_indices = np.asarray(eligible_positive, dtype=np.int64)
    pair_count = len(positive_indices)
    uniform = np.empty((epochs, pair_count), dtype=np.int64)

    positive_by_user = defaultdict(list)
    for slot, idx in enumerate(positive_indices):
        positive_by_user[_scalar(users[idx])].append(slot)
    for user_key, slots_list in positive_by_user.items():
        slots = np.asarray(slots_list, dtype=np.int64)
        negatives = np.asarray(negative_by_user[user_key], dtype=np.int64)
        draws = rng.integers(0, len(negatives), size=(epochs, len(slots)))
        uniform[:, slots] = negatives[draws]

    stratified = uniform.copy()
    day_count = _override_negative_bank(
        stratified, positive_indices, y, (users, dates), rng
    )
    tab_count = _override_negative_bank(
        stratified, positive_indices, y, (users, dates, tabs), rng
    )
    hour_count = _override_negative_bank(
        stratified, positive_indices, y, (users, dates, hours), rng
    )
    availability = {
        "eligible_pairs": int(pair_count),
        "same_day_available": int(day_count),
        "same_date_tab_available": int(tab_count),
        "same_date_hour_available": int(hour_count),
    }
    return positive_indices, uniform, stratified, availability


def _predict(model, x_val):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x_val), 65536):
            pieces.append(model(x_val[start:start + 65536]).detach().cpu().numpy())
    return np.concatenate(pieces)


def _train_candidate(x_train, y_train, x_val, val_user, val_y, field_dims,
                     positive_indices_np, uniform_bank, stratified_bank,
                     alpha, reg_lambda, bpr_weight, context_fraction,
                     epochs, seed, device, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = FM(int(field_dims.sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    reg_weights = torch.from_numpy(
        _frequency_weights(x_train.detach().cpu().numpy(), field_dims, alpha)
    ).to(device)

    n = len(y_train)
    pair_count = len(positive_indices_np)
    batch_size = 8192
    steps = (n + batch_size - 1) // batch_size
    positive_indices = torch.from_numpy(positive_indices_np).to(device)
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []
    mask_rng = np.random.default_rng(seed + 29021)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        if pair_count > 0 and bpr_weight > 0.0:
            pair_permutation = torch.randperm(pair_count, device=device)
            context_mask = mask_rng.random(pair_count) < context_fraction
            selected_negative = np.where(
                context_mask,
                stratified_bank[epoch],
                uniform_bank[epoch],
            ).astype(np.int64, copy=False)
            negative_indices = torch.from_numpy(selected_negative).to(device)
        else:
            pair_permutation = None
            negative_indices = None
            context_mask = np.zeros(pair_count, dtype=bool)

        loss_sum = 0.0
        batches = 0
        for step, start in enumerate(range(0, n, batch_size)):
            index = permutation[start:start + batch_size]
            xb = x_train[index]
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(xb, return_embeddings=True)
            adaptive = reg_weights[xb]
            penalty = (adaptive.unsqueeze(-1) * embeddings.square()).sum(2).mean()
            point_loss = bce(logits, y_train[index])

            if pair_count > 0 and bpr_weight > 0.0:
                pair_start = (step * pair_count) // steps
                pair_end = ((step + 1) * pair_count) // steps
                slots = pair_permutation[pair_start:pair_end]
                if len(slots) > 0:
                    pos_rows = positive_indices[slots]
                    neg_rows = negative_indices[slots]
                    pair_x = torch.cat((x_train[pos_rows], x_train[neg_rows]), dim=0)
                    pair_logits = model(pair_x)
                    pos_logits = pair_logits[:len(slots)]
                    neg_logits = pair_logits[len(slots):]
                    rank_loss = torch.nn.functional.softplus(-(pos_logits - neg_logits)).mean()
                    data_loss = (1.0 - bpr_weight) * point_loss + bpr_weight * rank_loss
                else:
                    data_loss = point_loss
            else:
                rank_loss = torch.zeros((), device=device)
                data_loss = point_loss

            loss = data_loss + reg_lambda * penalty
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            batches += 1

        scores = _predict(model, x_val)
        values = _metric_values(evaluate(val_user, val_y.astype(int), scores))
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(1, batches), 6),
            "realized_context_fraction": round(float(context_mask.mean()) if pair_count else 0.0, 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = _metric_values(evaluate(val_user, val_y.astype(int), best_scores))
    return best_scores, final_metrics, epoch_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    train, val, fast_path = _load_data(args.data_dir)
    evaluate = _make_evaluator(fast_path)
    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    x_train = torch.from_numpy(train["X"]).to(device)
    y_train = torch.from_numpy(train["y"]).to(device)
    x_val = torch.from_numpy(val["X"]).to(device)
    field_dims = train["field_dims"]

    positive_indices, uniform_bank, stratified_bank, availability = _build_pair_banks(
        train, epochs, args.seed
    )

    if smoke is not None:
        candidates = [(1.0, 0.03, 0.5, 0.30)]
    else:
        alphas = [0.75, 1.0, 1.25]
        lambdas = [0.01, 0.03, 0.10]
        if device.type == "cuda":
            pair_configs = [
                (0.30, 0.00),
                (0.50, 0.00),
                (0.30, 0.15),
                (0.50, 0.15),
                (0.30, 0.30),
                (0.50, 0.30),
                (0.50, 0.60),
            ]
        else:
            pair_configs = [
                (0.30, 0.00),
                (0.50, 0.00),
                (0.30, 0.30),
                (0.50, 0.30),
                (0.50, 0.60),
            ]
        candidates = [
            (alpha, reg_lambda, bpr_weight, context_fraction)
            for alpha in alphas
            for reg_lambda in lambdas
            for bpr_weight, context_fraction in pair_configs
        ]

    history = []
    best_scores = None
    best_metrics = None
    best_config = None
    best_epoch_history = None

    with open(progress_path, "w") as progress:
        for alpha, reg_lambda, bpr_weight, context_fraction in candidates:
            scores, metrics, epoch_history = _train_candidate(
                x_train, y_train, x_val, val["user"], val["y"], field_dims,
                positive_indices, uniform_bank, stratified_bank,
                alpha, reg_lambda, bpr_weight, context_fraction,
                epochs, args.seed, device, evaluate
            )
            config = {
                "frequency_alpha": alpha,
                "embedding_reg_lambda": reg_lambda,
                "bpr_weight": bpr_weight,
                "context_fraction": context_fraction,
            }
            record = {
                "config": config,
                "seed": args.seed,
                "epochs_run": len(epoch_history),
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
                "epochs": epoch_history,
            }
            history.append(record)
            progress.write(json.dumps({
                "config": config,
                "primary": metrics["primary"],
            }, sort_keys=True) + "\n")
            progress.flush()
            if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
                best_scores = scores
                best_metrics = metrics
                best_config = config
                best_epoch_history = epoch_history

    output_metrics = {
        "gauc": best_metrics["gauc"],
        "ndcg5": best_metrics["ndcg5"],
        "primary": best_metrics["primary"],
        "best_config": best_config,
        "best_epoch_history": best_epoch_history,
        "pair_availability": availability,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{val['user'][i]},{val['video'][i]},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
