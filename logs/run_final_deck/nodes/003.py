"""Gauge-fixed FM with adversarial-validation temporal importance weights.

A small linear categorical classifier distinguishes early from late training days
using only legal impression features. Its clipped density ratio supplies training
weights for the otherwise unchanged complete-slate, user-centered BCE model.
"""
import argparse
import csv
import json
import os
import sys
import time

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

    def raw_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def forward(self, x):
        return self.raw_score(x) + self.bias

    def centered_logits(self, x, group_counts):
        raw = self.raw_score(x)
        counts = torch.as_tensor(group_counts, dtype=torch.long, device=raw.device)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        prefix = torch.cat((raw.new_zeros(1), torch.cumsum(raw, dim=0)))
        group_sums = prefix[ends] - prefix[starts]
        group_means = group_sums / counts.to(raw.dtype)
        row_means = torch.repeat_interleave(group_means, counts)
        return raw - row_means + self.bias


class TemporalClassifier(torch.nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x).sum((1, 2)) + self.bias


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            rows.append({
                "user": row.get("user_id", ""),
                "video": row.get("video_id", ""),
                "author": row.get("author_id", "__missing_author__") if has_author else "__missing_author__",
                "tab": row.get("tab", ""),
                "duration": _parse_float(row.get("duration_ms", 0.0)),
                "y": _parse_float(row.get("long_view", 0.0)),
                "date": row.get("date", "") if training else "",
            })
    return rows


def _fit_category(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def _encode_csv(train_rows, val_rows):
    feature_names = ("user", "video", "author", "tab")
    maps = {name: _fit_category([r[name] for r in train_rows]) for name in feature_names}
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    if len(durations):
        edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    else:
        edges = np.zeros(9, dtype=np.float64)

    field_dims = [len(maps[name]) + 1 for name in feature_names] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(feature_names):
                x[i, j] = maps[name].get(row[name], len(maps[name])) + offsets[j]
            x[i, 4] = int(np.searchsorted(edges, row["duration"], side="right")) + offsets[4]
        return x

    return (
        transform(train_rows),
        np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        np.asarray([r["user"] for r in train_rows]),
        np.asarray([r["date"] for r in train_rows]),
        transform(val_rows),
        np.asarray([r["y"] for r in val_rows], dtype=np.float32),
        np.asarray([r["user"] for r in val_rows]),
        np.asarray(field_dims, dtype=np.int64),
        [r["video"] for r in val_rows],
    )


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xv = va["X"].astype(np.int64, copy=False)
        video_ids = ["0"] * len(xv)
        return {
            "Xt": tr["X"].astype(np.int64, copy=False),
            "yt": tr["y"].astype(np.float32, copy=False),
            "ut": np.asarray(tr["user"]),
            "date_t": np.asarray(tr["date"]),
            "Xv": xv,
            "yv": va["y"].astype(np.float32, copy=False),
            "uv": np.asarray(va["user"]),
            "field_dims": np.asarray(tr["field_dims"], dtype=np.int64),
            "video_ids": video_ids,
            "fast_path": True,
        }

    train_rows = _read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = _read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    Xt, yt, ut, date_t, Xv, yv, uv, field_dims, video_ids = _encode_csv(train_rows, val_rows)
    return {
        "Xt": Xt,
        "yt": yt,
        "ut": ut,
        "date_t": date_t,
        "Xv": Xv,
        "yv": yv,
        "uv": uv,
        "field_dims": field_dims,
        "video_ids": video_ids,
        "fast_path": False,
    }


def make_user_groups(users):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(order)])))
    return [order[start:end] for start, end in zip(starts, ends)]


def complete_slate_batches(groups, permutation, target_size):
    pending = []
    pending_size = 0
    for group_number in permutation:
        group = groups[int(group_number)]
        if pending and pending_size + len(group) > target_size:
            yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)
            pending = []
            pending_size = 0
    if pending:
        yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)


def predict(model, x, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def temporal_labels(dates):
    canonical = np.asarray([str(v) for v in dates])
    unique_dates = np.unique(canonical)
    if len(unique_dates) < 2:
        return np.zeros(len(canonical), dtype=np.float32), unique_dates.tolist()
    split = len(unique_dates) // 2
    late_dates = unique_dates[split:]
    labels = np.isin(canonical, late_dates).astype(np.float32)
    return labels, unique_dates.tolist()


def learn_importance_weights(x, dates, total_dim, device, seed, epochs):
    labels, unique_dates = temporal_labels(dates)
    n = len(x)
    if n == 0 or labels.min(initial=0.0) == labels.max(initial=0.0):
        return np.ones(n, dtype=np.float32), {
            "classifier_epochs": 0,
            "classifier_loss": 0.0,
            "early_rows": int(np.sum(labels == 0)),
            "late_rows": int(np.sum(labels == 1)),
            "unique_dates": len(unique_dates),
        }

    n_early = max(int(np.sum(labels == 0)), 1)
    n_late = max(int(np.sum(labels == 1)), 1)
    class_weights = np.where(labels > 0.5, n / (2.0 * n_late), n / (2.0 * n_early)).astype(np.float32)

    model = TemporalClassifier(total_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.RandomState(seed + 1709)
    batch_size = 32768
    final_loss = 0.0

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(labels[idx]).to(device)
            wb = torch.from_numpy(class_weights[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            loss_sum += float((losses * wb).sum().detach().cpu())
            weight_sum += float(wb.sum().detach().cpu())
        final_loss = loss_sum / max(weight_sum, 1e-12)

    probabilities = torch.sigmoid(torch.from_numpy(predict(model, x, device))).numpy()
    probabilities = np.clip(probabilities, 0.02, 0.98)
    density_ratio = probabilities / (1.0 - probabilities)
    density_ratio /= max(float(np.mean(density_ratio)), 1e-12)
    weights = np.clip(density_ratio, 0.25, 4.0).astype(np.float32)
    weights /= max(float(np.mean(weights)), 1e-12)
    effective_n = float(np.square(weights.sum()) / max(float(np.square(weights).sum()), 1e-12))

    diagnostics = {
        "classifier_epochs": epochs,
        "classifier_loss": final_loss,
        "early_rows": n_early,
        "late_rows": n_late,
        "unique_dates": len(unique_dates),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "effective_sample_size": effective_n,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return weights, diagnostics


def metric_values(result):
    return (
        float(result.get("GAUC", result.get("gauc"))),
        float(result.get("nDCG@5", result.get("ndcg5"))),
        float(result["primary"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs
    classifier_epochs = 3
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        cap = max(1, int(smoke_epochs))
        epochs = min(epochs, cap)
        classifier_epochs = min(classifier_epochs, cap)

    started = time.monotonic()
    data = load_data(args.data_dir)
    if data["fast_path"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt = data["Xt"]
    yt = data["yt"]
    groups = make_user_groups(data["ut"])
    total_dim = int(data["field_dims"].sum())

    importance_weights, weight_diagnostics = learn_importance_weights(
        Xt, data["date_t"], total_dim, device, args.seed, classifier_epochs
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = FM(total_dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(args.seed)

    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    patience = 0
    history = []
    batch_size = 8192

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(groups))
        total_loss = 0.0
        total_weight = 0.0
        for indices, counts in complete_slate_batches(groups, permutation, batch_size):
            xb = torch.from_numpy(Xt[indices]).to(device)
            yb = torch.from_numpy(yt[indices]).to(device)
            wb = torch.from_numpy(importance_weights[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.centered_logits(xb, counts)
            row_losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (row_losses * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            total_loss += float((row_losses * wb).sum().detach().cpu())
            total_weight += float(wb.sum().detach().cpu())

        scores = predict(model, data["Xv"], device)
        result = evaluate(data["uv"], data["yv"].astype(int), scores)
        gauc, ndcg5, primary = metric_values(result)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(total_weight, 1e-12),
            "val_gauc": gauc,
            "val_ndcg5": ndcg5,
            "val_primary": primary,
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        best_scores = predict(model, data["Xv"], device)

    final_result = evaluate(data["uv"], data["yv"].astype(int), best_scores)
    gauc, ndcg5, primary = metric_values(final_result)
    runtime_seconds = time.monotonic() - started

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
        "acceptance_threshold": 0.002,
        "method": "gauge-fixed-bce-adversarial-recency",
        "importance_weight_diagnostics": weight_diagnostics,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uv"][i], data["video_ids"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
