import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from data.official.evaluate import evaluate
except ImportError:
    from harness.evaluate_provisional import evaluate


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


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
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = {
                    "user_id": row.get("user_id", "0"),
                    "video_id": row.get("video_id", "0"),
                    "author_id": row.get("author_id", row.get("video_id", "0")),
                    "tab": row.get("tab", "0"),
                    "duration_ms": float(row.get("duration_ms", 0) or 0),
                    "long_view": float(row.get("long_view", 0) or 0),
                }
                rows.append(item)
        return rows

    train_rows = read_rows(train_path, True)
    val_rows = read_rows(val_path, False)
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    if len(durations):
        edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        edges = np.asarray([], dtype=np.float64)

    field_names = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    for name in field_names:
        values = sorted({row[name] for row in train_rows})
        mappings.append({value: index + 1 for index, value in enumerate(values)})
    field_dims = [len(mapping) + 1 for mapping in mappings] + [10]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int64)

    def encode(rows):
        n = len(rows)
        X = np.zeros((n, 5), dtype=np.int32)
        for column, name in enumerate(field_names):
            mapping = mappings[column]
            X[:, column] = np.asarray(
                [mapping.get(row[name], 0) for row in rows], dtype=np.int32
            ) + offsets[column]
        duration_values = np.asarray([row["duration_ms"] for row in rows], dtype=np.float64)
        buckets = np.minimum(np.searchsorted(edges, duration_values, side="right"), 9)
        X[:, 4] = buckets.astype(np.int32) + offsets[4]
        return {
            "X": X,
            "y": np.asarray([row["long_view"] for row in rows], dtype=np.float32),
            "user": np.asarray([parse_int(row["user_id"]) for row in rows], dtype=np.int64),
            "field_dims": np.asarray(field_dims, dtype=np.int64),
            "video_output": np.asarray([row["video_id"] for row in rows], dtype=object),
        }

    return encode(train_rows), encode(val_rows)


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_file = np.load(train_path)
        val_file = np.load(val_path)
        train = {
            key: train_file[key]
            for key in train_file.files
            if key in {"X", "y", "user", "field_dims"}
        }
        val = {
            key: val_file[key]
            for key in val_file.files
            if key in {"X", "y", "user", "field_dims"}
        }
        train_file.close()
        val_file.close()
        val["video_output"] = np.zeros(len(val["y"]), dtype=np.int64)
        return train, val
    return load_csv_data(data_dir)


class FM(torch.nn.Module):
    def __init__(self, total_dim, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(total_dim, embedding_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.embedding.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        embeddings = self.embedding(x)
        summed = embeddings.sum(dim=1)
        interaction = 0.5 * (
            summed.square() - embeddings.square().sum(dim=1)
        ).sum(dim=1)
        linear = self.linear(x).sum(dim=(1, 2))
        return self.bias + linear + interaction


def predict(model, X, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            batch = X[start:start + 65536].to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.float32)


def set_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def train_pointwise(seed, epochs, train, val, device):
    set_seed(seed, device)
    X_train = torch.from_numpy(train["X"].astype(np.int64))
    y_train = torch.from_numpy(train["y"].astype(np.float32))
    X_val = torch.from_numpy(val["X"].astype(np.int64))
    model = FM(int(train["field_dims"].sum()), 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_primary = -float("inf")
    best_scores = None
    stale = 0
    curve = []
    n = len(y_train)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, n, 8192):
            indices = permutation[start:start + 8192]
            xb = X_train[indices].to(device)
            yb = y_train[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_size = len(indices)
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size

        scores = predict(model, X_val, device)
        values = metric_values(evaluate(val["user"], val["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 7),
            "val_gauc": round(values["gauc"], 7),
            "val_primary": round(values["primary"], 7),
        })
        if values["primary"] > best_primary + 1e-7:
            best_primary = values["primary"]
            best_scores = scores.copy()
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
    return best_scores, best_primary, curve


def build_mixed_user_groups(users, labels):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    groups = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right].astype(np.int64, copy=False)
        group_labels = labels[indices]
        positives = int(np.count_nonzero(group_labels > 0.5))
        if positives > 0 and positives < len(indices):
            groups.append(indices)
    return groups


def make_group_batches(groups, permutation, target_impressions=16384):
    current = []
    current_size = 0
    for position in permutation:
        group = groups[int(position)]
        if current and current_size + len(group) > target_impressions:
            yield current
            current = []
            current_size = 0
        current.append(group)
        current_size += len(group)
    if current:
        yield current


def listwise_softmax_loss(scores, labels, group_ids, group_count):
    maxima = torch.full(
        (group_count,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maxima.scatter_reduce_(0, group_ids, scores.detach(), reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maxima[group_ids])
    denominators = torch.zeros(group_count, dtype=scores.dtype, device=scores.device)
    denominators.scatter_add_(0, group_ids, exponentials)
    log_normalizers = maxima + torch.log(denominators.clamp_min(1e-12))
    positive_scores = torch.zeros(group_count, dtype=scores.dtype, device=scores.device)
    positive_counts = torch.zeros(group_count, dtype=scores.dtype, device=scores.device)
    positive_scores.scatter_add_(0, group_ids, scores * labels)
    positive_counts.scatter_add_(0, group_ids, labels)
    return (log_normalizers - positive_scores / positive_counts.clamp_min(1.0)).mean()


def train_listwise(seed, epochs, train, val, device):
    set_seed(seed, device)
    X_train = torch.from_numpy(train["X"].astype(np.int64))
    y_train = torch.from_numpy(train["y"].astype(np.float32))
    X_val = torch.from_numpy(val["X"].astype(np.int64))
    groups = build_mixed_user_groups(train["user"].astype(np.int64), train["y"])
    if not groups:
        raise RuntimeError("No users with both positive and negative training labels")

    model = FM(int(train["field_dims"].sum()), 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.RandomState(seed)
    best_primary = -float("inf")
    best_scores = None
    stale = 0
    curve = []

    for epoch in range(epochs):
        model.train()
        group_permutation = rng.permutation(len(groups))
        loss_sum = 0.0
        group_sum = 0
        impression_sum = 0
        for batch_groups in make_group_batches(groups, group_permutation, 16384):
            flat_indices = np.concatenate(batch_groups)
            local_group_ids = np.concatenate([
                np.full(len(group), index, dtype=np.int64)
                for index, group in enumerate(batch_groups)
            ])
            index_tensor = torch.from_numpy(flat_indices)
            xb = X_train[index_tensor].to(device)
            yb = y_train[index_tensor].to(device)
            group_tensor = torch.from_numpy(local_group_ids).to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(xb)
            loss = listwise_softmax_loss(scores, yb, group_tensor, len(batch_groups))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            group_count = len(batch_groups)
            loss_sum += float(loss.detach().cpu()) * group_count
            group_sum += group_count
            impression_sum += len(flat_indices)

        scores = predict(model, X_val, device)
        values = metric_values(evaluate(val["user"], val["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_listwise_loss": round(loss_sum / max(group_sum, 1), 7),
            "mixed_users": int(group_sum),
            "training_impressions": int(impression_sum),
            "val_gauc": round(values["gauc"], 7),
            "val_primary": round(values["primary"], 7),
        })
        if values["primary"] > best_primary + 1e-7:
            best_primary = values["primary"]
            best_scores = scores.copy()
            stale = 0
        else:
            stale += 1
            if stale >= 10:
                break
    return best_scores, best_primary, curve, len(groups)


def safe_correlation(left, right):
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def selected_epoch(curve):
    values = [entry["val_primary"] for entry in curve]
    return int(np.argmax(values) + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    listwise_epochs = max(1, args.epochs)
    baseline_epochs = min(12, listwise_epochs)
    if smoke_value is not None:
        smoke_epochs = max(1, int(smoke_value))
        listwise_epochs = min(listwise_epochs, smoke_epochs)
        baseline_epochs = min(baseline_epochs, smoke_epochs)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    train, val = load_data(args.data_dir)
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    baseline_members = []
    listwise_members = []
    baseline_primaries = []
    listwise_primaries = []
    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")

    with open(progress_path, "w") as progress:
        for seed in seeds:
            baseline_scores, baseline_primary, baseline_curve = train_pointwise(
                seed, baseline_epochs, train, val, device
            )
            baseline_members.append(baseline_scores)
            baseline_primaries.append(float(baseline_primary))
            history.append({
                "config": "matched_parent_fm_k16_bce_adam_lr1e-3_patience2",
                "seed": seed,
                "primary": float(baseline_primary),
                "selected_epoch": selected_epoch(baseline_curve),
                "curve": baseline_curve,
            })
            progress.write(json.dumps({
                "config": "matched_parent_fm",
                "seed": seed,
                "primary": float(baseline_primary),
            }) + "\n")
            progress.flush()

            listwise_scores, listwise_primary, listwise_curve, mixed_users = train_listwise(
                seed, listwise_epochs, train, val, device
            )
            assert not np.allclose(listwise_scores, baseline_scores), "Listwise predictions equal parent predictions"
            listwise_members.append(listwise_scores)
            listwise_primaries.append(float(listwise_primary))
            history.append({
                "config": "listwise_regime_fm_k32_softmax_lr3e-4_patience10",
                "seed": seed,
                "primary": float(listwise_primary),
                "matched_parent_primary": float(baseline_primary),
                "matched_delta": float(listwise_primary - baseline_primary),
                "parent_prediction_correlation": safe_correlation(listwise_scores, baseline_scores),
                "selected_epoch": selected_epoch(listwise_curve),
                "maximum_epochs": int(listwise_epochs),
                "mixed_training_users": int(mixed_users),
                "curve": listwise_curve,
            })
            progress.write(json.dumps({
                "config": "listwise_regime",
                "seed": seed,
                "primary": float(listwise_primary),
                "matched_delta": float(listwise_primary - baseline_primary),
            }) + "\n")
            progress.flush()

    for left in range(len(listwise_members)):
        for right in range(left + 1, len(listwise_members)):
            assert not np.allclose(listwise_members[left], listwise_members[right]), "Listwise ensemble members are identical"

    final_scores = np.mean(np.stack(listwise_members, axis=0), axis=0)
    parent_scores = np.mean(np.stack(baseline_members, axis=0), axis=0)
    assert not np.allclose(final_scores, parent_scores), "Final listwise ensemble equals parent ensemble"

    final_metrics = metric_values(evaluate(val["user"], val["y"].astype(int), final_scores))
    parent_metrics = metric_values(evaluate(val["user"], val["y"].astype(int), parent_scores))
    deltas = np.asarray(listwise_primaries) - np.asarray(baseline_primaries)
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "summary": {
            "diagnosis": "metric_mismatch; parent learning-curve telemetry is missing",
            "acceptance_threshold": 0.002,
            "seeds": seeds,
            "listwise_member_primary_mean": float(np.mean(listwise_primaries)),
            "listwise_member_primary_std": float(np.std(listwise_primaries, ddof=1)),
            "parent_member_primary_mean": float(np.mean(baseline_primaries)),
            "parent_member_primary_std": float(np.std(baseline_primaries, ddof=1)),
            "matched_deltas": [float(value) for value in deltas],
            "matched_delta_mean": float(np.mean(deltas)),
            "matched_delta_std": float(np.std(deltas, ddof=1)),
            "accepted_by_predeclared_threshold": bool(float(np.mean(deltas)) >= 0.002),
            "parent_ensemble_primary": parent_metrics["primary"],
            "ensemble_prediction_correlation_with_parent": safe_correlation(final_scores, parent_scores),
            "ensemble_members": 3,
            "ensemble_rule": "predeclared consecutive-seed mean-logit",
        },
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        video_output = val["video_output"]
        for index, score in enumerate(final_scores):
            handle.write(f"{index},{val['user'][index]},{video_output[index]},{score:.8g}\n")


if __name__ == "__main__":
    main()
