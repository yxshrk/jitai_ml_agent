"""Paired evaluation of adversarial-recency weighting on the gauge-fixed FM."""
import argparse
import csv
import json
import os
import sys

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

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


class AdversarialDayClassifier(torch.nn.Module):
    def __init__(self, total_dim):
        super().__init__()
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x).sum((1, 2)) + self.bias


def scalar_id(value):
    text = str(value)
    try:
        return int(text)
    except ValueError:
        return text


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    with open(train_path, "r", newline="") as fh:
        train_rows = list(csv.DictReader(fh))
    with open(val_path, "r", newline="") as fh:
        val_rows = list(csv.DictReader(fh))

    durations = np.asarray([float(r["duration_ms"]) for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    fields = ["user_id", "video_id", "author_id", "tab"]
    maps = []
    for field in fields:
        mapping = {}
        for row in train_rows:
            value = row.get(field, "")
            if value not in mapping:
                mapping[value] = len(mapping) + 1
        maps.append(mapping)

    dims = [len(mapping) + 1 for mapping in maps] + [10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = maps[j].get(row.get(field, ""), 0) + offsets[j]
            duration = float(row["duration_ms"])
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right")) + offsets[4]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([float(r["long_view"]) for r in train_rows], dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray([int(float(r["long_view"])) for r in val_rows], dtype=np.int64),
        "train_user": np.asarray([scalar_id(r["user_id"]) for r in train_rows]),
        "val_user": np.asarray([scalar_id(r["user_id"]) for r in val_rows]),
        "val_video": np.asarray([scalar_id(r["video_id"]) for r in val_rows]),
        "train_date": np.asarray([str(r.get("date", "")) for r in train_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "split": "train.csv/val.csv"
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.int64),
            "train_user": np.asarray(tr["user"]),
            "val_user": np.asarray(va["user"]),
            "val_video": va["X"][:, 1].astype(np.int64),
            "train_date": np.asarray(tr["date"]).astype(str),
            "field_dims": tr["field_dims"].astype(np.int64),
            "split": "train.npz/val.npz"
        }, True
    return load_csv_data(data_dir), False


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"])
    }


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    if len(order) == 0:
        return []
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [group.astype(np.int64, copy=False) for group in np.split(order, cuts)]


def packed_user_batches(groups, seed, epoch, batch_size):
    rng = np.random.RandomState(seed + 1009 * (epoch + 1))
    group_order = rng.permutation(len(groups))
    batches = []
    pending = []
    size = 0
    for group_id in group_order:
        group = groups[int(group_id)]
        if pending and size + len(group) > batch_size:
            batches.append(np.concatenate(pending))
            pending = []
            size = 0
        pending.append(group)
        size += len(group)
    if pending:
        batches.append(np.concatenate(pending))
    return batches


def gauge_center(logits, batch_users, global_bias):
    _, counts = torch.unique_consecutive(batch_users, return_counts=True)
    prefix = torch.cat((torch.zeros(1, dtype=logits.dtype, device=logits.device),
                        torch.cumsum(logits, dim=0)))
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    sums = prefix[ends] - prefix[starts]
    means = sums / counts.to(logits.dtype)
    return logits - torch.repeat_interleave(means, counts) + global_bias


def predict(model, xv, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(xv), batch_size):
            parts.append(model(xv[start:start + batch_size]).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def date_partition(dates):
    normalized = np.asarray([str(value) for value in dates])
    unique_dates = np.unique(normalized)
    unique_dates.sort()
    if len(unique_dates) < 2:
        return np.zeros(len(normalized), dtype=np.float32), np.ones(len(normalized), dtype=bool), {
            "unique_dates": unique_dates.tolist(),
            "late_dates": unique_dates.tolist(),
            "fallback": True
        }
    late_count = max(1, int(np.ceil(0.25 * len(unique_dates))))
    if late_count >= len(unique_dates):
        late_count = 1
    late_dates = unique_dates[-late_count:]
    late = np.isin(normalized, late_dates).astype(np.float32)
    selected = np.ones(len(normalized), dtype=bool)
    return late, selected, {
        "unique_dates": unique_dates.tolist(),
        "late_dates": late_dates.tolist(),
        "fallback": False
    }


def learn_adversarial_weights(xt, train_dates, total_dim, seed, epochs, device):
    labels_np, selected_np, partition = date_partition(train_dates)
    if partition["fallback"] or labels_np.min() == labels_np.max():
        weights = np.ones(len(labels_np), dtype=np.float32)
        return weights, {
            "partition": partition,
            "epochs_completed": 0,
            "history": [],
            "probability_summary": {"min": 1.0, "max": 1.0, "mean": 1.0},
            "weight_summary": {"min": 1.0, "max": 1.0, "mean": 1.0},
            "effective_sample_size": float(len(weights))
        }

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AdversarialDayClassifier(total_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-6)
    labels = torch.from_numpy(labels_np).to(device)
    selected_indices = np.flatnonzero(selected_np).astype(np.int64)
    positive_rate = float(labels_np[selected_np].mean())
    positive_weight = 0.5 / max(positive_rate, 1e-6)
    negative_weight = 0.5 / max(1.0 - positive_rate, 1e-6)
    batch_size = 16384
    history = []

    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(seed + 7919 * (epoch + 1))
        order = selected_indices[rng.permutation(len(selected_indices))]
        loss_sum = 0.0
        count_sum = 0
        for start in range(0, len(order), batch_size):
            idx_np = order[start:start + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            target = labels[idx]
            logits = model(xt[idx])
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            balancing = torch.where(target > 0.5,
                                    torch.full_like(target, positive_weight),
                                    torch.full_like(target, negative_weight))
            loss = (element_loss * balancing).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(idx_np)
            count_sum += len(idx_np)
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(count_sum, 1), 6)
        })

    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(labels_np), 65536):
            probabilities.append(torch.sigmoid(model(xt[start:start + 65536])).cpu().numpy())
    probabilities = np.concatenate(probabilities).astype(np.float64)
    clipped = np.clip(probabilities, 0.05, 0.95)
    weights = clipped / max(float(clipped.mean()), 1e-12)
    weights = np.clip(weights, 0.20, 5.0)
    weights = weights / max(float(weights.mean()), 1e-12)
    effective_sample_size = float(weights.sum() ** 2 / np.square(weights).sum())
    return weights.astype(np.float32), {
        "partition": partition,
        "epochs_completed": epochs,
        "history": history,
        "positive_rate": positive_rate,
        "probability_summary": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max()),
            "mean": float(probabilities.mean()),
            "std": float(probabilities.std())
        },
        "weight_summary": {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "std": float(weights.std())
        },
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / len(weights)
    }


def train_one(mode, seed, epochs, total_dim, xt, yt, xv, train_users,
              train_users_device, importance_weights, evaluator, val_users,
              val_labels, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FM(total_dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    groups = make_user_groups(train_users)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_checkpoint = None
    history = []
    patience = 0
    updates = 0

    for epoch in range(epochs):
        model.train()
        packed = packed_user_batches(groups, seed, epoch, batch_size)
        batches = [torch.as_tensor(idx, dtype=torch.long, device=device) for idx in packed]
        split_point = max(1, (len(batches) + 1) // 2)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_end_primary = -1.0
        for batch_number, idx in enumerate(batches, start=1):
            logits = model(xt[idx])
            logits = gauge_center(logits, train_users_device[idx], model.bias)
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, yt[idx], reduction="none")
            if mode == "adversarial_recency":
                batch_weights = importance_weights[idx]
                loss = (element_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            else:
                loss = element_loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = int(idx.numel())
            epoch_loss_sum += float(loss.detach().item()) * count
            epoch_examples += count
            updates += 1

            if batch_number == split_point or batch_number == len(batches):
                scores = predict(model, xv)
                metrics = metric_values(evaluator, val_users, val_labels, scores)
                fraction = 0.5 if batch_number == split_point and batch_number != len(batches) else 1.0
                checkpoint = epoch + fraction
                history.append({
                    "checkpoint": checkpoint,
                    "epoch": epoch + 1,
                    "fraction": fraction,
                    "train_loss": round(epoch_loss_sum / max(epoch_examples, 1), 6),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_ndcg5": round(metrics["ndcg5"], 6),
                    "val_primary": round(metrics["primary"], 6),
                    "updates": updates
                })
                if metrics["primary"] > best_primary + 1e-6:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
                    best_checkpoint = checkpoint
                if fraction == 1.0:
                    epoch_end_primary = metrics["primary"]

        if epoch_end_primary + 1e-6 < best_primary:
            patience += 1
        else:
            patience = 0
        if patience >= 2:
            break

    final_metrics = metric_values(evaluator, val_users, val_labels, best_scores)
    return {
        "mode": mode,
        "seed": seed,
        "best_checkpoint": best_checkpoint,
        "epochs_completed": int(history[-1]["epoch"]),
        "updates": updates,
        "metrics": final_metrics,
        "history": history,
        "scores": best_scores
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    epochs = args.epochs
    adversarial_epochs = 3
    if "SMOKE_EPOCHS" in os.environ:
        smoke_epochs = max(1, int(os.environ["SMOKE_EPOCHS"]))
        epochs = min(epochs, smoke_epochs)
        adversarial_epochs = min(adversarial_epochs, smoke_epochs)

    data, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    os.makedirs(args.out_dir, exist_ok=True)
    xt = torch.from_numpy(data["Xt"]).to(device)
    yt = torch.from_numpy(data["yt"]).to(device)
    xv = torch.from_numpy(data["Xv"]).to(device)
    train_users_device = torch.from_numpy(
        np.asarray(data["train_user"], dtype=np.int64)).to(device)
    total_dim = int(np.asarray(data["field_dims"]).sum())

    learned_weights, adversarial_report = learn_adversarial_weights(
        xt, data["train_date"], total_dim, args.seed + 1000003,
        adversarial_epochs, device)
    importance_weights = torch.from_numpy(learned_weights).to(device)

    seeds = [args.seed, args.seed + 1, args.seed + 2]
    runs = []
    base_seed_scores = None
    weighted_seed_scores = None
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "a") as fh:
        fh.write(json.dumps({
            "probe": "adversarial_classifier",
            "epochs": adversarial_epochs,
            "effective_sample_fraction": adversarial_report.get("effective_sample_fraction", 1.0),
            "weight_summary": adversarial_report["weight_summary"]
        }, sort_keys=True) + "\n")

    for seed in seeds:
        for mode in ("uniform_gauge_fixed", "adversarial_recency"):
            result = train_one(
                mode, seed, epochs, total_dim, xt, yt, xv,
                np.asarray(data["train_user"]), train_users_device,
                importance_weights, evaluator, data["val_user"], data["yv"], device)
            runs.append(result)
            if seed == args.seed and mode == "uniform_gauge_fixed":
                base_seed_scores = result["scores"].copy()
            if seed == args.seed and mode == "adversarial_recency":
                weighted_seed_scores = result["scores"].copy()
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "mode": mode,
                    "seed": seed,
                    "primary": result["metrics"]["primary"],
                    "best_checkpoint": result["best_checkpoint"],
                    "updates": result["updates"]
                }, sort_keys=True) + "\n")

    base_runs = [run for run in runs if run["mode"] == "uniform_gauge_fixed"]
    weighted_runs = [run for run in runs if run["mode"] == "adversarial_recency"]
    base_primary = np.asarray([run["metrics"]["primary"] for run in base_runs])
    weighted_primary = np.asarray([run["metrics"]["primary"] for run in weighted_runs])
    paired_delta = weighted_primary - base_primary
    accepted = bool(paired_delta.mean() >= 0.002 and
                    np.sum(paired_delta > 0.0) >= 2 and
                    paired_delta.min() > -0.001)
    selected_mode = "adversarial_recency" if accepted else "uniform_gauge_fixed"
    selected_scores = weighted_seed_scores if accepted else base_seed_scores
    selected_metrics = metric_values(evaluator, data["val_user"], data["yv"], selected_scores)

    serialized_runs = []
    for run in runs:
        serialized_runs.append({
            "mode": run["mode"],
            "seed": run["seed"],
            "best_checkpoint": run["best_checkpoint"],
            "epochs_completed": run["epochs_completed"],
            "updates": run["updates"],
            "metrics": run["metrics"],
            "history": run["history"]
        })

    report = {
        "gauc": selected_metrics["gauc"],
        "ndcg5": selected_metrics["ndcg5"],
        "primary": selected_metrics["primary"],
        "selected_mode": selected_mode,
        "accepted": accepted,
        "acceptance_criterion": "mean paired primary delta >= 0.002, at least two of three deltas positive, and no delta below -0.001",
        "diagnosis": "learning-curve telemetry is missing, so no overfit or underfit claim is made; the tested diagnosis is temporal train-to-validation shift",
        "mechanism": "a train-only classifier distinguishes the latest quarter of training dates from earlier dates using the parent model's five legal categorical fields; normalized clipped late-day probabilities weight gauge-fixed BCE",
        "failure_mode": "the adversarial classifier may capture noise or overly concentrate weights, reducing effective sample size and producing flat or negative paired deltas",
        "replication_plan": "if the mean paired delta is positive but below 0.002 with no materially worse seed, repeat two additional paired seeds before considering promotion",
        "split": data["split"],
        "fast_path": fast_path,
        "device": device.type,
        "seeds": seeds,
        "epochs_cap": epochs,
        "adversarial_classifier": adversarial_report,
        "configuration_diff": {
            "uniform_gauge_fixed": "accepted parent gauge-fixed FM, Adam lr 1e-3, complete-user batches, unweighted BCE",
            "adversarial_recency": "identical model, optimizer, batches, early stopping and seeds; only BCE sample weights change to train-only adversarial late-day probabilities"
        },
        "paired_summary": {
            "uniform_mean": float(base_primary.mean()),
            "uniform_std": float(base_primary.std(ddof=1)),
            "weighted_mean": float(weighted_primary.mean()),
            "weighted_std": float(weighted_primary.std(ddof=1)),
            "paired_deltas": paired_delta.tolist(),
            "paired_delta_mean": float(paired_delta.mean()),
            "paired_delta_std": float(paired_delta.std(ddof=1))
        },
        "history": serialized_runs
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(report, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(selected_scores):
            writer.writerow([i, data["val_user"][i], data["val_video"][i],
                             format(float(score), ".9g")])


if __name__ == "__main__":
    main()
