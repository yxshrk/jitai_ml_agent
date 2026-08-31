"""Paired three-seed evaluation of gauge-fixed BCE versus the official FM baseline."""
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
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


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

    dims = [len(m) + 1 for m in maps] + [10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = maps[j].get(row.get(field, ""), 0) + offsets[j]
            duration = float(row["duration_ms"])
            x[i, 4] = int(np.searchsorted(quantiles, duration, side="right")) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    yt = np.asarray([float(r["long_view"]) for r in train_rows], dtype=np.float32)
    yv = np.asarray([int(float(r["long_view"])) for r in val_rows], dtype=np.int64)
    train_user = np.asarray([scalar_id(r["user_id"]) for r in train_rows])
    val_user = np.asarray([scalar_id(r["user_id"]) for r in val_rows])
    val_video = np.asarray([scalar_id(r["video_id"]) for r in val_rows])
    return {
        "Xt": xt,
        "yt": yt,
        "Xv": xv,
        "yv": yv,
        "train_user": train_user,
        "val_user": val_user,
        "val_video": val_video,
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
    sorted_users = users[order]
    if len(order) == 0:
        return []
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return [g.astype(np.int64, copy=False) for g in np.split(order, cuts)]


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


def train_one(mode, seed, epochs, xt, yt, xv, train_users, evaluator,
              val_users, val_labels, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FM(int(xt.max().item()) + 1, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    n = len(yt)
    batch_size = 8192
    user_groups = make_user_groups(train_users) if mode == "gauge_fixed_bce" else None
    best_primary = -1.0
    best_scores = None
    best_checkpoint = None
    history = []
    patience = 0
    updates = 0

    for epoch in range(epochs):
        model.train()
        if mode == "baseline_bce":
            batches = list(torch.randperm(n, device=device).split(batch_size))
        else:
            packed = packed_user_batches(user_groups, seed, epoch, batch_size)
            batches = [torch.as_tensor(idx, dtype=torch.long, device=device) for idx in packed]

        split_point = max(1, (len(batches) + 1) // 2)
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_end_primary = -1.0
        for batch_number, idx in enumerate(batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(xt[idx])
            if mode == "gauge_fixed_bce":
                logits = gauge_center(logits, train_users_device[idx], model.bias)
            loss = criterion(logits, yt[idx])
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
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    data, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    os.makedirs(args.out_dir, exist_ok=True)
    xt = torch.from_numpy(data["Xt"]).to(device)
    yt = torch.from_numpy(data["yt"]).to(device)
    xv = torch.from_numpy(data["Xv"]).to(device)
    global train_users_device
    train_users_device = torch.from_numpy(np.asarray(data["train_user"], dtype=np.int64)).to(device)

    seeds = [args.seed, args.seed + 1, args.seed + 2]
    runs = []
    baseline_seed_scores = None
    gauge_seed_scores = None
    progress_path = os.path.join(args.out_dir, "progress.log")
    for seed in seeds:
        for mode in ("baseline_bce", "gauge_fixed_bce"):
            result = train_one(mode, seed, epochs, xt, yt, xv,
                               np.asarray(data["train_user"]), evaluator,
                               data["val_user"], data["yv"], device)
            runs.append(result)
            if seed == args.seed and mode == "baseline_bce":
                baseline_seed_scores = result["scores"].copy()
            if seed == args.seed and mode == "gauge_fixed_bce":
                gauge_seed_scores = result["scores"].copy()
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "mode": mode,
                    "seed": seed,
                    "primary": result["metrics"]["primary"],
                    "best_checkpoint": result["best_checkpoint"],
                    "updates": result["updates"]
                }, sort_keys=True) + "\n")

    baseline_runs = [r for r in runs if r["mode"] == "baseline_bce"]
    gauge_runs = [r for r in runs if r["mode"] == "gauge_fixed_bce"]
    baseline_primary = np.asarray([r["metrics"]["primary"] for r in baseline_runs])
    gauge_primary = np.asarray([r["metrics"]["primary"] for r in gauge_runs])
    paired_delta = gauge_primary - baseline_primary
    accepted = bool(paired_delta.mean() >= 0.002 and
                    np.sum(paired_delta > 0.0) >= 2 and
                    paired_delta.min() > -0.001)
    selected_mode = "gauge_fixed_bce" if accepted else "baseline_bce"
    selected_scores = gauge_seed_scores if accepted else baseline_seed_scores
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
        "diagnosis": "validation primary peaks before training ends and subsequently falls",
        "split": data["split"],
        "fast_path": fast_path,
        "device": device.type,
        "seeds": seeds,
        "epochs_cap": epochs,
        "configuration_diff": {
            "baseline_bce": "parent FM, random impression batches, pointwise BCE",
            "gauge_fixed_bce": "same FM/optimizer/lr/batch size/early-stop rule, complete-user batches, logits centered by each user's batch-complete mean plus learned global bias"
        },
        "paired_summary": {
            "baseline_mean": float(baseline_primary.mean()),
            "baseline_std": float(baseline_primary.std(ddof=1)),
            "gauge_mean": float(gauge_primary.mean()),
            "gauge_std": float(gauge_primary.std(ddof=1)),
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
            writer.writerow([i, data["val_user"][i], data["val_video"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
