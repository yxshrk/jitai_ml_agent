"""Gauge-fixed FM with frequency-adaptive sparse embedding regularization."""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch


class GaugeFixedFM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair


def read_csv_rows(path):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", "__unknown_author__"),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "long_view": float(row["long_view"]),
            })
    return rows


def make_mapping(values):
    return {value: i + 1 for i, value in enumerate(sorted(set(values)))}


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))

    user_map = make_mapping([r["user_id"] for r in train_rows])
    video_map = make_mapping([r["video_id"] for r in train_rows])
    author_map = make_mapping([r["author_id"] for r in train_rows])
    tab_map = make_mapping([r["tab"] for r in train_rows])

    train_durations = np.asarray(
        [r["duration_ms"] for r in train_rows], dtype=np.float64
    )
    quantiles = np.quantile(train_durations, np.linspace(0.1, 0.9, 9))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        len(author_map) + 1,
        len(tab_map) + 1,
        10,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int32)
        y = np.empty(len(rows), dtype=np.float32)
        users = np.empty(len(rows), dtype=np.int64)
        videos = []
        for i, row in enumerate(rows):
            raw_user = row["user_id"]
            try:
                users[i] = int(raw_user)
            except ValueError:
                users[i] = user_map.get(raw_user, 0)
            videos.append(row["video_id"])
            values = (
                user_map.get(raw_user, 0),
                video_map.get(row["video_id"], 0),
                author_map.get(row["author_id"], 0),
                tab_map.get(row["tab"], 0),
                int(np.searchsorted(
                    quantiles, row["duration_ms"], side="right"
                )),
            )
            x[i] = np.asarray(values, dtype=np.int64) + offsets
            y[i] = row["long_view"]
        return x, y, users, videos

    xt, yt, train_users, _ = encode(train_rows)
    xv, yv, val_users, val_videos = encode(val_rows)
    return {
        "Xt": xt,
        "yt": yt,
        "train_user": train_users,
        "Xv": xv,
        "yv": yv,
        "val_user": val_users,
        "val_video": val_videos,
        "field_dims": field_dims,
        "fast_path": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            return {
                "Xt": tr["X"].astype(np.int64),
                "yt": tr["y"].astype(np.float32),
                "train_user": np.asarray(tr["user"]),
                "Xv": va["X"].astype(np.int64),
                "yv": va["y"].astype(np.float32),
                "val_user": np.asarray(va["user"]),
                "val_video": ["0"] * len(va["y"]),
                "field_dims": tr["field_dims"].astype(np.int64),
                "fast_path": True,
            }
    return load_csv_data(data_dir)


def build_user_slates(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return [order[start:end] for start, end in zip(starts, ends)]


def make_complete_slate_batches(slates, rng, target_size):
    slate_order = rng.permutation(len(slates))
    batches = []
    current = []
    current_size = 0
    for slate_id in slate_order:
        slate = slates[int(slate_id)]
        if current and current_size + len(slate) > target_size:
            batches.append(current)
            current = []
            current_size = 0
        current.append(slate)
        current_size += len(slate)
        if current_size >= target_size:
            batches.append(current)
            current = []
            current_size = 0
    if current:
        batches.append(current)
    return batches


def centered_logits(raw_logits, lengths, global_bias):
    lengths_tensor = torch.as_tensor(
        lengths, device=raw_logits.device, dtype=torch.long
    )
    group_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=raw_logits.device), lengths_tensor
    )
    sums = torch.zeros(
        len(lengths), device=raw_logits.device, dtype=raw_logits.dtype
    )
    sums.scatter_add_(0, group_ids, raw_logits)
    means = sums / lengths_tensor.to(raw_logits.dtype)
    return raw_logits - means[group_ids] + global_bias


def get_evaluator(fast_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    if fast_path:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_value(metrics, upper, lower):
    return metrics[upper] if upper in metrics else metrics[lower]


def make_frequency_weights(x, field_dims, alpha):
    total_dim = int(field_dims.sum())
    counts = np.bincount(x.reshape(-1), minlength=total_dim).astype(np.float64)
    weights = np.zeros(total_dim, dtype=np.float32)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    # Sparse identity fields only: user, video, and author. Context fields retain
    # the exact parent behavior.
    for field in range(min(3, len(field_dims))):
        start = int(offsets[field])
        end = start + int(field_dims[field])
        local = counts[start:end]
        active = local > 0
        if not np.any(active):
            continue
        mean_count = float(local[active].mean())
        raw = np.zeros_like(local)
        raw[active] = np.power(mean_count / local[active], alpha)
        raw[active] = np.minimum(raw[active], 20.0)
        occurrence_mean = float(
            np.sum(local[active] * raw[active]) / np.sum(local[active])
        )
        if occurrence_mean > 0:
            raw[active] /= occurrence_mean
        weights[start:end] = raw.astype(np.float32)
    return weights


def set_determinism(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_once(data, slates, evaluate, device, seed, epochs, reg_lambda, alpha):
    set_determinism(seed)
    model = GaugeFixedFM(int(data["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)

    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    reg_weights = None
    if reg_lambda > 0:
        reg_weights = torch.from_numpy(
            make_frequency_weights(data["Xt"], data["field_dims"], alpha)
        ).to(device)

    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []

    for epoch in range(epochs):
        model.train()
        batches = make_complete_slate_batches(slates, rng, 8192)
        total_loss = 0.0
        total_bce = 0.0
        total_reg = 0.0
        total_examples = 0

        for slate_batch in batches:
            indices = np.concatenate(slate_batch)
            lengths = [len(slate) for slate in slate_batch]
            idx = torch.from_numpy(indices.astype(np.int64, copy=False))
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            fixed = centered_logits(raw, lengths, model.global_bias)
            bce = criterion(fixed, yb)
            if reg_weights is not None:
                row_weights = reg_weights[xb]
                latent_norm = model.emb(xb).pow(2).sum(dim=2)
                linear_norm = model.lin(xb).squeeze(-1).pow(2)
                reg = ((latent_norm + linear_norm) * row_weights).mean()
                loss = bce + reg_lambda * reg
            else:
                reg = torch.zeros((), device=device)
                loss = bce
            loss.backward()
            optimizer.step()

            count = len(indices)
            total_loss += float(loss.detach().cpu()) * count
            total_bce += float(bce.detach().cpu()) * count
            total_reg += float(reg.detach().cpu()) * count
            total_examples += count

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(xv), 65536):
                xb = xv[start:start + 65536].to(device, non_blocking=True)
                score_parts.append(model(xb).detach().cpu().numpy())
        scores = np.concatenate(score_parts)
        metrics = evaluate(data["val_user"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(total_loss / max(1, total_examples), 6),
            "train_bce": round(total_bce / max(1, total_examples), 6),
            "train_reg": round(total_reg / max(1, total_examples), 6),
            "val_gauc": round(float(metric_value(metrics, "GAUC", "gauc")), 6),
            "val_ndcg5": round(float(metric_value(metrics, "nDCG@5", "ndcg5")), 6),
            "val_primary": round(primary, 6),
        })

        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    return {
        "best_primary": best_primary,
        "scores": best_scores,
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    epochs = args.epochs
    smoke_epochs_env = os.environ.get("SMOKE_EPOCHS")
    smoke = smoke_epochs_env is not None
    if smoke:
        epochs = min(epochs, max(1, int(smoke_epochs_env)))

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast_path"])
    slates = build_user_slates(np.asarray(data["train_user"]))
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")

    if smoke:
        seeds = [args.seed]
        candidates = [(0.1, 1.0)]
    elif device.type == "cuda":
        seeds = [args.seed + i for i in range(10)]
        candidates = [
            (lam, alpha)
            for alpha in (0.5, 1.0, 1.5)
            for lam in (0.01, 0.03, 0.1, 0.3, 1.0)
        ]
    else:
        seeds = [args.seed + i for i in range(5)]
        candidates = [
            (lam, alpha)
            for alpha in (0.5, 1.0)
            for lam in (0.01, 0.03, 0.1, 0.3, 1.0)
        ]

    history = []
    baseline_by_seed = {}
    candidate_scores = {
        (lam, alpha): [] for lam, alpha in candidates
    }

    for seed in seeds:
        baseline = train_once(
            data, slates, evaluate, device, seed, epochs, 0.0, 0.0
        )
        baseline_by_seed[seed] = baseline["best_primary"]
        baseline_record = {
            "phase": "probe",
            "seed": seed,
            "config": {"reg_lambda": 0.0, "alpha": 0.0},
            "best_primary": baseline["best_primary"],
            "paired_delta": 0.0,
            "curve": baseline["curve"],
        }
        history.append(baseline_record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({
                "seed": seed,
                "reg_lambda": 0.0,
                "alpha": 0.0,
                "primary": baseline["best_primary"],
            }) + "\n")

        for reg_lambda, alpha in candidates:
            trial = train_once(
                data, slates, evaluate, device, seed, epochs,
                reg_lambda, alpha
            )
            delta = trial["best_primary"] - baseline_by_seed[seed]
            candidate_scores[(reg_lambda, alpha)].append(
                trial["best_primary"]
            )
            history.append({
                "phase": "probe",
                "seed": seed,
                "config": {
                    "reg_lambda": reg_lambda,
                    "alpha": alpha,
                },
                "best_primary": trial["best_primary"],
                "paired_delta": delta,
                "curve": trial["curve"],
            })
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "seed": seed,
                    "reg_lambda": reg_lambda,
                    "alpha": alpha,
                    "primary": trial["best_primary"],
                    "paired_delta": delta,
                }) + "\n")

    baseline_values = np.asarray(
        [baseline_by_seed[s] for s in seeds], dtype=np.float64
    )
    summaries = []
    for reg_lambda, alpha in candidates:
        values = np.asarray(
            candidate_scores[(reg_lambda, alpha)], dtype=np.float64
        )
        deltas = values - baseline_values
        summaries.append({
            "reg_lambda": reg_lambda,
            "alpha": alpha,
            "mean_primary": float(values.mean()),
            "mean_paired_delta": float(deltas.mean()),
            "paired_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "paired_delta_se": float(
                deltas.std(ddof=1) / np.sqrt(len(deltas))
            ) if len(deltas) > 1 else 0.0,
            "per_seed_primary": [float(v) for v in values],
            "per_seed_delta": [float(v) for v in deltas],
        })

    summaries.sort(
        key=lambda item: (
            item["mean_primary"],
            -item["reg_lambda"],
            -item["alpha"],
        ),
        reverse=True,
    )
    winner = summaries[0]

    final_run = train_once(
        data,
        slates,
        evaluate,
        device,
        args.seed,
        epochs,
        winner["reg_lambda"],
        winner["alpha"],
    )
    best_scores = final_run["scores"]
    final_metrics = evaluate(
        data["val_user"], data["yv"].astype(int), best_scores
    )
    history.append({
        "phase": "final",
        "seed": args.seed,
        "config": {
            "reg_lambda": winner["reg_lambda"],
            "alpha": winner["alpha"],
        },
        "best_primary": final_run["best_primary"],
        "curve": final_run["curve"],
    })

    result = {
        "gauc": float(metric_value(final_metrics, "GAUC", "gauc")),
        "ndcg5": float(metric_value(final_metrics, "nDCG@5", "ndcg5")),
        "primary": float(final_metrics["primary"]),
        "selected_config": {
            "reg_lambda": winner["reg_lambda"],
            "alpha": winner["alpha"],
        },
        "baseline_mean_primary": float(baseline_values.mean()),
        "selected_mean_paired_delta": winner["mean_paired_delta"],
        "search_summary": summaries,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(
                f"{i},{data['val_user'][i]},{data['val_video'][i]},"
                f"{score:.6g}\n"
            )


if __name__ == "__main__":
    main()
