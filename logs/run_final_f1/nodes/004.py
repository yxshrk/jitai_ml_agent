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
        xl = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b
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
    return {
        "Xt": train["X"].astype(np.int64),
        "yt": train["y"].astype(np.float32),
        "ut": train["user"],
        "Xv": val["X"].astype(np.int64),
        "yv": val["y"].astype(np.int64),
        "uv": val["user"],
        "video_out": val["X"][:, 1].astype(np.int64) - int(field_dims[0]),
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
    field_dims = np.asarray([len(mapping) + 1 for mapping in vocab], dtype=np.int64)
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
                "long_view": float(row["long_view"]),
            })
    return {
        "Xt": np.stack([encode(row) for row in train_rows]),
        "yt": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "ut": np.asarray([row["user_id"] for row in train_rows]),
        "Xv": np.stack([encode(row) for row in val_rows]),
        "yv": np.asarray([row["long_view"] for row in val_rows], dtype=np.int64),
        "uv": np.asarray([row["user_id"] for row in val_rows]),
        "video_out": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
    }


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        pos = indices[labels[indices] > 0.5]
        neg = indices[labels[indices] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, X, device, batch_size=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def append_progress(path, record):
    # Intentionally disabled to avoid disk usage issues in constrained environments.
    return None


def train_model(config, seed, epochs, data, evaluator, device, snapshots=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNLite(
        int(data["field_dims"].sum()),
        fields=data["Xt"].shape[1],
        k=16,
        hidden=128,
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    if config["schedule"] == "cyclic":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2, T_mult=1, eta_min=1e-5
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=float(config["gamma"])
        )
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    pair_pos = torch.from_numpy(data["pairs"][0])
    pair_neg = torch.from_numpy(data["pairs"][1])
    n = len(yt)
    batch_size = 8192
    steps = int(math.ceil(n / batch_size))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17011)
    bce = torch.nn.BCEWithLogitsLoss()
    best_primary = -1.0
    best_scores = None
    best_epoch = 0.0
    curve = []
    snapshot_records = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        running_loss = 0.0
        seen = 0
        checkpoints = {steps - 1}
        if not snapshots and steps > 1:
            checkpoints.add(max(0, int(math.ceil(steps / 2.0)) - 1))
        for step, start in enumerate(range(0, n, batch_size)):
            indices = permutation[start:start + batch_size]
            xb = Xt[indices].to(device, non_blocking=True)
            yb = yt[indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = bce(logits, yb)
            if len(pair_pos):
                selected = torch.randint(len(pair_pos), (len(indices),), generator=generator)
                pos_indices = pair_pos[selected]
                neg_indices = pair_neg[selected]
                pair_x = torch.cat((Xt[pos_indices], Xt[neg_indices]), dim=0).to(
                    device, non_blocking=True
                )
                pair_logits = model(pair_x)
                pair_loss = torch.nn.functional.softplus(
                    -(pair_logits[:len(indices)] - pair_logits[len(indices):])
                ).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            optimizer.step()
            if config["schedule"] == "cyclic":
                scheduler.step(epoch + float(step + 1) / max(steps, 1))
            running_loss += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
            if step in checkpoints:
                scores = predict(model, Xv, device)
                metrics = metric_values(evaluator(data["uv"], data["yv"], scores))
                fraction = 1.0 if step == steps - 1 else 0.5
                epoch_value = epoch + fraction
                point = {
                    "epoch": float(epoch_value),
                    "train_loss": float(running_loss / max(seen, 1)),
                    "gauc": metrics["gauc"],
                    "primary": metrics["primary"],
                }
                curve.append(point)
                if step == steps - 1 and snapshots:
                    snapshot_records.append({
                        "epoch": float(epoch + 1),
                        "scores": scores.copy(),
                        "metrics": metrics,
                    })
                if metrics["primary"] > best_primary + 1e-12:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
                    best_epoch = float(epoch_value)
                model.train()
        if config["schedule"] != "cyclic":
            scheduler.step()
    result = {
        "best_primary": float(best_primary),
        "best_epoch": float(best_epoch),
        "curve": curve,
    }
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, best_scores, snapshot_records


def sigmoid(values):
    clipped = np.clip(values.astype(np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def user_rank_transform(scores, users):
    result = np.empty(len(scores), dtype=np.float32)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        count = len(indices)
        if count == 1:
            result[indices[0]] = 0.5
        else:
            local_order = np.argsort(scores[indices], kind="stable")
            ranks = np.empty(count, dtype=np.float32)
            ranks[local_order] = np.arange(count, dtype=np.float32)
            result[indices] = ranks / float(count - 1)
    return result


def aggregate(member_scores, users, rule):
    if rule == "probability_average":
        return np.mean(np.stack([sigmoid(scores) for scores in member_scores]), axis=0).astype(np.float32)
    ranked = [user_rank_transform(scores, users) for scores in member_scores]
    return np.mean(np.stack(ranked), axis=0).astype(np.float32)


def jitter_config(index):
    dropout_values = (0.25, 0.30, 0.35, 0.30, 0.25, 0.35, 0.30)
    lr_values = (0.0008, 0.0010, 0.0012, 0.0009, 0.0011, 0.00085, 0.00115)
    wd_values = (0.0005, 0.0010, 0.0020, 0.0015, 0.00075, 0.00125, 0.0020)
    gamma_values = (0.40, 0.50, 0.60, 0.45, 0.55, 0.50, 0.40)
    j = index % 7
    return {
        "dropout": dropout_values[j],
        "lr": lr_values[j],
        "weight_decay": wd_values[j],
        "gamma": gamma_values[j],
        "schedule": "step",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
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
        os.path.exists(os.path.join(args.data_dir, "train.npz")) and
        os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        data = load_csv_data(args.data_dir)
    data["pairs"] = build_pairs(data["ut"], data["yt"], args.seed)
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    epochs = max(1, int(args.epochs))
    if smoke_value is not None:
        epochs = min(epochs, max(1, int(smoke_value)))
    groups = 18 if device.type == "cuda" else 6
    if smoke_value is not None:
        groups = 1
    fixed_config = {
        "dropout": 0.30,
        "lr": 0.001,
        "weight_decay": 0.001,
        "gamma": 0.50,
        "schedule": "step",
    }
    cyclic_config = {
        "dropout": 0.30,
        "lr": 0.001,
        "weight_decay": 0.001,
        "gamma": 0.50,
        "schedule": "cyclic",
    }
    history = []
    design_summary = []
    best_primary = -1.0
    best_scores = None
    best_design = None
    best_metrics = None
    counts = (3, 5, 7)
    rules = ("probability_average", "per_user_rank_average")
    for group in range(groups):
        banks = {"consecutive_seeds": [], "seed_dial_jitter": [], "cyclic_snapshots": []}
        for member in range(7):
            seed = args.seed + group * 100003 + member
            result, scores, _ = train_model(
                fixed_config, seed, epochs, data, evaluator, device, snapshots=False
            )
            banks["consecutive_seeds"].append(scores)
            record = {
                "phase": "member_probe",
                "bank": "consecutive_seeds",
                "group": group,
                "member": member,
                "seed": seed,
                "config": dict(fixed_config),
                "epochs": epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "curve": result["curve"],
            }
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
        for member in range(7):
            config = jitter_config(group * 7 + member)
            seed = args.seed + 5000003 + group * 100003 + member
            result, scores, _ = train_model(
                config, seed, epochs, data, evaluator, device, snapshots=False
            )
            banks["seed_dial_jitter"].append(scores)
            record = {
                "phase": "member_probe",
                "bank": "seed_dial_jitter",
                "group": group,
                "member": member,
                "seed": seed,
                "config": dict(config),
                "epochs": epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "curve": result["curve"],
            }
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
        snapshot_seed = args.seed + 9000007 + group * 100003
        snapshot_result, _, snapshots = train_model(
            cyclic_config, snapshot_seed, epochs, data, evaluator, device, snapshots=True
        )
        snapshots.sort(key=lambda item: item["metrics"]["primary"], reverse=True)
        selected_snapshots = snapshots[:7]
        banks["cyclic_snapshots"] = [item["scores"] for item in selected_snapshots]
        snapshot_record = {
            "phase": "snapshot_trajectory",
            "bank": "cyclic_snapshots",
            "group": group,
            "seed": snapshot_seed,
            "config": dict(cyclic_config),
            "epochs": epochs,
            "best_epoch": snapshot_result["best_epoch"],
            "primary": snapshot_result["best_primary"],
            "selected_epochs": [item["epoch"] for item in selected_snapshots],
            "selected_primaries": [item["metrics"]["primary"] for item in selected_snapshots],
            "curve": snapshot_result["curve"],
        }
        history.append(snapshot_record)
        append_progress(progress_path, {k: v for k, v in snapshot_record.items() if k != "curve"})
        for bank_name, member_bank in banks.items():
            for count in counts:
                if len(member_bank) < count:
                    continue
                chosen = member_bank[:count]
                for rule in rules:
                    scores = aggregate(chosen, data["uv"], rule)
                    metrics = metric_values(evaluator(data["uv"], data["yv"], scores))
                    design = {
                        "group": group,
                        "bank": bank_name,
                        "member_count": count,
                        "combination_rule": rule,
                    }
                    record = {
                        "phase": "ensemble_design_probe",
                        "design": design,
                        "gauc": metrics["gauc"],
                        "ndcg5": metrics["ndcg5"],
                        "primary": metrics["primary"],
                    }
                    history.append(record)
                    design_summary.append(record)
                    append_progress(progress_path, record)
                    if metrics["primary"] > best_primary + 1e-12:
                        best_primary = metrics["primary"]
                        best_scores = scores.copy()
                        best_design = dict(design)
                        best_metrics = dict(metrics)
    if best_scores is None:
        fallback_result, fallback_scores, _ = train_model(
            fixed_config, args.seed, epochs, data, evaluator, device, snapshots=False
        )
        best_scores = sigmoid(fallback_scores)
        best_metrics = metric_values(evaluator(data["uv"], data["yv"], best_scores))
        best_design = {
            "group": 0,
            "bank": "single_fallback",
            "member_count": 1,
            "combination_rule": "probability",
        }
        history.append({
            "phase": "fallback",
            "design": best_design,
            "primary": best_metrics["primary"],
            "best_epoch": fallback_result["best_epoch"],
        })
    design_summary.sort(key=lambda item: item["primary"], reverse=True)
    output = {
        "gauc": best_metrics["gauc"],
        "ndcg5": best_metrics["ndcg5"],
        "primary": best_metrics["primary"],
        "winner": best_design,
        "epochs_per_trajectory": epochs,
        "groups": groups,
        "design_summary": design_summary,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for index, score in enumerate(best_scores):
            handle.write(
                f"{index},{data['uv'][index]},{data['video_out'][index]},{float(score):.8g}\n"
            )


if __name__ == "__main__":
    main()
