import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Recommender(torch.nn.Module):
    def __init__(self, total_dim, architecture="fm", k=16, dropout=0.0):
        super().__init__()
        self.architecture = architecture
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            d = 5 * k
            self.cross_w = torch.nn.Parameter(torch.empty(d))
            self.cross_b = torch.nn.Parameter(torch.zeros(d))
            self.deep = torch.nn.Sequential(
                torch.nn.Linear(d, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 1),
            )
            self.deep_scale = torch.nn.Parameter(torch.tensor(0.1))
            torch.nn.init.normal_(self.cross_w, std=0.01)
            torch.nn.init.xavier_uniform_(self.deep[0].weight)
            torch.nn.init.zeros_(self.deep[0].bias)
            torch.nn.init.xavier_uniform_(self.deep[3].weight)
            torch.nn.init.zeros_(self.deep[3].bias)

    def forward(self, x):
        e0 = self.emb(x)
        e = self.drop(e0)
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        out = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.architecture == "dcn-lite":
            x0 = e.reshape(e.shape[0], -1)
            crossed = x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b + x0
            out = out + self.deep_scale * self.deep(crossed).squeeze(1)
        return out


def ordinal_from_yyyymmdd(value):
    s = str(value)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("-", "")
    try:
        return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal()
    except Exception:
        return datetime.date(2022, 4, 21).toordinal()


def encode_csv(train_path, val_path):
    allowed = ["user_id", "video_id", "tab", "date", "duration_ms", "long_view"]

    def read_file(path):
        values = {k: [] for k in allowed}
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for key in allowed:
                    values[key].append(row.get(key, ""))
        return values

    tr = read_file(train_path)
    va = read_file(val_path)
    tr_dur = np.asarray([float(x or 0) for x in tr["duration_ms"]], dtype=np.float64)
    va_dur = np.asarray([float(x or 0) for x in va["duration_ms"]], dtype=np.float64)
    edges = np.unique(np.quantile(tr_dur, np.linspace(0.1, 0.9, 9)))
    tr_bucket = np.searchsorted(edges, tr_dur, side="right").astype(np.int64)
    va_bucket = np.searchsorted(edges, va_dur, side="right").astype(np.int64)

    train_columns = [tr["user_id"], tr["video_id"], ["__author__"] * len(tr_dur), tr["tab"], tr_bucket]
    val_columns = [va["user_id"], va["video_id"], ["__author__"] * len(va_dur), va["tab"], va_bucket]
    encoded_train = []
    encoded_val = []
    dims = []
    offset = 0
    for tc, vc in zip(train_columns, val_columns):
        mapping = {}
        for value in tc:
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping) + 1
        dim = len(mapping) + 1
        encoded_train.append(np.asarray([mapping[str(x)] + offset for x in tc], dtype=np.int64))
        encoded_val.append(np.asarray([mapping.get(str(x), 0) + offset for x in vc], dtype=np.int64))
        dims.append(dim)
        offset += dim

    train = {
        "X": np.stack(encoded_train, axis=1),
        "y": np.asarray([float(x or 0) for x in tr["long_view"]], dtype=np.float32),
        "user": np.asarray(tr["user_id"]),
        "date": np.asarray(tr["date"]),
        "field_dims": np.asarray(dims, dtype=np.int64),
    }
    val = {
        "X": np.stack(encoded_val, axis=1),
        "y": np.asarray([float(x or 0) for x in va["long_view"]], dtype=np.float32),
        "user": np.asarray(va["user_id"]),
        "video": np.asarray(va["video_id"]),
        "date": np.asarray(va["date"]),
        "field_dims": np.asarray(dims, dtype=np.int64),
    }
    return train, val, "csv"


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        tr = {k: tr_file[k] for k in tr_file.files}
        va = {k: va_file[k] for k in va_file.files}
        dims = tr["field_dims"].astype(np.int64)
        video_offset = int(dims[0])
        va["video"] = va["X"][:, 1].astype(np.int64) - video_offset
        return tr, va, "npz"
    return encode_csv(os.path.join(data_dir, "train.csv"), os.path.join(data_dir, "val.csv"))


def metric_values(metric):
    return {
        "gauc": float(metric.get("GAUC", metric.get("gauc"))),
        "ndcg5": float(metric.get("nDCG@5", metric.get("ndcg5"))),
        "primary": float(metric["primary"]),
    }


def recency_weights(dates):
    reference = datetime.date(2022, 4, 21).toordinal()
    ordinals = np.asarray([ordinal_from_yyyymmdd(x) for x in dates], dtype=np.float64)
    age = np.maximum(0.0, reference - ordinals)
    weights = np.exp(-np.log(2.0) * age / 7.0)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def make_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, X, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.from_numpy(X[start:start + 65536]).to(device=device, dtype=torch.long)
            chunks.append(model(xb).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def run_fit(config, seed, epochs, tr, va, device, evaluator):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    dropout = 0.20 if config["regularization"] == "strong" else 0.0
    model = Recommender(total_dim, architecture=config["architecture"], k=16, dropout=dropout).to(device)
    weight_decay = 1e-5 if config["regularization"] == "strong" else 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = None
    if config["regularization"] == "strong":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6], gamma=0.3)

    X = np.asarray(tr["X"], dtype=np.int64)
    y = np.asarray(tr["y"], dtype=np.float32)
    Xv = np.asarray(va["X"], dtype=np.int64)
    sample_weights = recency_weights(tr["date"]) if config["weighting"] == "recency-7d" else np.ones(len(y), dtype=np.float32)
    if config["loss"] == "bpr-hybrid":
        pair_pos, pair_neg = make_pairs(tr["user"], y, seed + 991)
    else:
        pair_pos = pair_neg = np.empty(0, dtype=np.int64)

    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    best_primary = -1.0
    best_scores = None
    best_checkpoint = 0.0
    curve = []
    update_count = 0

    for epoch in range(epochs):
        permutation = torch.randperm(len(y), generator=generator).numpy()
        halves = np.array_split(permutation, 2)
        for half_number, half_indices in enumerate(halves, start=1):
            model.train()
            loss_sum = 0.0
            example_count = 0
            for start in range(0, len(half_indices), batch_size):
                idx = half_indices[start:start + batch_size]
                xb = torch.from_numpy(X[idx]).to(device=device, dtype=torch.long)
                yb = torch.from_numpy(y[idx]).to(device=device)
                wb = torch.from_numpy(sample_weights[idx]).to(device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                bce_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-8)
                if len(pair_pos):
                    pair_choice = torch.randint(len(pair_pos), (len(idx),), generator=generator).numpy()
                    pos_idx = pair_pos[pair_choice]
                    neg_idx = pair_neg[pair_choice]
                    xp = torch.from_numpy(X[pos_idx]).to(device=device, dtype=torch.long)
                    xn = torch.from_numpy(X[neg_idx]).to(device=device, dtype=torch.long)
                    joined = torch.cat([xp, xn], dim=0)
                    joined_scores = model(joined)
                    pos_scores, neg_scores = joined_scores.chunk(2)
                    pair_loss_each = torch.nn.functional.softplus(-(pos_scores - neg_scores))
                    pair_w_np = 0.5 * (sample_weights[pos_idx] + sample_weights[neg_idx])
                    pair_w = torch.from_numpy(pair_w_np).to(device=device)
                    pair_loss = (pair_loss_each * pair_w).sum() / pair_w.sum().clamp_min(1e-8)
                    loss = 0.5 * bce_loss + 0.5 * pair_loss
                else:
                    loss = bce_loss
                loss.backward()
                optimizer.step()
                update_count += 1
                loss_sum += float(loss.detach().cpu()) * len(idx)
                example_count += len(idx)

            scores = predict(model, Xv, device)
            metrics = metric_values(evaluator(va["user"], np.asarray(va["y"]).astype(int), scores))
            checkpoint = epoch + 0.5 * half_number
            curve.append({
                "checkpoint_epoch": checkpoint,
                "train_loss": round(loss_sum / max(example_count, 1), 7),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": metrics["gauc"],
                "val_ndcg5": metrics["ndcg5"],
                "val_primary": metrics["primary"],
            })
            if metrics["primary"] > best_primary + 1e-9:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_checkpoint = checkpoint
        if scheduler is not None:
            scheduler.step()

    final_metrics = metric_values(evaluator(va["user"], np.asarray(va["y"]).astype(int), best_scores))
    record = {
        "seed": int(seed),
        "config": dict(config),
        "config_diff_from_baseline": {
            key: value for key, value in config.items()
            if value != {"architecture": "fm", "loss": "logloss", "weighting": "uniform", "regularization": "mild"}[key]
        },
        "best_epoch": best_checkpoint,
        "metrics": final_metrics,
        "curve": curve,
        "cost": {
            "epochs": int(epochs),
            "half_epoch_checkpoints": int(2 * epochs),
            "optimizer_updates": int(update_count),
            "train_rows": int(len(y)),
            "pair_pool_size": int(len(pair_pos)),
        },
    }
    return record, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = min(args.epochs, int(smoke)) if smoke is not None else args.epochs
    epochs = max(1, epochs)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)

    tr, va, source = load_data(args.data_dir)
    if source == "npz":
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    arms = [
        ("baseline_fm", {"architecture": "fm", "loss": "logloss", "weighting": "uniform", "regularization": "mild"}),
        ("dcn_only", {"architecture": "dcn-lite", "loss": "logloss", "weighting": "uniform", "regularization": "mild"}),
        ("hybrid_only", {"architecture": "fm", "loss": "bpr-hybrid", "weighting": "uniform", "regularization": "mild"}),
        ("regularization_only", {"architecture": "fm", "loss": "logloss", "weighting": "uniform", "regularization": "strong"}),
        ("recency_only", {"architecture": "fm", "loss": "logloss", "weighting": "recency-7d", "regularization": "mild"}),
        ("full_package", {"architecture": "dcn-lite", "loss": "bpr-hybrid", "weighting": "recency-7d", "regularization": "strong"}),
    ]
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    records = []
    score_store = {}
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for arm_name, config in arms:
            score_store[arm_name] = {}
            for seed in seeds:
                record, scores = run_fit(config, seed, epochs, tr, va, device, evaluator)
                record["arm"] = arm_name
                records.append(record)
                score_store[arm_name][seed] = scores
                progress.write(json.dumps({"arm": arm_name, "seed": seed, "primary": record["metrics"]["primary"], "best_epoch": record["best_epoch"], "config": config}, sort_keys=True) + "\n")
                progress.flush()

    by_arm = {}
    for arm_name, config in arms:
        arm_records = [r for r in records if r["arm"] == arm_name]
        values = np.asarray([r["metrics"]["primary"] for r in arm_records], dtype=np.float64)
        by_arm[arm_name] = {
            "config": config,
            "per_seed_primary": [float(x) for x in values],
            "mean_primary": float(values.mean()),
            "std_primary": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    baseline_values = np.asarray(by_arm["baseline_fm"]["per_seed_primary"], dtype=np.float64)
    promotable = []
    for arm_name, _ in arms:
        values = np.asarray(by_arm[arm_name]["per_seed_primary"], dtype=np.float64)
        deltas = values - baseline_values
        by_arm[arm_name]["paired_deltas_vs_baseline"] = [float(x) for x in deltas]
        by_arm[arm_name]["paired_mean_delta"] = float(deltas.mean())
        by_arm[arm_name]["positive_delta_all_seeds"] = bool(np.all(deltas > 0.0)) if arm_name != "baseline_fm" else False
        by_arm[arm_name]["accepted_by_gate"] = bool(arm_name != "baseline_fm" and deltas.mean() >= 0.002 and np.all(deltas > 0.0))
        if by_arm[arm_name]["accepted_by_gate"]:
            promotable.append(arm_name)

    if promotable:
        selected_arm = max(promotable, key=lambda name: by_arm[name]["mean_primary"])
        selection_reason = "highest paired-seed mean among arms exceeding 0.002 mean delta with positive deltas for all seeds"
    else:
        selected_arm = "baseline_fm"
        selection_reason = "no intervention passed the preregistered paired-seed acceptance gate; reverted to baseline"

    selected_scores = score_store[selected_arm][args.seed]
    final_metric = metric_values(evaluator(va["user"], np.asarray(va["y"]).astype(int), selected_scores))
    selected_seed_record = next(r for r in records if r["arm"] == selected_arm and r["seed"] == args.seed)

    metrics_payload = {
        "gauc": final_metric["gauc"],
        "ndcg5": final_metric["ndcg5"],
        "primary": final_metric["primary"],
        "history": records,
        "ablation_summary": by_arm,
        "selection": {
            "selected_arm": selected_arm,
            "output_seed": int(args.seed),
            "selection_reason": selection_reason,
            "acceptance_criterion": "paired mean primary delta >= 0.002 and positive paired delta on all three seeds",
            "replication_plan": "If an arm is numerically promising but fails the gate, repeat only that arm and baseline on three new paired seeds before promotion.",
            "expected_mechanism": "Strong regularization and rapid decay should delay late-epoch degradation, while ranking loss and recency weighting interact with DCN-lite representation learning.",
            "failure_mode": "All paired component and package effects remain below 0.002 or change sign across seeds, providing attribution but no promotable gain.",
            "selected_best_epoch": selected_seed_record["best_epoch"],
        },
        "split": {
            "source": source,
            "train_rows": int(len(tr["y"])),
            "validation_rows": int(len(va["y"])),
            "validation_used_for": "half-epoch checkpoint selection and preregistered six-arm comparison",
            "paired_seeds": seeds,
        },
        "compute_budget": {
            "fits": int(len(arms) * len(seeds)),
            "epochs_per_fit": int(epochs),
            "device": device.type,
            "design": "six controlled arms by three paired seeds",
        },
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    users = np.asarray(va["user"])
    videos = np.asarray(va["video"])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(selected_scores):
            fh.write(f"{i},{users[i]},{videos[i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
