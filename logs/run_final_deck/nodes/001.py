import argparse
import csv
import datetime
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, architecture, strong, k=16):
        super().__init__()
        self.architecture = architecture
        self.dropout = torch.nn.Dropout(0.30 if strong else 0.0)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            d = 5 * k
            self.cross_w = torch.nn.Linear(d, 1, bias=False)
            self.cross_b = torch.nn.Parameter(torch.zeros(d))
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30 if strong else 0.0),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30 if strong else 0.0),
            )
            self.deep_out = torch.nn.Linear(d + 64, 1)
            torch.nn.init.normal_(self.deep_out.weight, std=0.01)
            torch.nn.init.zeros_(self.deep_out.bias)

    def forward(self, x):
        raw_e = self.emb(x)
        e = self.dropout(raw_e)
        summed = e.sum(1)
        pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
        out = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.architecture == "dcn-lite":
            x0 = e.flatten(1)
            cross = x0 * self.cross_w(x0) + self.cross_b + x0
            deep = self.mlp(x0)
            out = out + self.deep_out(torch.cat((cross, deep), dim=1)).squeeze(1)
        return out


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def date_ordinals(values):
    values = np.asarray(values)
    result = np.empty(len(values), dtype=np.int64)
    cache = {}
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        text = text.replace("-", "")
        if text not in cache:
            try:
                cache[text] = datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
            except Exception:
                try:
                    cache[text] = int(float(text))
                except Exception:
                    cache[text] = 0
        result[i] = cache[text]
    return result


def encode_column(train_values, val_values):
    mapping = {}
    train_encoded = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        train_encoded[i] = mapping[key]
    unknown = len(mapping)
    val_encoded = np.asarray([mapping.get(str(v), unknown) for v in val_values], dtype=np.int64)
    return train_encoded, val_encoded, unknown + 1


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_features = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms")}
    train_y = []
    train_dates = []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_features["user_id"].append(row["user_id"])
            train_features["video_id"].append(row["video_id"])
            train_features["author_id"].append(row["author_id"] if has_author else "0")
            train_features["tab"].append(row["tab"])
            train_features["duration_ms"].append(float(row["duration_ms"]))
            train_y.append(float(row["long_view"]))
            train_dates.append(row["date"])
    val_features = {k: [] for k in ("user_id", "video_id", "author_id", "tab", "duration_ms")}
    val_y = []
    val_users = []
    val_videos = []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            user = row["user_id"]
            video = row["video_id"]
            val_features["user_id"].append(user)
            val_features["video_id"].append(video)
            val_features["author_id"].append(row["author_id"] if has_author else "0")
            val_features["tab"].append(row["tab"])
            val_features["duration_ms"].append(float(row["duration_ms"]))
            val_y.append(float(row["long_view"]))
            val_users.append(user)
            val_videos.append(video)
    duration_train = np.asarray(train_features["duration_ms"], dtype=np.float64)
    duration_val = np.asarray(val_features["duration_ms"], dtype=np.float64)
    edges = np.unique(np.quantile(duration_train, np.linspace(0.1, 0.9, 9)))
    duration_train_bucket = np.searchsorted(edges, duration_train, side="right").astype(np.int64)
    duration_val_bucket = np.searchsorted(edges, duration_val, side="right").astype(np.int64)
    train_columns = []
    val_columns = []
    dims = []
    for name in ("user_id", "video_id", "author_id", "tab"):
        tr_col, va_col, dim = encode_column(train_features[name], val_features[name])
        train_columns.append(tr_col)
        val_columns.append(va_col)
        dims.append(dim)
    train_columns.append(duration_train_bucket)
    val_columns.append(duration_val_bucket)
    dims.append(int(max(duration_train_bucket.max(initial=0), duration_val_bucket.max(initial=0))) + 1)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)
    X_train = np.stack(train_columns, axis=1) + offsets
    X_val = np.stack(val_columns, axis=1) + offsets
    return {
        "X_train": X_train,
        "y_train": np.asarray(train_y, dtype=np.float32),
        "users_train": np.asarray(train_features["user_id"]),
        "dates_train": np.asarray(train_dates),
        "X_val": X_val,
        "y_val": np.asarray(val_y, dtype=np.int64),
        "users_val": np.asarray(val_users),
        "videos_val": np.asarray(val_videos),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        dims = np.asarray(tr["field_dims"], dtype=np.int64)
        encoded_video = np.asarray(va["X"][:, 1], dtype=np.int64) - int(dims[0])
        return {
            "X_train": np.asarray(tr["X"], dtype=np.int64),
            "y_train": np.asarray(tr["y"], dtype=np.float32),
            "users_train": np.asarray(tr["user"]),
            "dates_train": np.asarray(tr["date"]),
            "X_val": np.asarray(va["X"], dtype=np.int64),
            "y_val": np.asarray(va["y"], dtype=np.int64),
            "users_val": np.asarray(va["user"]),
            "videos_val": encoded_video,
            "field_dims": dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def build_pair_indices(users, labels, seed):
    rng = np.random.RandomState(seed)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group = order[left:right]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            count = len(group)
            positives.append(pos[rng.randint(0, len(pos), size=count)])
            negatives.append(neg[rng.randint(0, len(neg), size=count)])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pos = np.concatenate(positives).astype(np.int64, copy=False)
    neg = np.concatenate(negatives).astype(np.int64, copy=False)
    shuffle = rng.permutation(len(pos))
    return pos[shuffle], neg[shuffle]


def predict(model, X, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[start:start + batch_size]).to(device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def train_once(data, evaluate_fn, config, seed, epochs, device, half_checkpoints):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RankModel(
        int(data["field_dims"].sum()),
        config["architecture"],
        config["regularization"] == "strong",
    ).to(device)
    strong = config["regularization"] == "strong"
    if strong:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = data["X_train"]
    y = data["y_train"]
    n = len(y)
    batch_size = 32768
    if config["weighting"] == "recency-7d":
        ordinals = date_ordinals(data["dates_train"])
        weights = np.power(0.5, (ordinals.max() - ordinals) / 7.0).astype(np.float32)
        weights /= max(float(weights.mean()), 1e-8)
    else:
        weights = np.ones(n, dtype=np.float32)
    if config["loss"] == "bpr-hybrid":
        pos_idx, neg_idx = build_pair_indices(data["users_train"], y, seed + 7919)
    else:
        pos_idx = np.empty(0, dtype=np.int64)
        neg_idx = np.empty(0, dtype=np.int64)
    rng = np.random.RandomState(seed + 17)
    global_step = 0
    running_loss = 0.0
    running_examples = 0
    pair_cursor = 0
    pair_perm = None
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        if len(pos_idx):
            pair_perm = rng.permutation(len(pos_idx))
            pair_cursor = 0
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            xb = torch.from_numpy(X[indices]).to(device)
            yb = torch.from_numpy(y[indices]).to(device)
            wb = torch.from_numpy(weights[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            point_loss = (point_loss * wb).sum() / wb.sum().clamp_min(1e-8)
            if config["loss"] == "bpr-hybrid" and len(pos_idx):
                needed = len(indices)
                chosen_parts = []
                while needed > 0:
                    take = min(needed, len(pair_perm) - pair_cursor)
                    chosen_parts.append(pair_perm[pair_cursor:pair_cursor + take])
                    pair_cursor += take
                    needed -= take
                    if pair_cursor == len(pair_perm):
                        pair_perm = rng.permutation(len(pos_idx))
                        pair_cursor = 0
                chosen = np.concatenate(chosen_parts)
                pi = pos_idx[chosen]
                ni = neg_idx[chosen]
                xp = torch.from_numpy(X[pi]).to(device)
                xn = torch.from_numpy(X[ni]).to(device)
                pair_loss_values = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                pair_weights = torch.from_numpy(weights[pi]).to(device)
                pair_loss = (pair_loss_values * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            optimizer.step()
            global_step += 1
            running_loss += float(loss.detach().cpu()) * len(indices)
            running_examples += len(indices)

    scores = predict(model, data["X_val"], device)
    current = metric_values(evaluate_fn(data["users_val"], data["y_val"], scores))
    checkpoint = float(epochs)
    curve = [{
        "checkpoint_epoch": checkpoint,
        "global_step": int(global_step),
        "train_loss": float(running_loss / max(running_examples, 1)),
        "gauc": current["gauc"],
        "ndcg5": current["ndcg5"],
        "primary": current["primary"],
    }]
    return {
        "metrics": current,
        "scores": scores,
        "best_checkpoint_epoch": checkpoint,
        "curve": curve,
    }


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w"):
        pass
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = 1
    final_epochs = max(1, min(args.epochs, 3))
    if smoke_cap is not None:
        probe_epochs = max(1, min(probe_epochs, smoke_cap))
        final_epochs = max(1, min(final_epochs, smoke_cap))
    probe_repeats = 1
    baseline_primary = 0.601838
    acceptance_threshold = baseline_primary + 0.002
    configs = []
    for architecture in ("fm", "dcn-lite"):
        for regularization in ("mild", "strong"):
            configs.append({
                "architecture": architecture,
                "loss": "logloss",
                "weighting": "uniform",
                "regularization": regularization,
            })
    history = []
    summaries = []
    for config_id, config in enumerate(configs):
        scores = []
        for repeat in range(probe_repeats):
            seed = args.seed + repeat
            started = time.perf_counter()
            result = train_once(data, evaluate_fn, config, seed, probe_epochs, device, False)
            elapsed = time.perf_counter() - started
            primary = result["metrics"]["primary"]
            scores.append(primary)
            record = {
                "phase": "matrix_probe",
                "config_id": config_id,
                "config": config,
                "seed": seed,
                "epochs": probe_epochs,
                "runtime_seconds": float(elapsed),
                "acceptance_threshold": acceptance_threshold,
                "decision": "passes_screen" if primary >= acceptance_threshold else "rejected_below_minimum_effect",
                "best_checkpoint_epoch": result["best_checkpoint_epoch"],
                "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"],
                "primary": primary,
                "curve": result["curve"],
            }
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
        summaries.append({
            "config_id": config_id,
            "config": config,
            "seed_primaries": scores,
            "mean_primary": float(np.mean(scores)),
            "std_primary": float(np.std(scores)),
        })
    summaries.sort(key=lambda x: (-x["mean_primary"], x["config_id"]))
    top_count = 1
    refinement_epochs = final_epochs
    refinement_repeats = 1
    refined = []
    for summary in summaries[:top_count]:
        config = summary["config"]
        refine_scores = []
        for repeat in range(refinement_repeats):
            seed = args.seed + 100 + repeat
            started = time.perf_counter()
            result = train_once(data, evaluate_fn, config, seed, refinement_epochs, device, True)
            elapsed = time.perf_counter() - started
            primary = result["metrics"]["primary"]
            refine_scores.append(primary)
            record = {
                "phase": "refinement",
                "config_id": summary["config_id"],
                "config": config,
                "seed": seed,
                "epochs": refinement_epochs,
                "runtime_seconds": float(elapsed),
                "acceptance_threshold": acceptance_threshold,
                "decision": "passes_screen" if primary >= acceptance_threshold else "rejected_below_minimum_effect",
                "best_checkpoint_epoch": result["best_checkpoint_epoch"],
                "gauc": result["metrics"]["gauc"],
                "ndcg5": result["metrics"]["ndcg5"],
                "primary": primary,
                "curve": result["curve"],
            }
            history.append(record)
            append_progress(progress_path, {k: v for k, v in record.items() if k != "curve"})
        refined.append({
            "config_id": summary["config_id"],
            "config": config,
            "probe_mean_primary": summary["mean_primary"],
            "seed_primaries": refine_scores,
            "mean_primary": float(np.mean(refine_scores)),
            "std_primary": float(np.std(refine_scores)),
        })
    refined.sort(key=lambda x: (-x["mean_primary"], -x["probe_mean_primary"], x["config_id"]))
    winner = refined[0]
    started = time.perf_counter()
    final_result = train_once(data, evaluate_fn, winner["config"], args.seed, final_epochs, device, True)
    final_elapsed = time.perf_counter() - started
    final_record = {
        "phase": "final_training",
        "config_id": winner["config_id"],
        "config": winner["config"],
        "seed": args.seed,
        "epochs": final_epochs,
        "runtime_seconds": float(final_elapsed),
        "acceptance_threshold": acceptance_threshold,
        "decision": "accepted" if final_result["metrics"]["primary"] >= acceptance_threshold else "rejected_below_minimum_effect",
        "best_checkpoint_epoch": final_result["best_checkpoint_epoch"],
        "gauc": final_result["metrics"]["gauc"],
        "ndcg5": final_result["metrics"]["ndcg5"],
        "primary": final_result["metrics"]["primary"],
        "curve": final_result["curve"],
    }
    history.append(final_record)
    append_progress(progress_path, {k: v for k, v in final_record.items() if k != "curve"})
    scores = final_result["scores"]
    final_metrics = metric_values(evaluate_fn(data["users_val"], data["y_val"], scores))
    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "baseline_primary": baseline_primary,
        "acceptance_threshold": acceptance_threshold,
        "accepted": bool(final_metrics["primary"] >= acceptance_threshold),
        "winner": winner,
        "matrix_summary": summaries,
        "refinement_summary": refined,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(scores):
            writer.writerow([i, data["users_val"][i], data["videos_val"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
