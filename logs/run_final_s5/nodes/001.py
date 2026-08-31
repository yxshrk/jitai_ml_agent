import argparse
import csv
import datetime
import json
import os
import sys
import copy
import itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, n_fields=5, k=16, architecture="fm", dropout=0.1):
        super().__init__()
        self.architecture = architecture
        self.n_fields = n_fields
        self.k = k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            dim = n_fields * k
            self.cross_w = torch.nn.Parameter(torch.empty(2, dim))
            self.cross_b = torch.nn.Parameter(torch.zeros(2, dim))
            torch.nn.init.normal_(self.cross_w, std=0.01)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, 1),
            )
            self.cross_out = torch.nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        wide = self.bias + self.lin(x).sum((1, 2))
        if self.architecture == "fm":
            summed = e.sum(1)
            pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
            return wide + pair
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for layer in range(2):
            scalar = (xl * self.cross_w[layer]).sum(1, keepdim=True)
            xl = x0 * scalar + self.cross_b[layer] + xl
        return wide + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def date_ord(value):
    text = str(int(value)) if isinstance(value, (int, np.integer, float, np.floating)) else str(value)
    text = text.strip().replace("-", "")
    try:
        return datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except Exception:
        try:
            return int(float(text))
        except Exception:
            return 0


def recency_weights(dates, half_life=7.0):
    vals = np.asarray([date_ord(x) for x in dates], dtype=np.float32)
    latest = float(vals.max()) if len(vals) else 0.0
    return np.exp2(-(latest - vals) / half_life).astype(np.float32)


def encode_map(train_values, val_values):
    mapping = {}
    encoded_train = np.empty(len(train_values), dtype=np.int64)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded_train[i] = mapping[key]
    unknown = len(mapping)
    encoded_val = np.asarray([mapping.get(str(v), unknown) for v in val_values], dtype=np.int64)
    return encoded_train, encoded_val, unknown + 1


def load_csv_data(data_dir):
    feature_names = ["user_id", "video_id", "tab", "duration_ms", "date"]
    def read_file(path, training):
        cols = {name: [] for name in feature_names}
        labels = []
        with open(path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for name in feature_names:
                    cols[name].append(row[name])
                labels.append(float(row["long_view"]))
        return cols, np.asarray(labels, dtype=np.float32)
    trc, ty = read_file(os.path.join(data_dir, "train.csv"), True)
    vac, vy = read_file(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([float(x) for x in trc["duration_ms"]], dtype=np.float64)
    val_durations = np.asarray([float(x) for x in vac["duration_ms"]], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(cuts, durations, side="right").astype(str)
    val_bucket = np.searchsorted(cuts, val_durations, side="right").astype(str)
    raw_train = [trc["user_id"], trc["video_id"], trc["video_id"], trc["tab"], train_bucket]
    raw_val = [vac["user_id"], vac["video_id"], vac["video_id"], vac["tab"], val_bucket]
    train_fields = []
    val_fields = []
    dims = []
    offset = 0
    for tv, vv in zip(raw_train, raw_val):
        et, ev, dim = encode_map(tv, vv)
        train_fields.append(et + offset)
        val_fields.append(ev + offset)
        dims.append(dim)
        offset += dim
    return {
        "Xt": np.stack(train_fields, axis=1).astype(np.int64),
        "yt": ty,
        "ut": np.asarray(trc["user_id"]),
        "dates": np.asarray(trc["date"]),
        "Xv": np.stack(val_fields, axis=1).astype(np.int64),
        "yv": vy,
        "uv": np.asarray(vac["user_id"]),
        "video": np.asarray(vac["video_id"]),
        "field_dims": np.asarray(dims, dtype=np.int64),
        "official": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        dims = tr["field_dims"].astype(np.int64)
        video_offset = int(dims[0])
        video = va["X"][:, 1].astype(np.int64) - video_offset
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "ut": tr["user"],
            "dates": tr["date"],
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "uv": va["user"],
            "video": video,
            "field_dims": dims,
            "official": True,
        }
    return load_csv_data(data_dir)


def build_pair_index(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = np.asarray(users)[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    pos_parts = []
    start_parts = []
    count_parts = []
    neg_parts = []
    neg_cursor = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        positive = idx[labels[idx] > 0.5]
        negative = idx[labels[idx] <= 0.5]
        if len(positive) and len(negative):
            pos_parts.append(positive.astype(np.int64, copy=False))
            start_parts.append(np.full(len(positive), neg_cursor, dtype=np.int64))
            count_parts.append(np.full(len(positive), len(negative), dtype=np.int64))
            neg_parts.append(negative.astype(np.int64, copy=False))
            neg_cursor += len(negative)
    if not pos_parts:
        return None
    return (
        np.concatenate(pos_parts),
        np.concatenate(start_parts),
        np.concatenate(count_parts),
        np.concatenate(neg_parts),
    )


def normalized_metrics(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, Xv, device, batch_size=65536):
    model.eval()
    pieces = []
    with torch.no_grad():
        for left in range(0, len(Xv), batch_size):
            xb = torch.from_numpy(Xv[left:left + batch_size]).to(device)
            pieces.append(model(xb).detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_once(data, evaluator, pair_index, config, seed, epochs, device, half_checkpoints=False, keep_scores=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    dropout = 0.10 if config["regularization"] == "mild" else 0.30
    weight_decay = 1e-5 if config["regularization"] == "mild" else 1e-3
    model = RankModel(int(data["field_dims"].sum()), architecture=config["architecture"], dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    scheduler = None
    if config["regularization"] == "strong":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    Xt = data["Xt"]
    yt = data["yt"]
    weights = np.ones(len(yt), dtype=np.float32)
    if config["weighting"] == "recency-7d":
        weights = recency_weights(data["dates"], 7.0)
        weights /= max(float(weights.mean()), 1e-8)
    n = len(yt)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    best_step = 0.0
    curve = []
    global_step = 0
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        starts = list(range(0, n, batch_size))
        checkpoint_batches = {len(starts) - 1}
        if half_checkpoints and len(starts) > 1:
            checkpoint_batches.add(max(0, len(starts) // 2 - 1))
        running_loss = 0.0
        batches = 0
        for batch_number, left in enumerate(starts):
            ids = permutation[left:left + batch_size]
            xb = torch.from_numpy(Xt[ids]).to(device)
            yb = torch.from_numpy(yt[ids]).to(device)
            wb = torch.from_numpy(weights[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = (bce(logits, yb) * wb).sum() / wb.sum().clamp_min(1e-8)
            if config["loss"] == "bpr-hybrid" and pair_index is not None:
                pos_all, neg_start, neg_count, neg_all = pair_index
                chosen = rng.integers(0, len(pos_all), size=len(ids), endpoint=False)
                offsets = (rng.random(len(ids)) * neg_count[chosen]).astype(np.int64)
                pos_ids = pos_all[chosen]
                neg_ids = neg_all[neg_start[chosen] + offsets]
                xp = torch.from_numpy(Xt[pos_ids]).to(device)
                xn = torch.from_numpy(Xt[neg_ids]).to(device)
                pair_w_np = 0.5 * (weights[pos_ids] + weights[neg_ids])
                pair_w = torch.from_numpy(pair_w_np.astype(np.float32, copy=False)).to(device)
                pair_each = torch.nn.functional.softplus(-(model(xp) - model(xn)))
                pair_loss = (pair_each * pair_w).sum() / pair_w.sum().clamp_min(1e-8)
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
            global_step += 1
            if batch_number in checkpoint_batches:
                scores = predict(model, data["Xv"], device)
                metrics = normalized_metrics(evaluator, data["uv"], data["yv"], scores)
                fraction = float(epoch) + float(batch_number + 1) / float(len(starts))
                curve.append({
                    "epoch": round(fraction, 3),
                    "train_loss": round(running_loss / max(batches, 1), 6),
                    "gauc": round(metrics["gauc"], 6),
                    "ndcg5": round(metrics["ndcg5"], 6),
                    "primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-8:
                    best_primary = metrics["primary"]
                    best_metrics = metrics
                    best_step = fraction
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()
        if scheduler is not None:
            scheduler.step()
    if keep_scores and best_scores is None:
        best_scores = predict(model, data["Xv"], device)
        best_metrics = normalized_metrics(evaluator, data["uv"], data["yv"], best_scores)
        best_primary = best_metrics["primary"]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "primary": float(best_primary),
        "metrics": best_metrics,
        "best_epoch": float(best_step),
        "curve": curve,
        "scores": best_scores,
    }


def config_key(config):
    return "|".join([config["architecture"], config["loss"], config["weighting"], config["regularization"]])


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    if data["official"]:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator
    pair_index = build_pair_index(data["ut"], data["yt"])
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    matrix_epochs = 10 if device.type == "cuda" else 8
    refine_epochs = 14 if device.type == "cuda" else 12
    final_epochs = args.epochs
    if smoke_cap is not None:
        matrix_epochs = min(matrix_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
    matrix_seed_count = 5 if device.type == "cuda" else 4
    refine_seed_count = 5 if device.type == "cuda" else 3
    if smoke_cap is not None:
        matrix_seed_count = 1
        refine_seed_count = 1
    configs = []
    for architecture, loss, weighting, regularization in itertools.product(
        ["fm", "dcn-lite"],
        ["logloss", "bpr-hybrid"],
        ["uniform", "recency-7d"],
        ["mild", "strong"],
    ):
        configs.append({
            "architecture": architecture,
            "loss": loss,
            "weighting": weighting,
            "regularization": regularization,
        })
    history = []
    aggregate = {}
    probe_number = 0
    for config_index, config in enumerate(configs):
        key = config_key(config)
        aggregate[key] = {"config": copy.deepcopy(config), "matrix_scores": [], "refine_scores": []}
        for seed_index in range(matrix_seed_count):
            run_seed = args.seed + 1009 * config_index + 37 * seed_index
            result = train_once(data, evaluator, pair_index, config, run_seed, matrix_epochs, device, False, False)
            record = {
                "phase": "matrix",
                "probe": probe_number,
                "config": copy.deepcopy(config),
                "seed": run_seed,
                "epochs": matrix_epochs,
                "best_epoch": round(result["best_epoch"], 3),
                "gauc": round(result["metrics"]["gauc"], 6),
                "ndcg5": round(result["metrics"]["ndcg5"], 6),
                "primary": round(result["primary"], 6),
            }
            history.append(record)
            aggregate[key]["matrix_scores"].append(result["primary"])
            append_progress(progress_path, record)
            probe_number += 1
    matrix_ranking = sorted(
        aggregate.values(),
        key=lambda item: float(np.mean(item["matrix_scores"])),
        reverse=True,
    )
    refine_count = 2 if smoke_cap is not None else 4
    finalists = matrix_ranking[:refine_count]
    for finalist_index, item in enumerate(finalists):
        config = item["config"]
        key = config_key(config)
        for seed_index in range(refine_seed_count):
            run_seed = args.seed + 50021 + 1291 * finalist_index + 53 * seed_index
            result = train_once(data, evaluator, pair_index, config, run_seed, refine_epochs, device, False, False)
            record = {
                "phase": "refinement",
                "probe": probe_number,
                "config": copy.deepcopy(config),
                "seed": run_seed,
                "epochs": refine_epochs,
                "best_epoch": round(result["best_epoch"], 3),
                "gauc": round(result["metrics"]["gauc"], 6),
                "ndcg5": round(result["metrics"]["ndcg5"], 6),
                "primary": round(result["primary"], 6),
            }
            history.append(record)
            aggregate[key]["refine_scores"].append(result["primary"])
            append_progress(progress_path, record)
            probe_number += 1
    chosen_item = max(
        finalists,
        key=lambda item: float(np.mean(item["refine_scores"])) if item["refine_scores"] else float(np.mean(item["matrix_scores"])),
    )
    chosen = chosen_item["config"]
    final_seed = args.seed + 900001
    final_result = train_once(data, evaluator, pair_index, chosen, final_seed, final_epochs, device, True, True)
    final_record = {
        "phase": "final",
        "config": copy.deepcopy(chosen),
        "seed": final_seed,
        "epochs": final_epochs,
        "best_epoch": round(final_result["best_epoch"], 3),
        "gauc": round(final_result["metrics"]["gauc"], 6),
        "ndcg5": round(final_result["metrics"]["ndcg5"], 6),
        "primary": round(final_result["primary"], 6),
        "curve": final_result["curve"],
    }
    history.append(final_record)
    append_progress(progress_path, final_record)
    matrix_summary = []
    for item in aggregate.values():
        matrix_summary.append({
            "config": item["config"],
            "matrix_mean": round(float(np.mean(item["matrix_scores"])), 6),
            "matrix_std": round(float(np.std(item["matrix_scores"])), 6),
            "matrix_scores": [round(float(x), 6) for x in item["matrix_scores"]],
            "refine_mean": round(float(np.mean(item["refine_scores"])), 6) if item["refine_scores"] else None,
            "refine_std": round(float(np.std(item["refine_scores"])), 6) if item["refine_scores"] else None,
            "refine_scores": [round(float(x), 6) for x in item["refine_scores"]],
        })
    matrix_summary.sort(key=lambda row: row["matrix_mean"], reverse=True)
    scores = final_result["scores"]
    metrics = final_result["metrics"]
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(scores):
            fh.write(f"{i},{data['uv'][i]},{data['video'][i]},{score:.8g}\n")
    output = {
        "gauc": metrics["gauc"],
        "ndcg5": metrics["ndcg5"],
        "primary": metrics["primary"],
        "chosen_config": chosen,
        "matrix": matrix_summary,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)


if __name__ == "__main__":
    main()
