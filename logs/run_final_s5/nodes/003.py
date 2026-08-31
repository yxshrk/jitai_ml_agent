import argparse
import csv
import datetime
import json
import os
import sys
import copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, n_fields=5, k=16, dropout=0.30):
        super().__init__()
        self.n_fields = n_fields
        self.k = k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        dim = n_fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(2, dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(2, dim))
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
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        wide = self.bias + self.lin(x).sum((1, 2))
        x0 = e.reshape(e.shape[0], -1)
        xl = x0
        for layer in range(2):
            scalar = (xl * self.cross_w[layer]).sum(1, keepdim=True)
            xl = x0 * scalar + self.cross_b[layer] + xl
        return wide + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def date_ord(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        text = str(int(value))
    else:
        text = str(value)
    text = text.strip().replace("-", "")
    try:
        return datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except Exception:
        try:
            return int(float(text))
        except Exception:
            return 0


def ordinal_dates(values):
    return np.asarray([date_ord(x) for x in values], dtype=np.int32)


def recency_weights(date_values, half_life=7.0):
    values = ordinal_dates(date_values).astype(np.float32)
    latest = float(values.max()) if len(values) else 0.0
    weights = np.exp2(-(latest - values) / half_life).astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


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

    def read_file(path):
        columns = {name: [] for name in feature_names}
        labels = []
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for name in feature_names:
                    columns[name].append(row[name])
                labels.append(float(row["long_view"]))
        return columns, np.asarray(labels, dtype=np.float32)

    train_columns, train_y = read_file(os.path.join(data_dir, "train.csv"))
    val_columns, val_y = read_file(os.path.join(data_dir, "val.csv"))
    train_duration = np.asarray([float(x) for x in train_columns["duration_ms"]], dtype=np.float64)
    val_duration = np.asarray([float(x) for x in val_columns["duration_ms"]], dtype=np.float64)
    cuts = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    train_bucket = np.searchsorted(cuts, train_duration, side="right").astype(str)
    val_bucket = np.searchsorted(cuts, val_duration, side="right").astype(str)
    raw_train = [
        train_columns["user_id"],
        train_columns["video_id"],
        train_columns["video_id"],
        train_columns["tab"],
        train_bucket,
    ]
    raw_val = [
        val_columns["user_id"],
        val_columns["video_id"],
        val_columns["video_id"],
        val_columns["tab"],
        val_bucket,
    ]
    train_fields = []
    val_fields = []
    dimensions = []
    offset = 0
    for train_values, val_values in zip(raw_train, raw_val):
        encoded_train, encoded_val, dimension = encode_map(train_values, val_values)
        train_fields.append(encoded_train + offset)
        val_fields.append(encoded_val + offset)
        dimensions.append(dimension)
        offset += dimension
    return {
        "Xt": np.stack(train_fields, axis=1).astype(np.int64),
        "yt": train_y,
        "ut": np.asarray(train_columns["user_id"]),
        "dates": np.asarray(train_columns["date"]),
        "Xv": np.stack(val_fields, axis=1).astype(np.int64),
        "yv": val_y,
        "uv": np.asarray(val_columns["user_id"]),
        "video": np.asarray(val_columns["video_id"]),
        "field_dims": np.asarray(dimensions, dtype=np.int64),
        "official": False,
    }


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train = np.load(train_path)
        val = np.load(val_path)
        dimensions = train["field_dims"].astype(np.int64)
        video_offset = int(dimensions[0])
        return {
            "Xt": train["X"].astype(np.int64),
            "yt": train["y"].astype(np.float32),
            "ut": train["user"],
            "dates": train["date"],
            "Xv": val["X"].astype(np.int64),
            "yv": val["y"].astype(np.float32),
            "uv": val["user"],
            "video": val["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": dimensions,
            "official": True,
        }
    return load_csv_data(data_dir)


def build_temporal_pair_index(users, labels, date_values, radius_days=7):
    dates = ordinal_dates(date_values)
    order = np.argsort(np.asarray(users), kind="mergesort")
    sorted_users = np.asarray(users)[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    positive_parts = []
    start_parts = []
    count_parts = []
    negative_parts = []
    negative_cursor = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        positives = indices[labels[indices] > 0.5]
        negatives = indices[labels[indices] <= 0.5]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        negative_order = np.argsort(dates[negatives], kind="mergesort")
        negatives = negatives[negative_order]
        negative_dates = dates[negatives]
        local_left = np.searchsorted(negative_dates, dates[positives] - radius_days, side="left")
        local_right = np.searchsorted(negative_dates, dates[positives] + radius_days, side="right")
        local_count = local_right - local_left
        empty = local_count == 0
        local_left[empty] = 0
        local_count[empty] = len(negatives)
        positive_parts.append(positives.astype(np.int64, copy=False))
        start_parts.append((negative_cursor + local_left).astype(np.int64, copy=False))
        count_parts.append(local_count.astype(np.int64, copy=False))
        negative_parts.append(negatives.astype(np.int64, copy=False))
        negative_cursor += len(negatives)
    if not positive_parts:
        return None
    return (
        np.concatenate(positive_parts),
        np.concatenate(start_parts),
        np.concatenate(count_parts),
        np.concatenate(negative_parts),
        dates,
    )


def normalized_metrics(evaluator, users, labels, scores):
    result = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def predict(model, features, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for left in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[left:left + batch_size]).to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_once(data, evaluator, pair_index, forward_multiplier, seed, epochs, device, keep_scores=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = RankModel(int(data["field_dims"].sum())).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    point_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    features = data["Xt"]
    labels = data["yt"]
    recency = recency_weights(data["dates"], 7.0)
    n = len(labels)
    batch_size = 8192
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    best_epoch = 0
    curve = []
    positive_all, negative_start, negative_count, negative_all, dates = pair_index
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        running_loss = 0.0
        batch_count = 0
        for left in range(0, n, batch_size):
            ids = permutation[left:left + batch_size]
            xb = torch.from_numpy(features[ids]).to(device)
            yb = torch.from_numpy(labels[ids]).to(device)
            wb = torch.from_numpy(recency[ids]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_each = point_criterion(logits, yb)
            point_loss = (point_each * wb).sum() / wb.sum().clamp_min(1e-8)

            chosen = rng.integers(0, len(positive_all), size=len(ids), endpoint=False)
            offsets = (rng.random(len(ids)) * negative_count[chosen]).astype(np.int64)
            positive_ids = positive_all[chosen]
            negative_ids = negative_all[negative_start[chosen] + offsets]
            positive_x = torch.from_numpy(features[positive_ids]).to(device)
            negative_x = torch.from_numpy(features[negative_ids]).to(device)
            pair_each = torch.nn.functional.softplus(-(model(positive_x) - model(negative_x)))
            pair_weights_np = 0.5 * (recency[positive_ids] + recency[negative_ids])
            forward_mask = dates[negative_ids] > dates[positive_ids]
            pair_weights_np = pair_weights_np * np.where(forward_mask, forward_multiplier, 1.0).astype(np.float32)
            pair_weights = torch.from_numpy(pair_weights_np.astype(np.float32, copy=False)).to(device)
            pair_loss = (pair_each * pair_weights).sum() / pair_weights.sum().clamp_min(1e-8)
            loss = 0.5 * point_loss + 0.5 * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batch_count += 1

        scheduler.step()
        scores = predict(model, data["Xv"], device)
        metrics = normalized_metrics(evaluator, data["uv"], data["yv"], scores)
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(running_loss / max(batch_count, 1), 6),
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-8:
            best_primary = metrics["primary"]
            best_metrics = metrics
            best_epoch = epoch + 1
            if keep_scores:
                best_scores = scores.copy()

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
        "best_epoch": int(best_epoch),
        "curve": curve,
        "scores": best_scores,
    }


def append_progress(path, record):
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


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

    pair_index = build_temporal_pair_index(data["ut"], data["yt"], data["dates"], radius_days=7)
    if pair_index is None:
        raise RuntimeError("No users with both positive and negative training impressions")

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    probe_epochs = 8
    refinement_epochs = 12
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refinement_epochs = min(refinement_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    multipliers = [1.0, 1.15, 1.3, 1.5, 2.0, 3.0]
    probe_seed_count = 6
    refinement_seed_count = 4
    finalist_count = 2
    if smoke_cap is not None:
        multipliers = [1.0, 1.5]
        probe_seed_count = 1
        refinement_seed_count = 1
        finalist_count = 1

    history = []
    summaries = {}
    probe_number = 0
    for multiplier_index, multiplier in enumerate(multipliers):
        summaries[multiplier] = {"probe_scores": [], "refinement_scores": []}
        for seed_index in range(probe_seed_count):
            run_seed = args.seed + 1009 * seed_index
            result = train_once(
                data, evaluator, pair_index, multiplier, run_seed,
                probe_epochs, device, keep_scores=False,
            )
            record = {
                "phase": "paired_probe",
                "probe": probe_number,
                "forward_multiplier": multiplier,
                "temporal_radius_days": 7,
                "seed": run_seed,
                "epochs": probe_epochs,
                "best_epoch": result["best_epoch"],
                "gauc": round(result["metrics"]["gauc"], 6),
                "ndcg5": round(result["metrics"]["ndcg5"], 6),
                "primary": round(result["primary"], 6),
            }
            history.append(record)
            summaries[multiplier]["probe_scores"].append(result["primary"])
            append_progress(progress_path, record)
            probe_number += 1

    ranked = sorted(
        multipliers,
        key=lambda value: float(np.mean(summaries[value]["probe_scores"])),
        reverse=True,
    )
    finalists = ranked[:finalist_count]
    for finalist_index, multiplier in enumerate(finalists):
        for seed_index in range(refinement_seed_count):
            run_seed = args.seed + 50021 + 1291 * seed_index
            result = train_once(
                data, evaluator, pair_index, multiplier, run_seed,
                refinement_epochs, device, keep_scores=False,
            )
            record = {
                "phase": "paired_refinement",
                "probe": probe_number,
                "forward_multiplier": multiplier,
                "temporal_radius_days": 7,
                "seed": run_seed,
                "epochs": refinement_epochs,
                "best_epoch": result["best_epoch"],
                "gauc": round(result["metrics"]["gauc"], 6),
                "ndcg5": round(result["metrics"]["ndcg5"], 6),
                "primary": round(result["primary"], 6),
            }
            history.append(record)
            summaries[multiplier]["refinement_scores"].append(result["primary"])
            append_progress(progress_path, record)
            probe_number += 1

    def selection_score(multiplier):
        values = summaries[multiplier]["refinement_scores"]
        if values:
            return float(np.mean(values))
        return float(np.mean(summaries[multiplier]["probe_scores"]))

    chosen_multiplier = max(finalists, key=selection_score)
    final_seed = args.seed + 900001
    final_result = train_once(
        data, evaluator, pair_index, chosen_multiplier, final_seed,
        final_epochs, device, keep_scores=True,
    )
    final_record = {
        "phase": "final",
        "forward_multiplier": chosen_multiplier,
        "temporal_radius_days": 7,
        "seed": final_seed,
        "epochs": final_epochs,
        "best_epoch": final_result["best_epoch"],
        "gauc": round(final_result["metrics"]["gauc"], 6),
        "ndcg5": round(final_result["metrics"]["ndcg5"], 6),
        "primary": round(final_result["primary"], 6),
        "curve": final_result["curve"],
    }
    history.append(final_record)
    append_progress(progress_path, final_record)

    summary_rows = []
    control_probe = np.asarray(summaries[1.0]["probe_scores"], dtype=np.float64)
    for multiplier in multipliers:
        probe_scores = np.asarray(summaries[multiplier]["probe_scores"], dtype=np.float64)
        refinement_scores = np.asarray(summaries[multiplier]["refinement_scores"], dtype=np.float64)
        paired_delta = probe_scores - control_probe if len(probe_scores) == len(control_probe) else np.asarray([])
        summary_rows.append({
            "forward_multiplier": multiplier,
            "probe_mean": round(float(probe_scores.mean()), 6),
            "probe_std": round(float(probe_scores.std()), 6),
            "probe_scores": [round(float(x), 6) for x in probe_scores],
            "paired_delta_vs_symmetric_mean": round(float(paired_delta.mean()), 6) if len(paired_delta) else None,
            "paired_deltas_vs_symmetric": [round(float(x), 6) for x in paired_delta],
            "refinement_mean": round(float(refinement_scores.mean()), 6) if len(refinement_scores) else None,
            "refinement_std": round(float(refinement_scores.std()), 6) if len(refinement_scores) else None,
            "refinement_scores": [round(float(x), 6) for x in refinement_scores],
        })
    summary_rows.sort(key=lambda row: row["probe_mean"], reverse=True)

    scores = final_result["scores"]
    metrics = final_result["metrics"]
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(scores):
            handle.write(f"{i},{data['uv'][i]},{data['video'][i]},{score:.8g}\n")

    output = {
        "gauc": metrics["gauc"],
        "ndcg5": metrics["ndcg5"],
        "primary": metrics["primary"],
        "chosen_config": {
            "architecture": "dcn-lite",
            "loss": "0.5-bce+0.5-temporal-bpr",
            "weighting": "recency-7d",
            "regularization": "strong",
            "temporal_radius_days": 7,
            "forward_multiplier": chosen_multiplier,
        },
        "search_summary": summary_rows,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output, handle)


if __name__ == "__main__":
    main()
