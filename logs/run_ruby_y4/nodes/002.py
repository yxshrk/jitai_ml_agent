import argparse
import csv
import datetime
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

    def forward(self, x, return_embeddings=False):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        logits = self.bias + self.lin(x).sum((1, 2)) + pair
        if return_embeddings:
            return logits, e
        return logits


def _read_csv(path, need_label):
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row.get("date", "0"),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def _build_csv_arrays(data_dir):
    train_rows = _read_csv(os.path.join(data_dir, "train.csv"), True)
    val_rows = _read_csv(os.path.join(data_dir, "val.csv"), True)
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))
    fields = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    field_dims = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        mapping = {value: i for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims.append(10)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                x[i, j] = offsets[j] + mappings[j].get(row[field], len(mappings[j]))
            bucket = int(np.searchsorted(quantiles, row["duration_ms"], side="right"))
            x[i, 4] = offsets[4] + bucket
        return x

    train = {
        "X": encode(train_rows),
        "y": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in train_rows]),
        "video_raw": np.asarray([r["video_id"] for r in train_rows]),
        "date": np.asarray([r["date"] for r in train_rows]),
        "field_dims": field_dims,
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "video_raw": np.asarray([r["video_id"] for r in val_rows]),
        "date": np.asarray([r["date"] for r in val_rows]),
        "field_dims": field_dims,
    }
    return train, val, False


def _load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr_file, np.load(val_npz) as va_file:
            field_dims = tr_file["field_dims"].astype(np.int64)
            video_offset = int(field_dims[0])
            train = {
                "X": tr_file["X"].astype(np.int64),
                "y": tr_file["y"].astype(np.float32),
                "user": np.asarray(tr_file["user"]),
                "video_raw": tr_file["X"][:, 1].astype(np.int64) - video_offset,
                "date": np.asarray(tr_file["date"]),
                "field_dims": field_dims,
            }
            val = {
                "X": va_file["X"].astype(np.int64),
                "y": va_file["y"].astype(np.float32),
                "user": np.asarray(va_file["user"]),
                "video_raw": va_file["X"][:, 1].astype(np.int64) - video_offset,
                "date": np.asarray(va_file["date"]),
                "field_dims": field_dims,
            }
        return train, val, True
    return _build_csv_arrays(data_dir)


def _make_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
        return evaluate
    from harness.evaluate_provisional import evaluate
    return evaluate


def _metric_values(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def _frequency_weights(x, field_dims, alpha):
    total_dim = int(field_dims.sum())
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))
    weights = np.ones(total_dim, dtype=np.float32)
    for field in range(3):
        start = int(offsets[field])
        size = int(field_dims[field])
        local = x[:, field] - start
        counts = np.bincount(local, minlength=size).astype(np.float64)
        present = counts > 0
        if not np.any(present):
            continue
        reference = float(np.median(counts[present]))
        raw = np.ones_like(counts)
        raw[present] = np.power(reference / counts[present], alpha)
        raw = np.clip(raw, 0.2, 8.0)
        raw /= max(float(np.sum(raw[present] * counts[present]) / np.sum(counts[present])), 1e-12)
        weights[start:start + size] = raw.astype(np.float32)
    return weights


def _train_candidate(x_np, y_np, x_val_np, val_user, val_y, field_dims,
                     alpha, reg_lambda, epochs, seed, device, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FM(int(field_dims.sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    reg_weights = torch.from_numpy(_frequency_weights(x_np, field_dims, alpha)).to(device)
    x_train = torch.from_numpy(x_np).to(device)
    y_train = torch.from_numpy(y_np).to(device)
    x_val = torch.from_numpy(x_val_np).to(device)
    n = len(y_np)
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        last_loss = 0.0
        for start in range(0, n, 8192):
            index = permutation[start:start + 8192]
            xb = x_train[index]
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(xb, return_embeddings=True)
            penalty = (reg_weights[xb].unsqueeze(-1) * embeddings.square()).sum(2).mean()
            loss = bce(logits, y_train[index]) + reg_lambda * penalty
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(x_val), 65536):
                parts.append(model(x_val[start:start + 65536]).detach().cpu().numpy())
        scores = np.concatenate(parts).astype(np.float64)
        values = _metric_values(evaluate(val_user, val_y.astype(int), scores))
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    metrics = _metric_values(evaluate(val_user, val_y.astype(int), best_scores))
    return best_scores, metrics, epoch_history


def _date_ordinals(values):
    result = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        text = str(value.decode() if isinstance(value, bytes) else value)
        digits = "".join(ch for ch in text if ch.isdigit())
        try:
            if len(digits) >= 8:
                result[i] = datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
            else:
                result[i] = int(float(text))
        except (ValueError, OverflowError):
            result[i] = 0
    return result


def _rolling_split(dates):
    ordinals = _date_ordinals(dates)
    unique = np.unique(ordinals)
    if len(unique) >= 2:
        held_days = max(1, int(np.ceil(0.2 * len(unique))))
        threshold = unique[-held_days]
        fit = ordinals < threshold
        hold = ~fit
        if np.any(fit) and np.any(hold):
            return fit, hold
    order = np.argsort(ordinals, kind="stable")
    cut = min(max(1, int(0.8 * len(order))), len(order) - 1)
    fit = np.zeros(len(order), dtype=bool)
    fit[order[:cut]] = True
    return fit, ~fit


def _signed_sketch(history_x, history_y, history_dates, query_x, field_dims, seed, dim=64):
    user_count = int(field_dims[0])
    video_count = int(field_dims[1])
    video_offset = user_count
    hu = history_x[:, 0].astype(np.int64)
    hv = history_x[:, 1].astype(np.int64) - video_offset
    qu = query_x[:, 0].astype(np.int64)
    qv = query_x[:, 1].astype(np.int64) - video_offset
    valid = (hu >= 0) & (hu < user_count) & (hv >= 0) & (hv < video_count)
    hu = hu[valid]
    hv = hv[valid]
    yy = history_y[valid].astype(np.float64)
    day = _date_ordinals(history_dates)[valid]
    age = np.maximum(0, int(day.max()) - day) if len(day) else np.zeros(0, dtype=np.int64)
    weight = np.exp2(-age.astype(np.float64) / 7.0)
    sum_w = np.bincount(hu, weights=weight, minlength=user_count)
    sum_y = np.bincount(hu, weights=weight * yy, minlength=user_count)
    global_mean = float(np.sum(weight * yy) / max(np.sum(weight), 1e-12))
    user_mean = np.full(user_count, global_mean, dtype=np.float64)
    seen = sum_w > 0
    user_mean[seen] = sum_y[seen] / sum_w[seen]
    residual = np.sqrt(weight) * (yy - user_mean[hu])
    rng = np.random.RandomState(seed)
    hashes = rng.randint(0, 2, size=(user_count, dim)).astype(np.float32)
    hashes = hashes * 2.0 - 1.0
    hashes /= np.sqrt(float(dim))
    video_sketch = np.zeros((video_count, dim), dtype=np.float32)
    chunk = 100000
    for start in range(0, len(hu), chunk):
        end = min(start + chunk, len(hu))
        np.add.at(video_sketch, hv[start:end], residual[start:end, None].astype(np.float32) * hashes[hu[start:end]])
    norms = np.linalg.norm(video_sketch, axis=1, keepdims=True)
    video_sketch /= np.maximum(norms, 1e-8)
    user_taste = np.zeros((user_count, dim), dtype=np.float32)
    for start in range(0, len(hu), chunk):
        end = min(start + chunk, len(hu))
        np.add.at(user_taste, hu[start:end], residual[start:end, None].astype(np.float32) * video_sketch[hv[start:end]])
    norms = np.linalg.norm(user_taste, axis=1, keepdims=True)
    user_taste /= np.maximum(norms, 1e-8)
    result = np.zeros(len(query_x), dtype=np.float64)
    qvalid = (qu >= 0) & (qu < user_count) & (qv >= 0) & (qv < video_count)
    result[qvalid] = np.sum(user_taste[qu[qvalid]] * video_sketch[qv[qvalid]], axis=1)
    return result


def _within_user_ranks(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores, dtype=np.float64)
    result = np.zeros(len(scores), dtype=np.float64)
    order = np.argsort(users, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and users[order[end]] == users[order[start]]:
            end += 1
        idx = order[start:end]
        local_order = np.argsort(scores[idx], kind="mergesort")
        sorted_scores = scores[idx][local_order]
        ranks = np.empty(len(idx), dtype=np.float64)
        a = 0
        while a < len(idx):
            b = a + 1
            while b < len(idx) and sorted_scores[b] == sorted_scores[a]:
                b += 1
            ranks[local_order[a:b]] = 0.5 * (a + b - 1)
            a = b
        if len(idx) > 1:
            ranks /= float(len(idx) - 1)
        else:
            ranks[:] = 0.5
        result[idx] = ranks
        start = end
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, val, fast_path = _load_data(args.data_dir)
    evaluate = _make_evaluator(fast_path)
    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    lambdas = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    candidates = [(a, l) for a in alphas for l in lambdas]
    if smoke is not None:
        candidates = [(1.0, 0.03)]
    history = []
    best_scores = None
    best_metrics = None
    best_config = None
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for alpha, reg_lambda in candidates:
            scores, metrics, epoch_history = _train_candidate(
                train["X"], train["y"], val["X"], val["user"], val["y"],
                train["field_dims"], alpha, reg_lambda, epochs, args.seed, device, evaluate
            )
            record = {
                "stage": "parent_probe",
                "config": {"frequency_alpha": alpha, "embedding_reg_lambda": reg_lambda},
                "seed": args.seed,
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
                "epochs": epoch_history,
            }
            history.append(record)
            progress.write(json.dumps({"member": "parent", "frequency_alpha": alpha, "embedding_reg_lambda": reg_lambda, "primary": metrics["primary"]}, sort_keys=True) + "\n")
            progress.flush()
            if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
                best_scores = scores
                best_metrics = metrics
                best_config = record["config"]
        fit_mask, hold_mask = _rolling_split(train["date"])
        hold_scores, hold_parent_metrics, hold_epochs = _train_candidate(
            train["X"][fit_mask], train["y"][fit_mask], train["X"][hold_mask],
            train["user"][hold_mask], train["y"][hold_mask], train["field_dims"],
            best_config["frequency_alpha"], best_config["embedding_reg_lambda"],
            epochs, args.seed + 1, device, evaluate
        )
        graph_seed = args.seed + 1009
        hold_graph = _signed_sketch(
            train["X"][fit_mask], train["y"][fit_mask], train["date"][fit_mask],
            train["X"][hold_mask], train["field_dims"], graph_seed
        )
        assert not np.allclose(hold_scores, hold_graph)
        hold_parent_rank = _within_user_ranks(train["user"][hold_mask], hold_scores)
        hold_graph_rank = _within_user_ranks(train["user"][hold_mask], hold_graph)
        graph_metrics = _metric_values(evaluate(train["user"][hold_mask], train["y"][hold_mask].astype(int), hold_graph_rank))
        progress.write(json.dumps({"member": "rolling_parent", "seed": args.seed + 1, "primary": hold_parent_metrics["primary"]}, sort_keys=True) + "\n")
        progress.write(json.dumps({"member": "rolling_graph", "seed": graph_seed, "primary": graph_metrics["primary"]}, sort_keys=True) + "\n")
        blend_alphas = [0.05, 0.1, 0.2]
        best_blend_alpha = blend_alphas[0]
        best_hold_blend = None
        for blend_alpha in blend_alphas:
            blend = hold_parent_rank + blend_alpha * hold_graph_rank
            metrics = _metric_values(evaluate(train["user"][hold_mask], train["y"][hold_mask].astype(int), blend))
            record = {
                "stage": "rolling_blend_probe",
                "alpha": blend_alpha,
                "parent_seed": args.seed + 1,
                "graph_seed": graph_seed,
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
            }
            history.append(record)
            progress.write(json.dumps({"member": "rolling_blend", "alpha": blend_alpha, "primary": metrics["primary"]}, sort_keys=True) + "\n")
            progress.flush()
            if best_hold_blend is None or metrics["primary"] > best_hold_blend["primary"] + 1e-12:
                best_hold_blend = metrics
                best_blend_alpha = blend_alpha
        final_graph = _signed_sketch(
            train["X"], train["y"], train["date"], val["X"], train["field_dims"], graph_seed
        )
        assert not np.allclose(best_scores, final_graph)
        parent_rank = _within_user_ranks(val["user"], best_scores)
        graph_rank = _within_user_ranks(val["user"], final_graph)
        final_scores = parent_rank + best_blend_alpha * graph_rank
        assert not np.allclose(final_scores, best_scores)
        final_metrics = _metric_values(evaluate(val["user"], val["y"].astype(int), final_scores))
        final_graph_metrics = _metric_values(evaluate(val["user"], val["y"].astype(int), graph_rank))
        progress.write(json.dumps({"member": "validation_parent", "seed": args.seed, "primary": best_metrics["primary"]}, sort_keys=True) + "\n")
        progress.write(json.dumps({"member": "validation_graph", "seed": graph_seed, "primary": final_graph_metrics["primary"]}, sort_keys=True) + "\n")
        progress.write(json.dumps({"member": "validation_blend", "alpha": best_blend_alpha, "primary": final_metrics["primary"]}, sort_keys=True) + "\n")
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "best_config": best_config,
        "blend_alpha": best_blend_alpha,
        "parent_primary": best_metrics["primary"],
        "graph_primary": final_graph_metrics["primary"],
        "rolling_parent_primary": hold_parent_metrics["primary"],
        "rolling_graph_primary": graph_metrics["primary"],
        "rolling_blend_primary": best_hold_blend["primary"],
        "parent_seed": args.seed,
        "graph_seed": graph_seed,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{val['user'][i]},{val['video_raw'][i]},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
