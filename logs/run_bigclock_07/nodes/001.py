import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ArrayData:
    pass


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, architecture, dropout, k=16):
        super().__init__()
        self.architecture = architecture
        self.dropout = torch.nn.Dropout(dropout)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            width = 5 * k
            self.cross_w = torch.nn.Parameter(torch.empty(width))
            self.cross_b = torch.nn.Parameter(torch.zeros(width))
            self.cross_head = torch.nn.Linear(width, 1, bias=False)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(width, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 1),
            )
            torch.nn.init.normal_(self.cross_w, std=0.01)
            torch.nn.init.xavier_uniform_(self.cross_head.weight)
            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            torch.nn.init.zeros_(self.mlp[0].bias)
            torch.nn.init.xavier_uniform_(self.mlp[3].weight)
            torch.nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        linear = self.lin(x).sum((1, 2)) + self.bias
        if self.architecture == "fm":
            summed = e.sum(1)
            pair = 0.5 * (summed * summed - (e * e).sum(1)).sum(1)
            return linear + pair
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 + x0 * torch.matmul(x0, self.cross_w).unsqueeze(1) + self.cross_b
        return linear + self.cross_head(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def date_ordinals(values):
    result = np.zeros(len(values), dtype=np.int32)
    cache = {}
    for i, value in enumerate(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        if text not in cache:
            try:
                cache[text] = datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
            except Exception:
                cache[text] = 0
        result[i] = cache[text]
    return result


def recency_weights(dates):
    ordinals = date_ordinals(dates)
    valid = ordinals > 0
    if not np.any(valid):
        return np.ones(len(dates), dtype=np.float32)
    reference = max(datetime.date(2022, 4, 21).toordinal(), int(ordinals[valid].max()))
    age = np.maximum(0, reference - ordinals)
    weights = np.power(0.5, age.astype(np.float64) / 7.0).astype(np.float32)
    weights[~valid] = 1.0
    return weights


def encode_column(train_values, val_values):
    mapping = {}
    encoded_train = np.empty(len(train_values), dtype=np.int32)
    for i, value in enumerate(train_values):
        key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
        encoded_train[i] = mapping[key]
    unknown = len(mapping)
    encoded_val = np.asarray([mapping.get(str(v), unknown) for v in val_values], dtype=np.int32)
    return encoded_train, encoded_val, unknown + 1


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_columns = {k: [] for k in ["user_id", "video_id", "author_id", "tab", "duration_ms", "date", "long_view"]}
    val_columns = {k: [] for k in ["user_id", "video_id", "author_id", "tab", "duration_ms", "date", "long_view"]}
    with open(train_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_columns["user_id"].append(row["user_id"])
            train_columns["video_id"].append(row["video_id"])
            train_columns["author_id"].append(row["author_id"] if has_author else row["video_id"])
            train_columns["tab"].append(row["tab"])
            train_columns["duration_ms"].append(float(row["duration_ms"]))
            train_columns["date"].append(row["date"])
            train_columns["long_view"].append(float(row["long_view"]))
    with open(val_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_columns["user_id"].append(row["user_id"])
            val_columns["video_id"].append(row["video_id"])
            val_columns["author_id"].append(row["author_id"] if has_author else row["video_id"])
            val_columns["tab"].append(row["tab"])
            val_columns["duration_ms"].append(float(row["duration_ms"]))
            val_columns["date"].append(row["date"])
            val_columns["long_view"].append(float(row["long_view"]))
    encoded_train = []
    encoded_val = []
    dims = []
    for name in ["user_id", "video_id", "author_id", "tab"]:
        tr_col, va_col, dim = encode_column(train_columns[name], val_columns[name])
        encoded_train.append(tr_col)
        encoded_val.append(va_col)
        dims.append(dim)
    train_duration = np.asarray(train_columns["duration_ms"], dtype=np.float64)
    val_duration = np.asarray(val_columns["duration_ms"], dtype=np.float64)
    edges = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    encoded_train.append(np.searchsorted(edges, train_duration, side="right").astype(np.int32))
    encoded_val.append(np.searchsorted(edges, val_duration, side="right").astype(np.int32))
    dims.append(len(edges) + 1)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    tr = ArrayData()
    va = ArrayData()
    tr.X = np.column_stack(encoded_train).astype(np.int32) + offsets
    va.X = np.column_stack(encoded_val).astype(np.int32) + offsets
    tr.y = np.asarray(train_columns["long_view"], dtype=np.float32)
    va.y = np.asarray(val_columns["long_view"], dtype=np.float32)
    tr.user = np.asarray(train_columns["user_id"])
    va.user = np.asarray(val_columns["user_id"])
    tr.video = np.asarray(train_columns["video_id"])
    va.video = np.asarray(val_columns["video_id"])
    tr.date = np.asarray(train_columns["date"])
    va.date = np.asarray(val_columns["date"])
    tr.field_dims = np.asarray(dims, dtype=np.int64)
    va.field_dims = tr.field_dims
    return tr, va, "csv"


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        train_file = np.load(train_npz)
        val_file = np.load(val_npz)
        tr = ArrayData()
        va = ArrayData()
        tr.X = train_file["X"].astype(np.int32)
        va.X = val_file["X"].astype(np.int32)
        tr.y = train_file["y"].astype(np.float32)
        va.y = val_file["y"].astype(np.float32)
        tr.user = train_file["user"]
        va.user = val_file["user"]
        tr.date = train_file["date"]
        va.date = val_file["date"]
        tr.field_dims = train_file["field_dims"].astype(np.int64)
        va.field_dims = tr.field_dims
        if "video" in val_file.files:
            va.video = val_file["video"]
        else:
            va.video = va.X[:, 1].astype(np.int64) - int(tr.field_dims[0])
        return tr, va, "npz"
    return load_csv_data(data_dir)


def make_pairs(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    pos_parts = []
    neg_parts = []
    for j in range(len(boundaries) - 1):
        group = order[boundaries[j]:boundaries[j + 1]]
        positive = group[labels[group] > 0.5]
        negative = group[labels[group] <= 0.5]
        if len(positive) and len(negative):
            pos_parts.append(positive)
            neg_parts.append(np.resize(negative, len(positive)))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def predict(model, X):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            chunks.append(model(X[start:start + 65536]).cpu().numpy())
    return np.concatenate(chunks)


def make_model(total_dim, config, seed):
    torch.manual_seed(seed)
    dropout = 0.30 if config["regularization"] == "strong" else 0.0
    return RankModel(total_dim, config["architecture"], dropout)


def train_epochs(model, X, y, users, weights, config, epochs, seed, batch_size=8192):
    strong = config["regularization"] == "strong"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3 if strong else 0.0)
    pair_pos, pair_neg = make_pairs(users, y.numpy()) if config["loss"] == "bpr-hybrid" else (None, None)
    pair_pos_t = torch.from_numpy(pair_pos) if pair_pos is not None else None
    pair_neg_t = torch.from_numpy(pair_neg) if pair_neg is not None else None
    generator = torch.Generator().manual_seed(seed + 991)
    last_loss = 0.0
    n = len(y)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X[idx])
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx], reduction="none")
            point_loss = (point_loss * weights[idx]).sum() / weights[idx].sum().clamp_min(1e-6)
            if config["loss"] == "bpr-hybrid" and len(pair_pos_t):
                pair_batch = min(2048, max(1, len(idx) // 2))
                selected = torch.randint(len(pair_pos_t), (pair_batch,), generator=generator)
                pos_idx = pair_pos_t[selected]
                neg_idx = pair_neg_t[selected]
                difference = model(X[pos_idx]) - model(X[neg_idx])
                pair_weight = 0.5 * (weights[pos_idx] + weights[neg_idx])
                rank_loss = (torch.nn.functional.softplus(-difference) * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
                loss = 0.5 * point_loss + 0.5 * rank_loss
            else:
                loss = point_loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
        if strong:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    return last_loss


def run_probe(config, total_dim, X, y, users, weights, Xv, val_users, val_y, evaluator, epochs, seed):
    arch_seed = seed + (1000 if config["architecture"] == "dcn-lite" else 0)
    model = make_model(total_dim, config, arch_seed)
    chosen_weights = weights if config["weighting"] == "recency-7d" else torch.ones_like(weights)
    loss = train_epochs(model, X, y, users, chosen_weights, config, epochs, arch_seed)
    scores = predict(model, Xv)
    metrics = metric_values(evaluator(val_users, val_y, scores))
    return metrics, loss


def train_final(config, total_dim, X, y, users, weights, Xv, val_users, val_y, evaluator, epochs, seed):
    model = make_model(total_dim, config, seed)
    strong = config["regularization"] == "strong"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3 if strong else 0.0)
    selected_weights = weights if config["weighting"] == "recency-7d" else torch.ones_like(weights)
    pair_pos, pair_neg = make_pairs(users, y.numpy()) if config["loss"] == "bpr-hybrid" else (None, None)
    pair_pos_t = torch.from_numpy(pair_pos) if pair_pos is not None else None
    pair_neg_t = torch.from_numpy(pair_neg) if pair_neg is not None else None
    generator = torch.Generator().manual_seed(seed + 7331)
    best_primary = -1.0
    best_scores = None
    history = []
    patience = 0
    n = len(y)
    stop = False
    for epoch in range(epochs):
        permutation = torch.randperm(n, generator=generator)
        halves = [permutation[: (n + 1) // 2], permutation[(n + 1) // 2:]]
        for half_number, half in enumerate(halves, 1):
            if len(half) == 0:
                continue
            model.train()
            last_loss = 0.0
            for start in range(0, len(half), 8192):
                idx = half[start:start + 8192]
                optimizer.zero_grad()
                logits = model(X[idx])
                point = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx], reduction="none")
                point = (point * selected_weights[idx]).sum() / selected_weights[idx].sum().clamp_min(1e-6)
                if config["loss"] == "bpr-hybrid" and len(pair_pos_t):
                    pair_batch = min(2048, max(1, len(idx) // 2))
                    selected = torch.randint(len(pair_pos_t), (pair_batch,), generator=generator)
                    pi = pair_pos_t[selected]
                    ni = pair_neg_t[selected]
                    difference = model(X[pi]) - model(X[ni])
                    pair_weight = 0.5 * (selected_weights[pi] + selected_weights[ni])
                    rank = (torch.nn.functional.softplus(-difference) * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)
                    loss = 0.5 * point + 0.5 * rank
                else:
                    loss = point
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().item())
            scores = predict(model, Xv)
            measured = metric_values(evaluator(val_users, val_y, scores))
            history.append({
                "phase": "final",
                "epoch": epoch + 0.5 * half_number,
                "train_loss": round(last_loss, 6),
                "val_gauc": round(measured["gauc"], 6),
                "val_primary": round(measured["primary"], 6),
            })
            if measured["primary"] > best_primary + 1e-6:
                best_primary = measured["primary"]
                best_scores = scores.copy()
                patience = 0
            else:
                patience += 1
                if patience >= 4:
                    stop = True
                    break
        if strong:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
        if stop:
            break
    return best_scores, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke) if smoke is not None else None
    final_epochs = args.epochs if smoke_epochs is None else min(args.epochs, smoke_epochs)
    probe_epochs = 2 if smoke_epochs is None else min(2, smoke_epochs)
    refine_epochs = 4 if smoke_epochs is None else min(4, smoke_epochs)
    tr, va, source = load_data(args.data_dir)
    if source == "npz":
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator
    total_dim = int(np.asarray(tr.field_dims).sum())
    Xt = torch.from_numpy(tr.X.astype(np.int64))
    yt = torch.from_numpy(tr.y.astype(np.float32))
    Xv = torch.from_numpy(va.X.astype(np.int64))
    all_recency = torch.from_numpy(recency_weights(tr.date))
    n = len(yt)
    probe_limit = 50000 if smoke_epochs is not None else 300000
    rng = np.random.RandomState(args.seed + 17)
    if n > probe_limit:
        probe_indices = np.sort(rng.choice(n, probe_limit, replace=False)).astype(np.int64)
    else:
        probe_indices = np.arange(n, dtype=np.int64)
    probe_t = torch.from_numpy(probe_indices)
    probe_X = Xt[probe_t]
    probe_y = yt[probe_t]
    probe_users = np.asarray(tr.user)[probe_indices]
    probe_weights = all_recency[probe_t]
    configs = []
    for architecture in ["fm", "dcn-lite"]:
        for loss_name in ["logloss", "bpr-hybrid"]:
            for weighting in ["uniform", "recency-7d"]:
                for regularization in ["mild", "strong"]:
                    configs.append({
                        "architecture": architecture,
                        "loss": loss_name,
                        "weighting": weighting,
                        "regularization": regularization,
                    })
    history = []
    probe_results = []
    val_labels = va.y.astype(int)
    for cell, config in enumerate(configs):
        measured, last_loss = run_probe(
            config, total_dim, probe_X, probe_y, probe_users, probe_weights,
            Xv, va.user, val_labels, evaluator, probe_epochs, args.seed
        )
        entry = {
            "phase": "matrix_probe",
            "cell": cell,
            "config": config,
            "epochs": probe_epochs,
            "rows": int(len(probe_y)),
            "train_loss": round(last_loss, 6),
            "gauc": round(measured["gauc"], 6),
            "ndcg5": round(measured["ndcg5"], 6),
            "primary": round(measured["primary"], 6),
        }
        history.append(entry)
        probe_results.append((measured["primary"], cell, config))
    probe_results.sort(key=lambda item: (-item[0], item[1]))
    top_count = 1 if smoke_epochs is not None else 3
    refined = []
    for _, cell, config in probe_results[:top_count]:
        measured, last_loss = run_probe(
            config, total_dim, probe_X, probe_y, probe_users, probe_weights,
            Xv, va.user, val_labels, evaluator, refine_epochs, args.seed
        )
        entry = {
            "phase": "refinement",
            "source_cell": cell,
            "config": config,
            "epochs": refine_epochs,
            "rows": int(len(probe_y)),
            "train_loss": round(last_loss, 6),
            "gauc": round(measured["gauc"], 6),
            "ndcg5": round(measured["ndcg5"], 6),
            "primary": round(measured["primary"], 6),
        }
        history.append(entry)
        refined.append((measured["primary"], cell, config))
    refined.sort(key=lambda item: (-item[0], item[1]))
    winning_config = refined[0][2]
    best_scores, final_history = train_final(
        winning_config, total_dim, Xt, yt, np.asarray(tr.user), all_recency,
        Xv, va.user, val_labels, evaluator, final_epochs, args.seed
    )
    history.extend(final_history)
    final_metrics = metric_values(evaluator(va.user, val_labels, best_scores))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": final_metrics["gauc"],
            "ndcg5": final_metrics["ndcg5"],
            "primary": final_metrics["primary"],
            "winning_config": winning_config,
            "history": history,
        }, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, va.user[i], va.video[i], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
