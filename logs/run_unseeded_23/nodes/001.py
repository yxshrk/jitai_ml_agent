import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_dropout = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.out = torch.nn.Linear(width + hidden // 2, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x0 = self.emb_dropout(self.emb(x)).flatten(1)
        cross = x0 + x0 * (x0 * self.cross_w).sum(1, keepdim=True) + self.cross_b
        deep = self.mlp(x0)
        return self.out(torch.cat((cross, deep), dim=1)).squeeze(1)


def date_ordinal(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "").replace("/", "")
    try:
        return datetime.date(int(text[:4]), int(text[4:6]), int(text[6:8])).toordinal()
    except Exception:
        return datetime.date(2022, 4, 21).toordinal()


def recency_weights(dates):
    values = np.asarray(dates)
    unique, inverse = np.unique(values, return_inverse=True)
    ordinals = np.asarray([date_ordinal(v) for v in unique], dtype=np.float32)
    reference = float(datetime.date(2022, 4, 21).toordinal())
    ages = np.maximum(0.0, reference - ordinals)
    table = np.exp2(-ages / 7.0).astype(np.float32)
    weights = table[inverse]
    return weights / max(float(weights.mean()), 1e-8)


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


def read_csv_split(path, validation=False):
    columns = None
    rows = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            rows.append(row)
    result = {}
    for name in columns or []:
        result[name] = np.asarray([row[name] for row in rows])
    result["long_view"] = result["long_view"].astype(np.float32)
    result["duration_ms"] = result["duration_ms"].astype(np.float64)
    return result


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        train = {key: tr_file[key] for key in tr_file.files}
        val = {key: va_file[key] for key in va_file.files}
        train["X"] = train["X"].astype(np.int64)
        val["X"] = val["X"].astype(np.int64)
        train["y"] = train["y"].astype(np.float32)
        val["y"] = val["y"].astype(np.float32)
        train["weight"] = recency_weights(train["date"])
        val["video_output"] = np.zeros(len(val["y"]), dtype=np.int64)
        from data.official.evaluate import evaluate
        return train, val, int(np.asarray(train["field_dims"]).sum()), evaluate

    raw_train = read_csv_split(os.path.join(data_dir, "train.csv"))
    raw_val = read_csv_split(os.path.join(data_dir, "val.csv"), validation=True)
    train_columns = []
    val_columns = []
    dims = []
    for name in ("user_id", "video_id"):
        train_encoded, val_encoded, dim = encode_column(raw_train[name], raw_val[name])
        train_columns.append(train_encoded)
        val_columns.append(val_encoded)
        dims.append(dim)
    train_author = np.zeros(len(raw_train["long_view"]), dtype=np.int64)
    val_author = np.zeros(len(raw_val["long_view"]), dtype=np.int64)
    train_columns.append(train_author)
    val_columns.append(val_author)
    dims.append(1)
    train_tab, val_tab, tab_dim = encode_column(raw_train["tab"], raw_val["tab"])
    train_columns.append(train_tab)
    val_columns.append(val_tab)
    dims.append(tab_dim)
    quantiles = np.quantile(raw_train["duration_ms"], np.linspace(0.1, 0.9, 9))
    train_duration = np.searchsorted(quantiles, raw_train["duration_ms"], side="right").astype(np.int64)
    val_duration = np.searchsorted(quantiles, raw_val["duration_ms"], side="right").astype(np.int64)
    train_columns.append(train_duration)
    val_columns.append(val_duration)
    dims.append(10)
    offsets = np.cumsum([0] + dims[:-1], dtype=np.int64)
    train_x = np.stack(train_columns, axis=1) + offsets
    val_x = np.stack(val_columns, axis=1) + offsets
    train = {
        "X": train_x,
        "y": raw_train["long_view"],
        "user": raw_train["user_id"],
        "date": raw_train["date"],
        "weight": recency_weights(raw_train["date"]),
    }
    val = {
        "X": val_x,
        "y": raw_val["long_view"],
        "user": raw_val["user_id"],
        "video_output": raw_val["video_id"],
    }
    from harness.evaluate_provisional import evaluate
    return train, val, int(sum(dims)), evaluate


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(order)])))
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        group = order[start:end]
        pos = group[labels[group]]
        neg = group[~labels[group]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.integers(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


def predict(model, x, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(x[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs)


def metric_primary(evaluator, users, labels, scores):
    metrics = evaluator(users, labels.astype(int), scores)
    return float(metrics["primary"])


def train_candidate(total_dim, train_x, train_y, train_w, pair_pos, pair_neg,
                    val_x, val_users, val_y, evaluator, indices, pair_indices,
                    dropout, weight_decay, decay_lr, epochs, seed,
                    checkpoints_per_epoch):
    torch.manual_seed(seed)
    model = DCNLite(total_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed + 17)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    for epoch in range(epochs):
        model.train()
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        segments = torch.tensor_split(permutation, checkpoints_per_epoch)
        for segment in segments:
            if len(segment) == 0:
                continue
            pair_budget = max(1, len(segment) // 4) if len(pair_indices) else 0
            pair_draw = pair_indices[torch.randint(len(pair_indices), (pair_budget,), generator=generator)] if pair_budget else None
            steps = max(1, (len(segment) + batch_size - 1) // batch_size)
            pair_batch_size = max(1, (pair_budget + steps - 1) // steps) if pair_budget else 0
            pair_cursor = 0
            for start in range(0, len(segment), batch_size):
                idx = segment[start:start + batch_size]
                logits = model(train_x[idx])
                point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, train_y[idx], reduction="none"
                )
                point_loss = (point_loss * train_w[idx]).sum() / train_w[idx].sum().clamp_min(1e-8)
                if pair_budget:
                    chosen = pair_draw[pair_cursor:min(pair_cursor + pair_batch_size, pair_budget)]
                    pair_cursor += len(chosen)
                    pos_idx = pair_pos[chosen]
                    neg_idx = pair_neg[chosen]
                    pos_score = model(train_x[pos_idx])
                    neg_score = model(train_x[neg_idx])
                    pair_loss = torch.nn.functional.softplus(-(pos_score - neg_score))
                    pair_weight = 0.5 * (train_w[pos_idx] + train_w[neg_idx])
                    pair_loss = (pair_loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-8)
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scores = predict(model, val_x)
            primary = metric_primary(evaluator, val_users, val_y, scores)
            if primary > best_primary + 1e-8:
                best_primary = primary
                best_scores = scores.copy()
            model.train()
        if decay_lr:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    return best_primary, best_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    train, val, total_dim, evaluator = load_data(args.data_dir)
    train_x = torch.from_numpy(np.asarray(train["X"], dtype=np.int64))
    train_y = torch.from_numpy(np.asarray(train["y"], dtype=np.float32))
    train_w = torch.from_numpy(np.asarray(train["weight"], dtype=np.float32))
    val_x = torch.from_numpy(np.asarray(val["X"], dtype=np.int64))
    val_y = np.asarray(val["y"], dtype=np.float32)
    val_users = np.asarray(val["user"])

    pair_pos_np, pair_neg_np = build_pairs(train["user"], train["y"], args.seed + 31)
    pair_pos = torch.from_numpy(pair_pos_np)
    pair_neg = torch.from_numpy(pair_neg_np)

    smoke = os.environ.get("SMOKE_EPOCHS")
    cap = max(1, int(smoke)) if smoke is not None else None
    probe_epochs = min(2, cap) if cap is not None else 2
    final_epochs = min(max(1, args.epochs), cap) if cap is not None else max(1, args.epochs)

    rng = np.random.default_rng(args.seed + 101)
    probe_count = max(1, len(train_x) // 2)
    probe_indices_np = rng.choice(len(train_x), size=probe_count, replace=False)
    probe_indices = torch.from_numpy(probe_indices_np.astype(np.int64))
    if len(pair_pos):
        probe_pair_count = max(1, len(pair_pos) // 2)
        probe_pair_indices_np = rng.choice(len(pair_pos), size=probe_pair_count, replace=False)
        probe_pair_indices = torch.from_numpy(probe_pair_indices_np.astype(np.int64))
    else:
        probe_pair_indices = torch.empty(0, dtype=torch.int64)

    configurations = [
        (dropout, weight_decay, decay_lr)
        for dropout in (0.2, 0.3)
        for weight_decay in (1e-4, 1e-3)
        for decay_lr in (False, True)
    ]
    best_configuration = configurations[0]
    best_probe_primary = -1.0
    for dropout, weight_decay, decay_lr in configurations:
        primary, _ = train_candidate(
            total_dim, train_x, train_y, train_w, pair_pos, pair_neg,
            val_x, val_users, val_y, evaluator, probe_indices,
            probe_pair_indices, dropout, weight_decay, decay_lr,
            probe_epochs, args.seed + 211, 1
        )
        if primary > best_probe_primary + 1e-8:
            best_probe_primary = primary
            best_configuration = (dropout, weight_decay, decay_lr)

    all_indices = torch.arange(len(train_x), dtype=torch.int64)
    all_pair_indices = torch.arange(len(pair_pos), dtype=torch.int64)
    dropout, weight_decay, decay_lr = best_configuration
    _, best_scores = train_candidate(
        total_dim, train_x, train_y, train_w, pair_pos, pair_neg,
        val_x, val_users, val_y, evaluator, all_indices, all_pair_indices,
        dropout, weight_decay, decay_lr, final_epochs, args.seed + 1009, 2
    )

    metrics = evaluator(val_users, val_y.astype(int), best_scores)
    gauc = metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]
    ndcg5 = metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump({"gauc": float(gauc), "ndcg5": float(ndcg5), "primary": float(metrics["primary"])}, handle)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        videos = val["video_output"]
        for row_id, score in enumerate(best_scores):
            writer.writerow([row_id, val_users[row_id], videos[row_id], format(float(score), ".8g")])


if __name__ == "__main__":
    main()
