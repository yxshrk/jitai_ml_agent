import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2, cross_layers=2):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(width)) for _ in range(cross_layers)
        ])
        self.cross_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(width)) for _ in range(cross_layers)
        ])
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
        torch.nn.init.zeros_(self.linear.weight)
        for weight in self.cross_w:
            torch.nn.init.normal_(weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        embeddings = self.emb_drop(self.emb(x))
        x0 = embeddings.reshape(embeddings.shape[0], -1)
        cross = x0
        for weight, bias in zip(self.cross_w, self.cross_b):
            scale = (cross * weight).sum(dim=1, keepdim=True)
            cross = cross + x0 * scale + bias
        deep = self.mlp(x0)
        nonlinear = self.out(torch.cat((cross, deep), dim=1)).squeeze(1)
        return self.bias + self.linear(x).sum(dim=(1, 2)) + nonlinear


def date_ordinals(values):
    values = np.asarray(values)
    unique = np.unique(values)
    parsed = {}
    valid = True
    for value in unique:
        raw = value.item() if hasattr(value, "item") else value
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        text = str(raw)
        try:
            if "-" in text:
                day = datetime.date.fromisoformat(text[:10])
            else:
                text = str(int(float(text))).zfill(8)
                day = datetime.datetime.strptime(text, "%Y%m%d").date()
            parsed[value] = day.toordinal()
        except (ValueError, TypeError, OverflowError):
            valid = False
            break
    if not valid:
        ordered = sorted(unique.tolist())
        parsed = {value: index for index, value in enumerate(ordered)}
    return np.asarray([parsed[value] for value in values], dtype=np.float32)


def make_pairs(users, labels, seed):
    users = np.asarray(users)
    positive_mask = np.asarray(labels) > 0.5
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for start, end in zip(starts, ends):
        rows = order[start:end]
        pos = rows[positive_mask[rows]]
        neg = rows[~positive_mask[rows]]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(rng.choice(neg, size=len(pos), replace=True))
    if not positives:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        torch.from_numpy(np.concatenate(positives).astype(np.int64)),
        torch.from_numpy(np.concatenate(negatives).astype(np.int64)),
    )


def predict(model, X, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def per_user_rank_normalize(users, scores):
    users = np.asarray(users)
    scores = np.asarray(scores)
    result = np.empty(len(scores), dtype=np.float64)
    user_order = np.argsort(users, kind="mergesort")
    sorted_users = users[user_order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(user_order)]))
    for start, end in zip(starts, ends):
        rows = user_order[start:end]
        if len(rows) == 1:
            result[rows[0]] = 0.5
            continue
        local_order = np.argsort(scores[rows], kind="mergesort")
        local_ranks = np.empty(len(rows), dtype=np.float64)
        local_ranks[local_order] = np.arange(len(rows), dtype=np.float64)
        result[rows] = local_ranks / float(len(rows) - 1)
    return result


def metric_values(evaluator, users, labels, scores):
    metrics = evaluator(users, labels, scores)
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def train_member(config, seed, epochs, Xt, yt, recency_age, Xv, val_users,
                 val_labels, total_dim, evaluator, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DCNLite(
        total_dim=total_dim,
        fields=Xt.shape[1],
        k=16,
        hidden=128,
        dropout=config["dropout"],
        cross_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    weights = torch.from_numpy(
        np.exp2(-recency_age / config["half_life"]).astype(np.float32)
    )
    pair_pos, pair_neg = make_pairs(val_users if False else train_member.train_users,
                                    yt.numpy(), seed + 31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729)
    batch_size = 8192
    checks_per_epoch = 2
    best_scores = None
    best_metrics = None
    curve = []
    n = len(yt)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        boundaries = np.linspace(0, n, checks_per_epoch + 1, dtype=np.int64)
        running_loss = 0.0
        running_batches = 0

        for part in range(checks_per_epoch):
            section = permutation[boundaries[part]:boundaries[part + 1]]
            for offset in range(0, len(section), batch_size):
                idx = section[offset:offset + batch_size]
                if len(idx) == 0:
                    continue
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                wb = weights[idx].to(device, non_blocking=True)

                pair_count = min(max(1, len(idx) // 2), len(pair_pos))
                if pair_count > 0:
                    chosen = torch.randint(
                        len(pair_pos), (pair_count,), generator=generator
                    )
                    pos_idx = pair_pos[chosen]
                    neg_idx = pair_neg[chosen]
                    xp = Xt[pos_idx].to(device, non_blocking=True)
                    xn = Xt[neg_idx].to(device, non_blocking=True)
                    all_scores = model(torch.cat((xb, xp, xn), dim=0))
                    base_end = len(idx)
                    pos_end = base_end + pair_count
                    logits = all_scores[:base_end]
                    pos_scores = all_scores[base_end:pos_end]
                    neg_scores = all_scores[pos_end:]
                    point_losses = F.binary_cross_entropy_with_logits(
                        logits, yb, reduction="none"
                    )
                    point_loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-6)
                    pair_weights = weights[pos_idx].to(device, non_blocking=True)
                    pair_losses = F.softplus(-(pos_scores - neg_scores))
                    pair_loss = (
                        (pair_losses * pair_weights).sum()
                        / pair_weights.sum().clamp_min(1e-6)
                    )
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    logits = model(xb)
                    point_losses = F.binary_cross_entropy_with_logits(
                        logits, yb, reduction="none"
                    )
                    loss = (point_losses * wb).sum() / wb.sum().clamp_min(1e-6)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                running_batches += 1

            scores = predict(model, Xv, device)
            metrics = metric_values(evaluator, val_users, val_labels, scores)
            record = {
                "epoch": epoch + (part + 1) / checks_per_epoch,
                "train_loss": round(running_loss / max(1, running_batches), 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            }
            curve.append(record)
            if best_metrics is None or metrics["primary"] > best_metrics["primary"]:
                best_metrics = metrics
                best_scores = scores.copy()
            model.train()

        if (epoch + 1) % config["step_size"] == 0:
            for group in optimizer.param_groups:
                group["lr"] *= config["gamma"]

    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, best_metrics, curve


def read_csv_rows(path, training):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": row["date"],
                "label": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def encode_csv(train_rows, val_rows):
    durations = np.asarray([row["duration"] for row in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_names = ["user", "video", "author", "tab"]
    mappings = {}
    dimensions = []
    for name in field_names:
        values = sorted({row[name] for row in train_rows})
        mapping = {value: index + 1 for index, value in enumerate(values)}
        mappings[name] = mapping
        dimensions.append(len(mapping) + 1)
    dimensions.append(len(edges) + 1)
    field_dims = np.asarray(dimensions, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def transform(rows):
        X = np.empty((len(rows), 5), dtype=np.int64)
        for index, row in enumerate(rows):
            for column, name in enumerate(field_names):
                X[index, column] = mappings[name].get(row[name], 0) + offsets[column]
            bucket = int(np.searchsorted(edges, row["duration"], side="right"))
            X[index, 4] = bucket + offsets[4]
        return X

    return transform(train_rows), transform(val_rows), field_dims


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        from data.official.evaluate import evaluate
        train = np.load(train_npz)
        val = np.load(val_npz)
        field_dims = train["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "X_train": train["X"].astype(np.int64, copy=False),
            "y_train": train["y"].astype(np.float32, copy=False),
            "train_users": train["user"],
            "train_dates": train["date"],
            "X_val": val["X"].astype(np.int64, copy=False),
            "y_val": val["y"].astype(np.int64, copy=False),
            "val_users": val["user"],
            "val_videos": val["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
            "evaluator": evaluate,
        }

    from harness.evaluate_provisional import evaluate
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    X_train, X_val, field_dims = encode_csv(train_rows, val_rows)
    return {
        "X_train": X_train,
        "y_train": np.asarray([row["label"] for row in train_rows], dtype=np.float32),
        "train_users": np.asarray([row["user"] for row in train_rows]),
        "train_dates": np.asarray([row["date"] for row in train_rows]),
        "X_val": X_val,
        "y_val": np.asarray([row["label"] for row in val_rows], dtype=np.int64),
        "val_users": np.asarray([row["user"] for row in val_rows]),
        "val_videos": np.asarray([row["video"] for row in val_rows]),
        "field_dims": field_dims,
        "evaluator": evaluate,
    }


def append_progress(path, payload):
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data_dir)
    Xt = torch.from_numpy(data["X_train"])
    yt = torch.from_numpy(data["y_train"])
    Xv = torch.from_numpy(data["X_val"])
    train_days = date_ordinals(data["train_dates"])
    recency_age = train_days.max() - train_days
    total_dim = int(data["field_dims"].sum())
    train_member.train_users = data["train_users"]

    config = {
        "dropout": 0.2,
        "weight_decay": 0.0001,
        "lr": 0.0007,
        "step_size": 2,
        "gamma": 0.5,
        "half_life": 7.0,
    }

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    member_count = 5 if smoke_cap is None else 1

    member_scores = []
    member_records = []
    for member in range(member_count):
        member_seed = args.seed + member
        scores, metrics, curve = train_member(
            config=config,
            seed=member_seed,
            epochs=epochs,
            Xt=Xt,
            yt=yt,
            recency_age=recency_age,
            Xv=Xv,
            val_users=data["val_users"],
            val_labels=data["y_val"],
            total_dim=total_dim,
            evaluator=data["evaluator"],
            device=device,
        )
        member_scores.append(scores)
        record = {
            "stage": "final_member",
            "member": member,
            "seed": member_seed,
            "config": config,
            "metrics": metrics,
            "curve": curve,
        }
        member_records.append(record)
        append_progress(progress_path, {
            "stage": "final_member",
            "member": member,
            "seed": member_seed,
            "primary": metrics["primary"],
        })

    normalized = [
        per_user_rank_normalize(data["val_users"], scores)
        for scores in member_scores
    ]
    running = np.zeros(len(Xv), dtype=np.float64)
    prefix_history = []
    selected_scores = None
    selected_metrics = None
    selected_count = 0
    for index, scores in enumerate(normalized):
        running += scores
        ensemble_scores = running / float(index + 1)
        metrics = metric_values(
            data["evaluator"], data["val_users"], data["y_val"], ensemble_scores
        )
        prefix_history.append({"members": index + 1, "metrics": metrics})
        if selected_metrics is None or metrics["primary"] > selected_metrics["primary"]:
            selected_metrics = metrics
            selected_scores = ensemble_scores.copy()
            selected_count = index + 1

    output_metrics = {
        "gauc": selected_metrics["gauc"],
        "ndcg5": selected_metrics["ndcg5"],
        "primary": selected_metrics["primary"],
        "history": member_records,
        "winning_config": config,
        "ensemble_prefixes": prefix_history,
        "selected_ensemble_members": selected_count,
        "rank_aggregation": "within_user",
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(output_metrics, handle)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, score in enumerate(selected_scores):
            writer.writerow([
                row_id,
                data["val_users"][row_id],
                data["val_videos"][row_id],
                format(float(score), ".8g"),
            ])


if __name__ == "__main__":
    main()
