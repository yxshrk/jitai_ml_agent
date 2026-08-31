"""Validation-selected cross-mechanism ensemble for within-user ranking."""
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

    def relative_logit(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def forward(self, x):
        return self.relative_logit(x) + self.bias


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        dim = fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.cross_out = torch.nn.Linear(dim, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.zeros_(self.mlp[0].bias)
        torch.nn.init.xavier_uniform_(self.mlp[3].weight)
        torch.nn.init.zeros_(self.mlp[3].bias)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)

    def relative_logit(self, x):
        e = self.emb(x)
        flat = e.flatten(1)
        crossed = flat + flat * torch.sum(flat * self.cross_w, dim=1, keepdim=True) + self.cross_b
        linear = self.lin(x).sum((1, 2))
        return linear + self.cross_out(crossed).squeeze(1) + self.mlp(flat).squeeze(1)

    def forward(self, x):
        return self.relative_logit(x) + self.bias


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def date_to_ordinal(value):
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "")
    try:
        return datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal()
    except (ValueError, TypeError):
        return 0


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            train_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "date": row["date"],
                "y": float(row["long_view"]),
            })
    val_rows = []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            val_rows.append({
                "user": row["user_id"],
                "video": row["video_id"],
                "author": row["author_id"] if has_author else "__NO_AUTHOR__",
                "tab": row["tab"],
                "duration": float(row["duration_ms"]),
                "y": float(row["long_view"]),
            })

    duration_train = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(duration_train, np.arange(1, 10) / 10.0)
    field_names = ("user", "video", "author", "tab")
    mappings = {}
    dims = []
    for field in field_names:
        values = sorted({r[field] for r in train_rows})
        mappings[field] = {value: i + 1 for i, value in enumerate(values)}
        dims.append(len(values) + 1)
    dims.append(10)
    field_dims = np.asarray(dims, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(field_names):
                x[i, j] = mappings[field].get(row[field], 0) + offsets[j]
            bucket = int(np.searchsorted(quantiles, row["duration"], side="right"))
            x[i, 4] = min(bucket, 9) + offsets[4]
        return x

    xt = encode(train_rows)
    xv = encode(val_rows)
    return {
        "Xt": xt,
        "yt": np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        "train_user": xt[:, 0].copy(),
        "train_date": np.asarray([date_to_ordinal(r["date"]) for r in train_rows], dtype=np.int64),
        "Xv": xv,
        "yv": np.asarray([r["y"] for r in val_rows], dtype=np.int64),
        "val_user": np.asarray([r["user"] for r in val_rows]),
        "val_video": np.asarray([r["video"] for r in val_rows]),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        xt = tr["X"].astype(np.int64)
        xv = va["X"].astype(np.int64)
        raw_dates = np.asarray(tr["date"])
        train_dates = np.asarray([date_to_ordinal(v) for v in raw_dates], dtype=np.int64)
        return {
            "Xt": xt,
            "yt": tr["y"].astype(np.float32),
            "train_user": xt[:, 0].copy(),
            "train_date": train_dates,
            "Xv": xv,
            "yv": va["y"].astype(np.int64),
            "val_user": np.asarray(va["user"]),
            "val_video": xv[:, 1] - int(field_dims[0]),
            "field_dims": field_dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(metrics):
    return (
        float(metrics.get("GAUC", metrics.get("gauc"))),
        float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        float(metrics["primary"]),
    )


def make_user_groups(user_ids):
    order = np.argsort(user_ids, kind="stable")
    sorted_users = user_ids[order]
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(order)]))
    return order, starts, ends


def make_complete_user_batches(order, starts, ends, rng, batch_size):
    group_order = rng.permutation(len(starts))
    batches = []
    pieces = []
    count = 0
    for group_index in group_order:
        piece = order[starts[group_index]:ends[group_index]]
        size = len(piece)
        if pieces and count + size > batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
        pieces.append(piece)
        count += size
        if count >= batch_size:
            batches.append(np.concatenate(pieces))
            pieces = []
            count = 0
    if pieces:
        batches.append(np.concatenate(pieces))
    return batches


def centered_logits(model, x, users):
    relative = model.relative_logit(x)
    _, inverse, counts = torch.unique(users, sorted=False, return_inverse=True, return_counts=True)
    sums = torch.zeros(len(counts), dtype=relative.dtype, device=relative.device)
    sums.scatter_add_(0, inverse, relative)
    means = sums / counts.to(relative.dtype)
    return relative - means[inverse] + model.bias


def pair_indices(batch_users, batch_labels, rng):
    boundaries = np.flatnonzero(batch_users[1:] != batch_users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(batch_users)]))
    pos_parts = []
    neg_parts = []
    for start, end in zip(starts, ends):
        local = np.arange(start, end)
        labels = batch_labels[start:end]
        pos = local[labels > 0.5]
        neg = local[labels <= 0.5]
        if len(pos) and len(neg):
            count = max(len(pos), len(neg))
            pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
            neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return None, None
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


def predict(model, xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(xv), 65536):
            xb = xv[start:start + 65536].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def recency_weights(train_dates):
    valid = train_dates[train_dates > 0]
    if len(valid) == 0:
        return np.ones(len(train_dates), dtype=np.float32)
    latest = int(valid.max())
    age = np.maximum(0, latest - train_dates)
    weights = np.exp2(-age.astype(np.float64) / 7.0)
    weights[train_dates <= 0] = 1.0
    weights /= weights.mean()
    return weights.astype(np.float32)


def train_one(data, evaluate, device, seed, epochs, family, groups):
    seed_everything(seed)
    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    users = torch.from_numpy(data["train_user"])
    xv = torch.from_numpy(data["Xv"])
    weights = torch.from_numpy(recency_weights(data["train_date"]))
    total_dim = int(data["field_dims"].sum())
    if family == "regularized_dcn_hybrid":
        model = DCNLite(total_dim, fields=data["Xt"].shape[1], k=16, hidden=128, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    else:
        model = FM(total_dim, k=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None

    order, starts, ends = groups
    rng = np.random.RandomState(seed)
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []

    for epoch in range(epochs):
        model.train()
        batches = make_complete_user_batches(order, starts, ends, rng, 8192)
        loss_sum = 0.0
        steps = 0
        for idx_np in batches:
            idx = torch.from_numpy(idx_np)
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            ub = users[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = centered_logits(model, xb, ub)
            bce_each = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            if family in ("recency_centered_fm", "regularized_dcn_hybrid"):
                wb = weights[idx].to(device, non_blocking=True)
                bce_loss = torch.sum(bce_each * wb) / torch.sum(wb)
            else:
                bce_loss = bce_each.mean()

            if family == "regularized_dcn_hybrid":
                batch_users = data["train_user"][idx_np]
                batch_labels = data["yt"][idx_np]
                pos_np, neg_np = pair_indices(batch_users, batch_labels, rng)
                if pos_np is not None:
                    pos_idx = torch.from_numpy(pos_np).to(device)
                    neg_idx = torch.from_numpy(neg_np).to(device)
                    relative = model.relative_logit(xb)
                    pair_loss = torch.nn.functional.softplus(-(relative[pos_idx] - relative[neg_idx])).mean()
                    loss = 0.5 * bce_loss + 0.5 * pair_loss
                else:
                    loss = bce_loss
            else:
                loss = bce_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1

        if scheduler is not None:
            scheduler.step()
        scores = predict(model, xv, device)
        metrics = evaluate(data["val_user"], data["yv"], scores)
        gauc, ndcg5, primary = metric_values(metrics)
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(steps, 1), 7),
            "val_gauc": round(gauc, 8),
            "val_ndcg5": round(ndcg5, 8),
            "val_primary": round(primary, 8),
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    final_metrics = evaluate(data["val_user"], data["yv"], best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    return {
        "seed": int(seed),
        "family": family,
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_epoch": int(max(epoch_history, key=lambda row: row["val_primary"])["epoch"]),
        "epochs": epoch_history,
    }, best_scores


def sigmoid(values):
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def normalized_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) > 1:
        ranks /= float(len(values) - 1)
    return ranks


def evaluate_candidate(evaluate, data, scores, config):
    metrics = evaluate(data["val_user"], data["yv"], scores)
    gauc, ndcg5, primary = metric_values(metrics)
    record = dict(config)
    record.update({"gauc": gauc, "ndcg5": ndcg5, "primary": primary})
    return record


def select_ensemble(family_members, evaluate, data):
    families = sorted(family_members)
    representations = {"probability": {}, "rank": {}}
    for family in families:
        matrix = np.stack(family_members[family], axis=0)
        representations["probability"][family] = sigmoid(matrix).mean(axis=0)
        representations["rank"][family] = np.stack(
            [normalized_ranks(row) for row in matrix], axis=0
        ).mean(axis=0)

    history = []
    for aggregation in ("probability", "rank"):
        for family in families:
            scores = representations[aggregation][family]
            history.append(evaluate_candidate(evaluate, data, scores, {
                "aggregation": aggregation,
                "families": [family],
                "weights": [1.0],
                "eligible": False,
            }))

        for left in range(len(families)):
            for right in range(left + 1, len(families)):
                selected = [families[left], families[right]]
                for step in range(1, 10):
                    weights = [step / 10.0, (10 - step) / 10.0]
                    scores = (
                        weights[0] * representations[aggregation][selected[0]] +
                        weights[1] * representations[aggregation][selected[1]]
                    )
                    history.append(evaluate_candidate(evaluate, data, scores, {
                        "aggregation": aggregation,
                        "families": selected,
                        "weights": weights,
                        "eligible": True,
                    }))

        if len(families) >= 3:
            selected = families[:3]
            for a in range(1, 9):
                for b in range(1, 10 - a):
                    c = 10 - a - b
                    if c < 1:
                        continue
                    weights = [a / 10.0, b / 10.0, c / 10.0]
                    scores = sum(
                        weight * representations[aggregation][family]
                        for weight, family in zip(weights, selected)
                    )
                    history.append(evaluate_candidate(evaluate, data, scores, {
                        "aggregation": aggregation,
                        "families": selected,
                        "weights": weights,
                        "eligible": True,
                    }))

    eligible = [row for row in history if row["eligible"]]
    best = max(eligible, key=lambda row: (row["primary"], row["gauc"], row["ndcg5"]))
    final_scores = sum(
        weight * representations[best["aggregation"]][family]
        for weight, family in zip(best["weights"], best["families"])
    )
    return best, final_scores, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data_dir)
    evaluate = get_evaluator(data["fast"])
    groups = make_user_groups(data["train_user"])
    families = ("gauge_fixed_fm", "recency_centered_fm", "regularized_dcn_hybrid")
    if smoke is not None:
        members_per_family = 1
    elif device.type == "cuda":
        members_per_family = 28
    else:
        members_per_family = 18

    history = []
    family_members = {family: [] for family in families}
    progress_path = os.path.join(args.out_dir, "progress.log")
    with open(progress_path, "w") as progress:
        for family_index, family in enumerate(families):
            for member in range(members_per_family):
                seed = args.seed + 1009 * member + 100003 * family_index
                result, scores = train_one(
                    data, evaluate, device, seed, epochs, family, groups
                )
                history.append(result)
                family_members[family].append(scores)
                progress.write(json.dumps({
                    "family": family,
                    "member": member,
                    "seed": seed,
                    "primary": result["primary"],
                    "best_epoch": result["best_epoch"],
                }, sort_keys=True) + "\n")
                progress.flush()

        best_ensemble, final_scores, ensemble_history = select_ensemble(
            family_members, evaluate, data
        )
        progress.write(json.dumps({
            "selection": "cross_mechanism_ensemble",
            "aggregation": best_ensemble["aggregation"],
            "families": best_ensemble["families"],
            "weights": best_ensemble["weights"],
            "primary": best_ensemble["primary"],
        }, sort_keys=True) + "\n")
        progress.flush()

    final_metrics = evaluate(data["val_user"], data["yv"], final_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    family_summary = {}
    for family in families:
        rows = [row for row in history if row["family"] == family]
        values = np.asarray([row["primary"] for row in rows], dtype=np.float64)
        family_summary[family] = {
            "members": int(len(rows)),
            "mean_primary": float(values.mean()),
            "std_primary": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "best_primary": float(values.max()),
        }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc,
            "ndcg5": ndcg5,
            "primary": primary,
            "selected_ensemble": best_ensemble,
            "family_summary": family_summary,
            "history": history,
            "ensemble_history": ensemble_history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
