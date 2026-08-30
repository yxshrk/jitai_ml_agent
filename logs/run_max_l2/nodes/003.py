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
        for w in self.cross_w:
            torch.nn.init.normal_(w, std=0.01)
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
        for w, b in zip(self.cross_w, self.cross_b):
            scale = (cross * w).sum(dim=1, keepdim=True)
            cross = cross + x0 * scale + b
        deep = self.mlp(x0)
        nonlinear = self.out(torch.cat((cross, deep), dim=1)).squeeze(1)
        return self.bias + self.linear(x).sum(dim=(1, 2)) + nonlinear


def metric_values(evaluator, users, labels, scores):
    result = evaluator(users, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


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
        ordered = sorted(unique.tolist(), key=lambda x: str(x))
        parsed = {value: i for i, value in enumerate(ordered)}
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


def predict(model, features, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = features[start:start + batch_size].to(device, non_blocking=True)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float64, copy=False)


def rank_normalize(scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= float(len(scores) - 1)
    return ranks


def recency_weights(age, floor, half_life=7.0):
    exponential = np.exp2(-age / half_life).astype(np.float32)
    weights = float(floor) + (1.0 - float(floor)) * exponential
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32, copy=False)


def effective_sample_fraction(weights):
    weights = np.asarray(weights, dtype=np.float64)
    ess = weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12)
    return float(ess / len(weights))


def train_one(floor, seed, epochs, checks_per_epoch, Xt, yt, ages, pair_pos, pair_neg,
              Xv, val_users, val_labels, total_dim, evaluator, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = DCNLite(
        total_dim=total_dim,
        fields=Xt.shape[1],
        k=16,
        hidden=128,
        dropout=0.2,
        cross_layers=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    weights_np = recency_weights(ages, floor, half_life=7.0)
    weights = torch.from_numpy(weights_np)
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729)
    best_primary = -1.0
    best_scores = None
    best_metrics = None
    curve = []
    parts = max(1, int(checks_per_epoch))

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        boundaries = np.linspace(0, n, parts + 1, dtype=np.int64)
        running_loss = 0.0
        running_batches = 0
        for part in range(parts):
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
                    chosen = torch.randint(len(pair_pos), (pair_count,), generator=generator)
                    pi = pair_pos[chosen]
                    ni = pair_neg[chosen]
                    xp = Xt[pi].to(device, non_blocking=True)
                    xn = Xt[ni].to(device, non_blocking=True)
                    all_scores = model(torch.cat((xb, xp, xn), dim=0))
                    base_end = len(idx)
                    pos_end = base_end + pair_count
                    logits = all_scores[:base_end]
                    pos_scores = all_scores[base_end:pos_end]
                    neg_scores = all_scores[pos_end:]
                    pointwise = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    point_loss = (pointwise * wb).sum() / wb.sum().clamp_min(1e-6)
                    pair_w = weights[pi].to(device, non_blocking=True)
                    pairwise = F.softplus(-(pos_scores - neg_scores))
                    pair_loss = (pairwise * pair_w).sum() / pair_w.sum().clamp_min(1e-6)
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    logits = model(xb)
                    pointwise = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                    loss = (pointwise * wb).sum() / wb.sum().clamp_min(1e-6)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                running_batches += 1

            scores = predict(model, Xv, device)
            metrics = metric_values(evaluator, val_users, val_labels, scores)
            record = {
                "epoch": epoch + (part + 1) / parts,
                "train_loss": round(running_loss / max(1, running_batches), 6),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_gauc": round(metrics["gauc"], 6),
                "val_primary": round(metrics["primary"], 6),
            }
            curve.append(record)
            if metrics["primary"] > best_primary + 1e-8:
                best_primary = metrics["primary"]
                best_scores = scores.copy()
                best_metrics = metrics
            model.train()

        if (epoch + 1) % 2 == 0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

    del optimizer
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_scores, best_metrics, curve, effective_sample_fraction(weights_np)


def append_progress(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def read_csv_rows(path, train):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))

    def raw_fields(row):
        bucket = int(np.searchsorted(edges, row["duration_ms"], side="right"))
        return [row["user_id"], row["video_id"], row["author_id"], row["tab"], str(bucket)]

    maps = []
    for field in range(5):
        values = sorted({raw_fields(row)[field] for row in train_rows}, key=str)
        maps.append({value: i for i, value in enumerate(values)})
    field_dims = np.asarray([len(mapping) + 1 for mapping in maps], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        result = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            fields = raw_fields(row)
            for field, value in enumerate(fields):
                local = maps[field].get(value, len(maps[field]))
                result[i, field] = int(offsets[field] + local)
        return result

    data = {
        "Xt": encode(train_rows),
        "yt": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "train_users": np.asarray([row["user_id"] for row in train_rows]),
        "train_dates": np.asarray([row["date"] for row in train_rows]),
        "Xv": encode(val_rows),
        "val_labels": np.asarray([row["long_view"] for row in val_rows], dtype=np.int64),
        "val_users": np.asarray([row["user_id"] for row in val_rows]),
        "video_ids": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
    }
    return data


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        from data.official.evaluate import evaluate
        tr = np.load(train_npz)
        va = np.load(val_npz)
        field_dims = tr["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        return {
            "Xt": tr["X"].astype(np.int64, copy=False),
            "yt": tr["y"].astype(np.float32, copy=False),
            "train_users": tr["user"],
            "train_dates": tr["date"],
            "Xv": va["X"].astype(np.int64, copy=False),
            "val_labels": va["y"].astype(np.int64, copy=False),
            "val_users": va["user"],
            "video_ids": va["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
        }, evaluate
    from harness.evaluate_provisional import evaluate
    return load_csv_data(data_dir), evaluate


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

    data, evaluator = load_data(args.data_dir)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    train_days = date_ordinals(data["train_dates"])
    ages = train_days.max() - train_days
    pair_pos, pair_neg = make_pairs(data["train_users"], data["yt"], args.seed + 31)
    total_dim = int(data["field_dims"].sum())

    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_raw) if smoke_raw is not None else None
    probe_epochs = 8 if smoke_cap is None else min(8, smoke_cap)
    final_epochs = args.epochs if smoke_cap is None else min(args.epochs, smoke_cap)
    floors = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
    probe_seed_count = 5
    final_seed_count = 5
    if smoke_cap is not None:
        floors = [0.0, 1.0]
        probe_seed_count = 1
        final_seed_count = 1

    history = []
    floor_summaries = []
    for floor in floors:
        seed_metrics = []
        for member in range(probe_seed_count):
            seed = args.seed + 1000 + member
            scores, metrics, curve, ess_fraction = train_one(
                floor, seed, probe_epochs, 1, Xt, yt, ages, pair_pos, pair_neg,
                Xv, data["val_users"], data["val_labels"], total_dim, evaluator, device,
            )
            entry = {
                "stage": "matched_recency_probe",
                "floor": floor,
                "half_life_days": 7.0,
                "seed": seed,
                "ess_fraction": ess_fraction,
                "metrics": metrics,
                "best_checkpoint": max(curve, key=lambda x: x["val_primary"]),
            }
            history.append(entry)
            seed_metrics.append(metrics)
            append_progress(progress_path, entry)
            del scores
        summary = {
            "floor": floor,
            "half_life_days": 7.0,
            "mean_primary": float(np.mean([m["primary"] for m in seed_metrics])),
            "mean_gauc": float(np.mean([m["gauc"] for m in seed_metrics])),
            "std_primary": float(np.std([m["primary"] for m in seed_metrics])),
        }
        floor_summaries.append(summary)

    winning_summary = max(floor_summaries, key=lambda item: item["mean_primary"])
    winning_floor = float(winning_summary["floor"])

    final_members = []
    final_records = []
    for member in range(final_seed_count):
        seed = args.seed + member
        scores, metrics, curve, ess_fraction = train_one(
            winning_floor, seed, final_epochs, 2, Xt, yt, ages, pair_pos, pair_neg,
            Xv, data["val_users"], data["val_labels"], total_dim, evaluator, device,
        )
        final_members.append(rank_normalize(scores))
        record = {
            "stage": "final",
            "member": member,
            "seed": seed,
            "floor": winning_floor,
            "half_life_days": 7.0,
            "ess_fraction": ess_fraction,
            "metrics": metrics,
            "curve": curve,
        }
        final_records.append(record)
        append_progress(progress_path, {
            "stage": "final",
            "member": member,
            "seed": seed,
            "floor": winning_floor,
            "primary": metrics["primary"],
        })

    prefix_history = []
    running = np.zeros(len(Xv), dtype=np.float64)
    selected_scores = None
    selected_metrics = None
    selected_count = 0
    for index, member_scores in enumerate(final_members):
        running += member_scores
        ensemble_scores = running / float(index + 1)
        metrics = metric_values(
            evaluator, data["val_users"], data["val_labels"], ensemble_scores
        )
        prefix_history.append({"members": index + 1, "metrics": metrics})
        if selected_metrics is None or metrics["primary"] > selected_metrics["primary"]:
            selected_metrics = metrics
            selected_scores = ensemble_scores.copy()
            selected_count = index + 1

    output = {
        "gauc": selected_metrics["gauc"],
        "ndcg5": selected_metrics["ndcg5"],
        "primary": selected_metrics["primary"],
        "history": history,
        "floor_summaries": floor_summaries,
        "winning_recency": {
            "half_life_days": 7.0,
            "floor": winning_floor,
            "normalized_to_mean_one": True,
        },
        "final_seeds": final_records,
        "ensemble_prefixes": prefix_history,
        "selected_ensemble_members": selected_count,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", encoding="utf-8") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(selected_scores):
            handle.write(
                f"{row_id},{data['val_users'][row_id]},{data['video_ids'][row_id]},{score:.8g}\n"
            )


if __name__ == "__main__":
    main()
