"""FM baseline with gauge-fixed, user-centered BCE.

Training batches contain complete user slates. The pointwise logits are centered by
 each user's slate mean and receive one learned global bias, removing per-user
constant degrees of freedom that do not affect within-user ranking metrics.
"""
import argparse
import csv
import json
import os
import sys
import time

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

    def raw_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def forward(self, x):
        return self.raw_score(x) + self.bias

    def centered_logits(self, x, group_counts):
        raw = self.raw_score(x)
        counts = torch.as_tensor(group_counts, dtype=torch.long, device=raw.device)
        ends = torch.cumsum(counts, dim=0)
        starts = ends - counts
        prefix = torch.cat((raw.new_zeros(1), torch.cumsum(raw, dim=0)))
        group_sums = prefix[ends] - prefix[starts]
        group_means = group_sums / counts.to(raw.dtype)
        row_means = torch.repeat_interleave(group_means, counts)
        return raw - row_means + self.bias


def _parse_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        has_author = "author_id" in (reader.fieldnames or [])
        for row in reader:
            record = {
                "user": row.get("user_id", ""),
                "video": row.get("video_id", ""),
                "author": row.get("author_id", "__missing_author__") if has_author else "__missing_author__",
                "tab": row.get("tab", ""),
                "duration": _parse_float(row.get("duration_ms", 0.0)),
                "y": _parse_float(row.get("long_view", 0.0)),
            }
            rows.append(record)
    return rows


def _fit_category(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def _encode_csv(train_rows, val_rows):
    feature_names = ("user", "video", "author", "tab")
    maps = {name: _fit_category([r[name] for r in train_rows]) for name in feature_names}
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    if len(durations):
        edges = np.quantile(durations, np.linspace(0.1, 0.9, 9))
    else:
        edges = np.zeros(9, dtype=np.float64)

    field_dims = [len(maps[name]) + 1 for name in feature_names] + [10]
    offsets = np.cumsum([0] + field_dims[:-1], dtype=np.int64)

    def transform(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, name in enumerate(feature_names):
                x[i, j] = maps[name].get(row[name], len(maps[name])) + offsets[j]
            x[i, 4] = int(np.searchsorted(edges, row["duration"], side="right")) + offsets[4]
        return x

    return (
        transform(train_rows),
        np.asarray([r["y"] for r in train_rows], dtype=np.float32),
        np.asarray([r["user"] for r in train_rows]),
        transform(val_rows),
        np.asarray([r["y"] for r in val_rows], dtype=np.float32),
        np.asarray([r["user"] for r in val_rows]),
        np.asarray(field_dims, dtype=np.int64),
        [r["video"] for r in val_rows],
    )


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        xv = va["X"].astype(np.int64, copy=False)
        video_ids = ["0"] * len(xv)
        return {
            "Xt": tr["X"].astype(np.int64, copy=False),
            "yt": tr["y"].astype(np.float32, copy=False),
            "ut": np.asarray(tr["user"]),
            "Xv": xv,
            "yv": va["y"].astype(np.float32, copy=False),
            "uv": np.asarray(va["user"]),
            "field_dims": np.asarray(tr["field_dims"], dtype=np.int64),
            "video_ids": video_ids,
            "fast_path": True,
        }

    train_rows = _read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = _read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    Xt, yt, ut, Xv, yv, uv, field_dims, video_ids = _encode_csv(train_rows, val_rows)
    return {
        "Xt": Xt,
        "yt": yt,
        "ut": ut,
        "Xv": Xv,
        "yv": yv,
        "uv": uv,
        "field_dims": field_dims,
        "video_ids": video_ids,
        "fast_path": False,
    }


def make_user_groups(users):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    boundaries = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(order)])))
    return [order[start:end] for start, end in zip(starts, ends)]


def complete_slate_batches(groups, permutation, target_size):
    pending = []
    pending_size = 0
    for group_number in permutation:
        group = groups[int(group_number)]
        if pending and pending_size + len(group) > target_size:
            yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)
            pending = []
            pending_size = 0
    if pending:
        yield np.concatenate(pending), np.asarray([len(g) for g in pending], dtype=np.int64)


def predict(model, x, device, batch_size=65536):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def metric_values(result):
    return (
        float(result.get("GAUC", result.get("gauc"))),
        float(result.get("nDCG@5", result.get("ndcg5"))),
        float(result["primary"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs
    smoke_epochs = os.environ.get("SMOKE_EPOCHS")
    if smoke_epochs is not None:
        epochs = min(epochs, max(1, int(smoke_epochs)))

    started = time.monotonic()
    data = load_data(args.data_dir)
    if data["fast_path"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt = data["Xt"]
    yt = data["yt"]
    groups = make_user_groups(data["ut"])
    total_dim = int(data["field_dims"].sum())

    model = FM(total_dim, k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(args.seed)

    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    patience = 0
    history = []
    batch_size = 8192

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(groups))
        total_loss = 0.0
        seen = 0
        for indices, counts in complete_slate_batches(groups, permutation, batch_size):
            xb = torch.from_numpy(Xt[indices]).to(device)
            yb = torch.from_numpy(yt[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.centered_logits(xb, counts)
            loss = bce(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)

        scores = predict(model, data["Xv"], device)
        result = evaluate(data["uv"], data["yv"].astype(int), scores)
        gauc, ndcg5, primary = metric_values(result)
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / max(seen, 1),
            "val_gauc": gauc,
            "val_ndcg5": ndcg5,
            "val_primary": primary,
        })
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        best_scores = predict(model, data["Xv"], device)

    final_result = evaluate(data["uv"], data["yv"].astype(int), best_scores)
    gauc, ndcg5, primary = metric_values(final_result)
    runtime_seconds = time.monotonic() - started

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = {
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "best_epoch": best_epoch,
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
        "acceptance_threshold": 0.002,
        "method": "gauge-fixed-bce",
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uv"][i], data["video_ids"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
