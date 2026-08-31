"""FM baseline with complete-user-slate gauge-fixed BCE.

The pointwise logit is centered within each complete training-user slate before a
single learned global bias is added. This removes metric-irrelevant per-user
constant shifts while retaining the parent's FM features and optimizer.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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


def segment_center(logits, lengths):
    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=logits.device)
    ends = torch.cumsum(lengths_t, dim=0)
    starts = ends - lengths_t
    prefix = torch.cat((logits.new_zeros(1), torch.cumsum(logits, dim=0)))
    means = (prefix[ends] - prefix[starts]) / lengths_t.to(logits.dtype)
    return logits - torch.repeat_interleave(means, lengths_t)


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    if len(order) == 0:
        return []
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)]))
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def complete_slate_batches(groups, rng, target_size):
    group_order = rng.permutation(len(groups))
    pending = []
    pending_size = 0
    for gi in group_order:
        group = groups[int(gi)]
        if pending and pending_size + len(group) > target_size:
            yield np.concatenate(pending), [len(g) for g in pending]
            pending = []
            pending_size = 0
        pending.append(group)
        pending_size += len(group)
        if pending_size >= target_size:
            yield np.concatenate(pending), [len(g) for g in pending]
            pending = []
            pending_size = 0
    if pending:
        yield np.concatenate(pending), [len(g) for g in pending]


def read_csv_rows(path, need_label):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def build_mapping(values):
    mapping = {}
    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)
    return mapping


def encode_csv(train_rows, val_rows):
    user_map = build_mapping([r["user_id"] for r in train_rows])
    video_map = build_mapping([r["video_id"] for r in train_rows])
    tab_map = build_mapping([r["tab"] for r in train_rows])

    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.linspace(0.1, 0.9, 9)
    cuts = np.quantile(durations, quantiles) if len(durations) else np.zeros(9)

    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))

    def transform(rows, with_label):
        x = np.empty((len(rows), 5), dtype=np.int64)
        users = np.empty(len(rows), dtype=np.int64)
        labels = np.empty(len(rows), dtype=np.float32) if with_label else None
        raw_users = []
        raw_videos = []
        for i, row in enumerate(rows):
            uid = user_map.get(row["user_id"], len(user_map))
            vid = video_map.get(row["video_id"], len(video_map))
            tab = tab_map.get(row["tab"], len(tab_map))
            dur = int(np.searchsorted(cuts, row["duration_ms"], side="right"))
            x[i] = np.asarray([uid, vid, 0, tab, dur], dtype=np.int64) + offsets
            users[i] = uid
            raw_users.append(row["user_id"])
            raw_videos.append(row["video_id"])
            if with_label:
                labels[i] = row["long_view"]
        return x, labels, users, raw_users, raw_videos

    xt, yt, ut, _, _ = transform(train_rows, True)
    xv, yv, uv, raw_users, raw_videos = transform(val_rows, True)
    return {
        "Xt": xt,
        "yt": yt,
        "ut": ut,
        "Xv": xv,
        "yv": yv,
        "uv": uv,
        "raw_users": raw_users,
        "raw_videos": raw_videos,
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        with np.load(train_npz) as tr, np.load(val_npz) as va:
            field_dims = tr["field_dims"].astype(np.int64)
            offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))
            xv = va["X"].astype(np.int64)
            raw_videos = (xv[:, 1] - offsets[1]).astype(np.int64)
            return {
                "Xt": tr["X"].astype(np.int64),
                "yt": tr["y"].astype(np.float32),
                "ut": tr["user"].copy(),
                "Xv": xv,
                "yv": va["y"].astype(np.float32),
                "uv": va["user"].copy(),
                "raw_users": va["user"].copy(),
                "raw_videos": raw_videos,
                "field_dims": field_dims,
                "fast": True,
            }

    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), True)
    return encode_csv(train_rows, val_rows)


def score_model(model, x, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            xb = x[start:start + 65536].to(device, non_blocking=True)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    epochs = args.epochs
    if "SMOKE_EPOCHS" in os.environ:
        epochs = min(epochs, max(1, int(os.environ["SMOKE_EPOCHS"])))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    xv = torch.from_numpy(data["Xv"])
    groups = make_user_groups(np.asarray(data["ut"]))

    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(args.seed)

    best = -1.0
    best_scores = None
    patience = 0
    history = []
    batch_size = 8192

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        seen = 0
        for indices, lengths in complete_slate_batches(groups, rng, batch_size):
            idx = torch.from_numpy(indices.astype(np.int64, copy=False))
            xb = xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            relative = model.relative_logit(xb)
            centered = segment_center(relative, lengths) + model.bias
            loss = bce(centered, yb)
            loss.backward()
            opt.step()
            count = len(indices)
            loss_sum += float(loss.detach().cpu()) * count
            seen += count

        scores = score_model(model, xv, device)
        metrics = evaluate(data["uv"], data["yv"].astype(int), scores)
        primary = float(metrics["primary"])
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 5),
            "val_gauc": round(float(metrics.get("GAUC", metrics.get("gauc", 0.0))), 6),
            "val_primary": round(primary, 6),
        })
        if primary > best + 1e-6:
            best = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    if best_scores is None:
        best_scores = score_model(model, xv, device)

    final_metrics = evaluate(data["uv"], data["yv"].astype(int), best_scores)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": float(final_metrics.get("GAUC", final_metrics.get("gauc"))),
            "ndcg5": float(final_metrics.get("nDCG@5", final_metrics.get("ndcg5"))),
            "primary": float(final_metrics["primary"]),
            "history": history,
        }, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["raw_users"][i], data["raw_videos"][i], format(float(score), ".6g")])


if __name__ == "__main__":
    main()
