import argparse
import csv
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, use_bias=True):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1)) if use_bias else None
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def raw_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    train = {
        "X": tr["X"].astype(np.int64, copy=False),
        "y": tr["y"].astype(np.float32, copy=False),
        "user": tr["user"],
        "field_dims": tr["field_dims"].astype(np.int64, copy=False),
    }
    valid = {
        "X": va["X"].astype(np.int64, copy=False),
        "y": va["y"].astype(np.float32, copy=False),
        "user": va["user"],
        "video": va["X"][:, 1].astype(np.int64, copy=False),
    }
    from data.official.evaluate import evaluate
    return train, valid, evaluate


def read_csv_rows(path, training, maps=None, duration_edges=None):
    users = []
    videos = []
    tabs = []
    durations = []
    labels = []
    raw_users = []
    raw_videos = []
    if training:
        maps = {"user": {}, "video": {}, "tab": {}}
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row["user_id"]
            vid = row["video_id"]
            tab = row["tab"]
            dur = float(row["duration_ms"])
            label = float(row["long_view"])
            raw_users.append(uid)
            raw_videos.append(vid)
            durations.append(dur)
            labels.append(label)
            for name, value, target in (("user", uid, users), ("video", vid, videos), ("tab", tab, tabs)):
                mapping = maps[name]
                if training and value not in mapping:
                    mapping[value] = len(mapping)
                target.append(mapping.get(value, len(mapping)))
    durations = np.asarray(durations, dtype=np.float64)
    if training:
        duration_edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    buckets = np.searchsorted(duration_edges, durations, side="right").astype(np.int64)
    return {
        "users": np.asarray(users, dtype=np.int64),
        "videos": np.asarray(videos, dtype=np.int64),
        "tabs": np.asarray(tabs, dtype=np.int64),
        "buckets": buckets,
        "y": np.asarray(labels, dtype=np.float32),
        "raw_user": np.asarray(raw_users),
        "raw_video": np.asarray(raw_videos),
    }, maps, duration_edges


def load_csv(data_dir):
    tr_raw, maps, edges = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    va_raw, _, _ = read_csv_rows(os.path.join(data_dir, "val.csv"), False, maps, edges)
    field_dims = np.asarray([
        len(maps["user"]) + 1,
        len(maps["video"]) + 1,
        1,
        len(maps["tab"]) + 1,
        len(edges) + 1,
    ], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]])

    def make_x(raw):
        x = np.column_stack([
            raw["users"], raw["videos"], np.zeros(len(raw["y"]), dtype=np.int64),
            raw["tabs"], raw["buckets"],
        ]).astype(np.int64)
        return x + offsets[None, :]

    train = {
        "X": make_x(tr_raw), "y": tr_raw["y"],
        "user": tr_raw["raw_user"], "field_dims": field_dims,
    }
    valid = {
        "X": make_x(va_raw), "y": va_raw["y"],
        "user": va_raw["raw_user"], "video": va_raw["raw_video"],
    }
    from harness.evaluate_provisional import evaluate
    return train, valid, evaluate


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = np.asarray(users)[order]
    if len(order) == 0:
        return order, np.asarray([0], dtype=np.int64)
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(order)])).astype(np.int64)
    return order, bounds


def complete_slate_batches(order, bounds, rng, target_size=8192):
    group_ids = rng.permutation(len(bounds) - 1)
    parts = []
    lengths = []
    count = 0
    for gid in group_ids:
        start = int(bounds[gid])
        end = int(bounds[gid + 1])
        length = end - start
        if parts and count + length > target_size:
            yield np.concatenate(parts), np.asarray(lengths, dtype=np.int64)
            parts = []
            lengths = []
            count = 0
        parts.append(order[start:end])
        lengths.append(length)
        count += length
    if parts:
        yield np.concatenate(parts), np.asarray(lengths, dtype=np.int64)


def center_contiguous(raw, lengths):
    ends = torch.cumsum(lengths, dim=0)
    starts = torch.cat((torch.zeros(1, dtype=torch.long, device=raw.device), ends[:-1]))
    prefix = torch.cat((torch.zeros(1, dtype=raw.dtype, device=raw.device), torch.cumsum(raw, dim=0)))
    sums = prefix[ends] - prefix[starts]
    means = sums / lengths.to(raw.dtype)
    return raw - torch.repeat_interleave(means, lengths)


def center_numpy(scores, users):
    _, inverse = np.unique(users, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=scores.astype(np.float64))
    return scores - (sums / count)[inverse]


def predict(model, X, users, centered, use_bias, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.from_numpy(X[start:start + 65536]).to(device)
            pieces.append(model.raw_score(xb).detach().cpu().numpy())
    raw = np.concatenate(pieces).astype(np.float64, copy=False)
    if centered:
        raw = center_numpy(raw, users)
    if use_bias:
        raw = raw + float(model.bias.detach().cpu().item())
    return raw


def train_probe(train, valid, evaluate, variant, run_seed, epochs, device):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    centered = bool(variant["centered"])
    use_bias = bool(variant["bias"])
    model = FM(int(train["field_dims"].sum()), k=16, use_bias=use_bias).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    order, bounds = make_user_groups(train["user"])
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    patience = 0
    curve = []
    for epoch in range(epochs):
        model.train()
        rng = np.random.RandomState(run_seed * 1009 + epoch)
        loss_sum = 0.0
        seen = 0
        for idx_np, lengths_np in complete_slate_batches(order, bounds, rng):
            xb = torch.from_numpy(train["X"][idx_np]).to(device)
            yb = torch.from_numpy(train["y"][idx_np]).to(device)
            opt.zero_grad(set_to_none=True)
            raw = model.raw_score(xb)
            if centered:
                lengths = torch.from_numpy(lengths_np).to(device)
                logits = center_contiguous(raw, lengths)
            else:
                logits = raw
            if use_bias:
                logits = logits + model.bias
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            batch_n = len(idx_np)
            loss_sum += float(loss.detach().cpu().item()) * batch_n
            seen += batch_n
        scores = predict(model, valid["X"], valid["user"], centered, use_bias, device)
        metrics = metric_values(evaluate(valid["user"], valid["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
            "global_bias": round(float(model.bias.detach().cpu().item()), 6) if use_bias else None,
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_epoch = epoch + 1
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    final_metrics = metric_values(evaluate(valid["user"], valid["y"].astype(int), best_scores))
    return {
        "variant": variant["name"],
        "centered": centered,
        "global_bias": use_bias,
        "seed": int(run_seed),
        "best_epoch": int(best_epoch),
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "curve": curve,
    }, best_scores


def mean_and_se(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = min(args.epochs, max(1, int(smoke))) if smoke is not None else args.epochs

    npz_path = os.path.join(args.data_dir, "train.npz")
    if os.path.exists(npz_path) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        train, valid, evaluate = load_npz(args.data_dir)
    else:
        train, valid, evaluate = load_csv(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    variants = [
        {"name": "ordinary_no_bias", "centered": False, "bias": False},
        {"name": "ordinary_global_bias", "centered": False, "bias": True},
        {"name": "centered_no_bias", "centered": True, "bias": False},
        {"name": "centered_global_bias", "centered": True, "bias": True},
    ]
    seed_offsets = [0, 17, 43, 89, 137]
    fixed_seeds = [args.seed + x for x in seed_offsets]
    probes = []
    score_store = {v["name"]: [] for v in variants}
    progress_path = os.path.join(args.out_dir, "progress.log")

    for variant in variants:
        for run_seed in fixed_seeds:
            result, scores = train_probe(train, valid, evaluate, variant, run_seed, epochs, device)
            probes.append(result)
            score_store[variant["name"]].append(scores)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "variant": result["variant"], "seed": result["seed"],
                    "best_epoch": result["best_epoch"], "primary": result["primary"],
                }, sort_keys=True) + "\n")

    summaries = []
    for variant in variants:
        name = variant["name"]
        rows = [p for p in probes if p["variant"] == name]
        mean_primary, se_primary = mean_and_se([p["primary"] for p in rows])
        mean_gauc, se_gauc = mean_and_se([p["gauc"] for p in rows])
        summaries.append({
            "variant": name,
            "centered": variant["centered"],
            "global_bias": variant["bias"],
            "mean_primary": mean_primary,
            "se_primary": se_primary,
            "mean_gauc": mean_gauc,
            "se_gauc": se_gauc,
        })

    by_name = {s["variant"]: s for s in summaries}
    baseline_rows = [p for p in probes if p["variant"] == "ordinary_global_bias"]
    comparisons = []
    for name in ("ordinary_no_bias", "centered_no_bias", "centered_global_bias"):
        rows = [p for p in probes if p["variant"] == name]
        deltas = [rows[i]["primary"] - baseline_rows[i]["primary"] for i in range(len(fixed_seeds))]
        delta_mean, delta_se = mean_and_se(deltas)
        comparisons.append({
            "variant": name,
            "control": "ordinary_global_bias",
            "paired_primary_delta_mean": delta_mean,
            "paired_primary_delta_se": delta_se,
            "paired_deltas": [float(x) for x in deltas],
        })

    centered_bias_rows = [p for p in probes if p["variant"] == "centered_global_bias"]
    centered_no_bias_rows = [p for p in probes if p["variant"] == "centered_no_bias"]
    bias_deltas = [centered_bias_rows[i]["primary"] - centered_no_bias_rows[i]["primary"] for i in range(len(fixed_seeds))]
    bias_delta_mean, bias_delta_se = mean_and_se(bias_deltas)
    comparisons.append({
        "variant": "centered_global_bias",
        "control": "centered_no_bias",
        "paired_primary_delta_mean": bias_delta_mean,
        "paired_primary_delta_se": bias_delta_se,
        "paired_deltas": [float(x) for x in bias_deltas],
    })

    winner = max(summaries, key=lambda x: (x["mean_primary"], not x["global_bias"]))
    selected_name = winner["variant"]
    final_scores = np.mean(np.stack(score_store[selected_name], axis=0), axis=0)
    final_metrics = metric_values(evaluate(valid["user"], valid["y"].astype(int), final_scores))

    metrics_out = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_variant": selected_name,
        "aggregation": "mean_of_five_matched_seed_logits",
        "fixed_seeds": fixed_seeds,
        "epochs_cap": epochs,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "history": probes,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{valid['user'][i]},{valid['video'][i]},{score:.9g}\n")


if __name__ == "__main__":
    main()
