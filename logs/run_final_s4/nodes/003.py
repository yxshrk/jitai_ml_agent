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
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def raw_score(self, x):
        e = self.emb(x)
        summed = e.sum(dim=1)
        pair = 0.5 * (summed * summed - (e * e).sum(dim=1)).sum(dim=1)
        return self.lin(x).sum(dim=(1, 2)) + pair


def metric_values(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics["nDCG@5"] if "nDCG@5" in metrics else metrics["ndcg5"]),
        "primary": float(metrics["primary"]),
    }


def hour_bucket(values):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.number):
        numeric = arr.astype(np.int64, copy=False)
        return np.where(numeric >= 100, numeric // 100, numeric).astype(np.int64)
    result = np.empty(len(arr), dtype=np.int64)
    for i, value in enumerate(arr):
        text = str(value).strip()
        try:
            if ":" in text:
                result[i] = int(text.split(":", 1)[0])
            else:
                number = int(float(text))
                result[i] = number // 100 if number >= 100 else number
        except ValueError:
            result[i] = 0
    return result


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    train = {
        "X": tr["X"].astype(np.int64, copy=False),
        "y": tr["y"].astype(np.float32, copy=False),
        "user": tr["user"],
        "date": tr["date"],
        "hour": hour_bucket(tr["hourmin"]),
        "tab": tr["X"][:, 3].astype(np.int64, copy=False),
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
    dates = []
    hours = []
    if training:
        maps = {"user": {}, "video": {}, "tab": {}}
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            uid = row["user_id"]
            vid = row["video_id"]
            tab = row["tab"]
            duration = float(row["duration_ms"])
            label = float(row["long_view"])
            raw_users.append(uid)
            raw_videos.append(vid)
            dates.append(row["date"])
            hours.append(row["hourmin"])
            durations.append(duration)
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
        "date": np.asarray(dates),
        "hour": hour_bucket(np.asarray(hours)),
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
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def make_x(raw):
        x = np.column_stack([
            raw["users"],
            raw["videos"],
            np.zeros(len(raw["y"]), dtype=np.int64),
            raw["tabs"],
            raw["buckets"],
        ]).astype(np.int64)
        return x + offsets[None, :]

    train = {
        "X": make_x(tr_raw),
        "y": tr_raw["y"],
        "user": tr_raw["raw_user"],
        "date": tr_raw["date"],
        "hour": tr_raw["hour"],
        "tab": tr_raw["tabs"],
        "field_dims": field_dims,
    }
    valid = {
        "X": make_x(va_raw),
        "y": va_raw["y"],
        "user": va_raw["raw_user"],
        "video": va_raw["raw_video"],
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
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=scores.astype(np.float64))
    return scores - (sums / counts)[inverse]


def predict(model, X, users, device):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), 65536):
            xb = torch.from_numpy(X[start:start + 65536]).to(device)
            pieces.append(model.raw_score(xb).detach().cpu().numpy())
    raw = np.concatenate(pieces).astype(np.float64, copy=False)
    return center_numpy(raw, users)


def append_map(mapping, key, value):
    if key in mapping:
        mapping[key].append(value)
    else:
        mapping[key] = [value]


def make_pair_positions(idx_np, lengths_np, train, rng, context_fraction):
    pos_output = []
    neg_output = []
    offset = 0
    labels = train["y"][idx_np]
    dates = train["date"][idx_np]
    hours = train["hour"][idx_np]
    tabs = train["tab"][idx_np]
    for length_value in lengths_np:
        length = int(length_value)
        end = offset + length
        local_labels = labels[offset:end]
        positive = np.flatnonzero(local_labels > 0.5)
        negative = np.flatnonzero(local_labels <= 0.5)
        if len(positive) and len(negative):
            by_date_hour = {}
            by_date_tab = {}
            by_date = {}
            for n in negative:
                date_value = dates[offset + n].item() if hasattr(dates[offset + n], "item") else dates[offset + n]
                hour_value = int(hours[offset + n])
                tab_value = int(tabs[offset + n])
                append_map(by_date_hour, (date_value, hour_value), int(n))
                append_map(by_date_tab, (date_value, tab_value), int(n))
                append_map(by_date, date_value, int(n))
            for p in positive:
                candidates = None
                if rng.random_sample() < context_fraction:
                    date_value = dates[offset + p].item() if hasattr(dates[offset + p], "item") else dates[offset + p]
                    hour_value = int(hours[offset + p])
                    tab_value = int(tabs[offset + p])
                    first = by_date_hour.get((date_value, hour_value), [])
                    second = by_date_tab.get((date_value, tab_value), [])
                    if first or second:
                        candidates = first + second
                    else:
                        candidates = by_date.get(date_value)
                if not candidates:
                    candidates = negative
                chosen = int(candidates[rng.randint(len(candidates))])
                pos_output.append(offset + int(p))
                neg_output.append(offset + chosen)
        offset = end
    if not pos_output:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.asarray(pos_output, dtype=np.int64), np.asarray(neg_output, dtype=np.int64)


def train_probe(train, valid, evaluate, context_fraction, run_seed, epochs, device):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    model = FM(int(train["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        bce_sum = 0.0
        bpr_sum = 0.0
        seen = 0
        pair_seen = 0
        for idx_np, lengths_np in complete_slate_batches(order, bounds, rng):
            pos_np, neg_np = make_pair_positions(idx_np, lengths_np, train, rng, context_fraction)
            xb = torch.from_numpy(train["X"][idx_np]).to(device)
            yb = torch.from_numpy(train["y"][idx_np]).to(device)
            lengths = torch.from_numpy(lengths_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model.raw_score(xb)
            centered_logits = center_contiguous(raw, lengths)
            bce_loss = bce(centered_logits, yb)
            if len(pos_np):
                pos_t = torch.from_numpy(pos_np).to(device)
                neg_t = torch.from_numpy(neg_np).to(device)
                bpr_loss = torch.nn.functional.softplus(-(raw[pos_t] - raw[neg_t])).mean()
                loss = 0.5 * bce_loss + 0.5 * bpr_loss
                bpr_value = float(bpr_loss.detach().cpu().item())
            else:
                loss = bce_loss
                bpr_value = 0.0
            loss.backward()
            optimizer.step()
            batch_n = len(idx_np)
            pair_n = len(pos_np)
            loss_sum += float(loss.detach().cpu().item()) * batch_n
            bce_sum += float(bce_loss.detach().cpu().item()) * batch_n
            bpr_sum += bpr_value * pair_n
            seen += batch_n
            pair_seen += pair_n
        scores = predict(model, valid["X"], valid["user"], device)
        metrics = metric_values(evaluate(valid["user"], valid["y"].astype(int), scores))
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(seen, 1), 6),
            "train_bce": round(bce_sum / max(seen, 1), 6),
            "train_bpr": round(bpr_sum / max(pair_seen, 1), 6),
            "pairs": int(pair_seen),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
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
        "context_fraction": float(context_fraction),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = min(args.epochs, max(1, int(smoke))) if smoke is not None else args.epochs

    if os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz")):
        train, valid, evaluate = load_npz(args.data_dir)
    else:
        train, valid, evaluate = load_csv(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    context_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    seed_offsets = [0, 17, 43, 89, 137, 193, 251]
    fixed_seeds = [args.seed + offset for offset in seed_offsets]
    probes = []
    score_store = {fraction: [] for fraction in context_fractions}
    progress_path = os.path.join(args.out_dir, "progress.log")

    for fraction in context_fractions:
        for run_seed in fixed_seeds:
            result, scores = train_probe(train, valid, evaluate, fraction, run_seed, epochs, device)
            probes.append(result)
            score_store[fraction].append(scores)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "context_fraction": result["context_fraction"],
                    "seed": result["seed"],
                    "best_epoch": result["best_epoch"],
                    "primary": result["primary"],
                }, sort_keys=True) + "\n")

    summaries = []
    rows_by_fraction = {}
    for fraction in context_fractions:
        rows = [row for row in probes if row["context_fraction"] == fraction]
        rows_by_fraction[fraction] = rows
        mean_primary, se_primary = mean_and_se([row["primary"] for row in rows])
        mean_gauc, se_gauc = mean_and_se([row["gauc"] for row in rows])
        summaries.append({
            "context_fraction": float(fraction),
            "mean_primary": mean_primary,
            "se_primary": se_primary,
            "mean_gauc": mean_gauc,
            "se_gauc": se_gauc,
        })

    uniform_rows = rows_by_fraction[0.0]
    comparisons = []
    for fraction in context_fractions[1:]:
        rows = rows_by_fraction[fraction]
        paired = [rows[i]["primary"] - uniform_rows[i]["primary"] for i in range(len(fixed_seeds))]
        delta_mean, delta_se = mean_and_se(paired)
        comparisons.append({
            "context_fraction": float(fraction),
            "control_fraction": 0.0,
            "paired_primary_delta_mean": delta_mean,
            "paired_primary_delta_se": delta_se,
            "paired_deltas": [float(value) for value in paired],
        })

    winner = max(summaries, key=lambda row: (row["mean_primary"], -abs(row["context_fraction"] - 0.3)))
    selected_fraction = float(winner["context_fraction"])
    final_scores = np.mean(np.stack(score_store[selected_fraction], axis=0), axis=0)
    final_metrics = metric_values(evaluate(valid["user"], valid["y"].astype(int), final_scores))

    metrics_out = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_context_fraction": selected_fraction,
        "aggregation": "mean_of_seven_matched_seed_logits",
        "objective": "0.5_centered_bce_plus_0.5_bpr",
        "negative_fallback_order": ["same_date_hour_or_same_date_tab", "same_date", "uniform_within_user"],
        "fixed_seeds": fixed_seeds,
        "epochs_cap": epochs,
        "summaries": summaries,
        "paired_comparisons_vs_uniform_bpr": comparisons,
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
