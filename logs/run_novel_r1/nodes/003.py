import argparse
import csv
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

    def raw_score(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.lin(x).sum((1, 2)) + pair

    def forward(self, x):
        return self.raw_score(x) + self.bias


def _as_number(value):
    try:
        return int(value)
    except Exception:
        return value


def _fit_map(values):
    mapping = {}
    encoded = np.empty(len(values), dtype=np.int64)
    for i, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping) + 1
        encoded[i] = mapping[value]
    return mapping, encoded


def _apply_map(values, mapping):
    return np.asarray([mapping.get(value, 0) for value in values], dtype=np.int64)


def load_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    tr_user, tr_video, tr_tab, tr_duration, tr_y = [], [], [], [], []
    with open(train_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tr_user.append(row["user_id"])
            tr_video.append(row["video_id"])
            tr_tab.append(row["tab"])
            tr_duration.append(float(row["duration_ms"]))
            tr_y.append(float(row["long_view"]))
    va_user, va_video, va_tab, va_duration, va_y = [], [], [], [], []
    with open(val_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            va_user.append(row["user_id"])
            va_video.append(row["video_id"])
            va_tab.append(row["tab"])
            va_duration.append(float(row["duration_ms"]))
            va_y.append(float(row["long_view"]))

    user_map, tr_u = _fit_map(tr_user)
    video_map, tr_v = _fit_map(tr_video)
    tab_map, tr_t = _fit_map(tr_tab)
    va_u = _apply_map(va_user, user_map)
    va_v = _apply_map(va_video, video_map)
    va_t = _apply_map(va_tab, tab_map)
    tr_duration_np = np.asarray(tr_duration, dtype=np.float64)
    va_duration_np = np.asarray(va_duration, dtype=np.float64)
    edges = np.unique(np.quantile(tr_duration_np, np.linspace(0.1, 0.9, 9)))
    tr_d = np.searchsorted(edges, tr_duration_np, side="right").astype(np.int64)
    va_d = np.searchsorted(edges, va_duration_np, side="right").astype(np.int64)
    dur_dim = max(10, len(edges) + 1)
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        dur_dim,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))
    Xt = np.stack((tr_u, tr_v, np.zeros(len(tr_u), dtype=np.int64), tr_t, tr_d), axis=1)
    Xv = np.stack((va_u, va_v, np.zeros(len(va_u), dtype=np.int64), va_t, va_d), axis=1)
    Xt += offsets
    Xv += offsets
    return {
        "Xt": Xt,
        "yt": np.asarray(tr_y, dtype=np.float32),
        "train_user": np.asarray(tr_u, dtype=np.int64),
        "Xv": Xv,
        "yv": np.asarray(va_y, dtype=np.float32),
        "val_user": np.asarray([_as_number(x) for x in va_user]),
        "val_video": np.asarray([_as_number(x) for x in va_video]),
        "field_dims": field_dims,
        "fast": False,
    }


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr = np.load(train_npz)
        va = np.load(val_npz)
        dims = tr["field_dims"].astype(np.int64)
        return {
            "Xt": tr["X"].astype(np.int64),
            "yt": tr["y"].astype(np.float32),
            "train_user": tr["user"],
            "Xv": va["X"].astype(np.int64),
            "yv": va["y"].astype(np.float32),
            "val_user": va["user"],
            "val_video": np.zeros(len(va["y"]), dtype=np.int64),
            "field_dims": dims,
            "fast": True,
        }
    return load_csv_data(data_dir)


def make_user_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = np.asarray(users)[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts)).astype(np.int64)
    ends = np.concatenate((cuts, [len(order)])).astype(np.int64)
    return order, starts, ends


def complete_slate_batches(order, starts, ends, group_permutation, max_rows):
    pending = []
    pending_rows = 0
    for group in group_permutation:
        length = int(ends[group] - starts[group])
        if pending and pending_rows + length > max_rows:
            pieces = [order[starts[g]:ends[g]] for g in pending]
            yield np.concatenate(pieces), np.asarray(
                [ends[g] - starts[g] for g in pending], dtype=np.int64
            )
            pending = []
            pending_rows = 0
        pending.append(int(group))
        pending_rows += length
        if pending_rows >= max_rows:
            pieces = [order[starts[g]:ends[g]] for g in pending]
            yield np.concatenate(pieces), np.asarray(
                [ends[g] - starts[g] for g in pending], dtype=np.int64
            )
            pending = []
            pending_rows = 0
    if pending:
        pieces = [order[starts[g]:ends[g]] for g in pending]
        yield np.concatenate(pieces), np.asarray(
            [ends[g] - starts[g] for g in pending], dtype=np.int64
        )


def centered_logits(raw_logits, lengths, global_bias):
    lengths_t = torch.as_tensor(lengths, dtype=torch.long, device=raw_logits.device)
    end_positions = torch.cumsum(lengths_t, dim=0)
    cumulative = torch.cumsum(raw_logits, dim=0)
    ends = cumulative[end_positions - 1]
    previous = torch.cat((torch.zeros_like(ends[:1]), cumulative[end_positions[:-1] - 1]))
    means = (ends - previous) / lengths_t.to(raw_logits.dtype)
    row_means = torch.repeat_interleave(means, lengths_t)
    return raw_logits - row_means + global_bias


def predict(model, Xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = torch.as_tensor(Xv[start:start + 65536], dtype=torch.long, device=device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def metric_values(evaluate_fn, users, labels, scores):
    result = evaluate_fn(users, labels.astype(int), scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result["primary"]),
    }


def train_one(kind, seed, epochs, data, device, evaluate_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FM(int(data["field_dims"].sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt = data["Xt"]
    yt = data["yt"]
    n = len(yt)
    batch_size = 8192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 101)
    if kind == "gauge_fixed_bce":
        order, starts, ends = make_user_groups(data["train_user"])
    best_primary = -1.0
    best_scores = None
    patience = 0
    curve = []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        row_count = 0
        if kind == "gauge_fixed_bce":
            groups = torch.randperm(len(starts), generator=generator).numpy()
            iterator = complete_slate_batches(order, starts, ends, groups, batch_size)
            for idx, lengths in iterator:
                xb = torch.as_tensor(Xt[idx], dtype=torch.long, device=device)
                yb = torch.as_tensor(yt[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                raw = model.raw_score(xb)
                logits = centered_logits(raw, lengths, model.bias)
                loss = bce(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(idx)
                row_count += len(idx)
        else:
            permutation = torch.randperm(n, generator=generator).numpy()
            for start in range(0, n, batch_size):
                idx = permutation[start:start + batch_size]
                xb = torch.as_tensor(Xt[idx], dtype=torch.long, device=device)
                yb = torch.as_tensor(yt[idx], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = bce(model(xb), yb)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu()) * len(idx)
                row_count += len(idx)
        scores = predict(model, data["Xv"], device)
        metrics = metric_values(evaluate_fn, data["val_user"], data["yv"], scores)
        curve.append({
            "epoch": epoch + 1,
            "train_loss": round(loss_sum / max(row_count, 1), 6),
            "val_gauc": round(metrics["gauc"], 6),
            "val_primary": round(metrics["primary"], 6),
        })
        if metrics["primary"] > best_primary + 1e-6:
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    return best_scores, best_primary, curve


def rank_transform(scores, groups):
    transformed = np.empty(len(scores), dtype=np.float32)
    for idx in groups:
        if len(idx) == 1:
            transformed[idx[0]] = 0.5
        else:
            local_order = np.argsort(scores[idx], kind="stable")
            ranks = np.empty(len(idx), dtype=np.float32)
            ranks[local_order] = np.arange(len(idx), dtype=np.float32) / float(len(idx) - 1)
            transformed[idx] = ranks
    return transformed


def validation_groups(users):
    order = np.argsort(users, kind="stable")
    sorted_users = np.asarray(users)[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    return np.split(order, cuts)


def candidate_counts(n):
    fixed = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96]
    return sorted(set([x for x in fixed if x <= n] + [n]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    data = load_data(args.data_dir)
    if data["fast"]:
        from data.official.evaluate import evaluate as evaluate_fn
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = args.epochs
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    if smoke is not None:
        pair_count = 1
    elif device.type == "cuda":
        pair_count = 48
    else:
        pair_count = 24

    all_scores = {"gauge_fixed_bce": [], "pointwise_bce_control": []}
    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    for probe_index in range(pair_count):
        probe_seed = args.seed + probe_index * 9973
        for kind in ("pointwise_bce_control", "gauge_fixed_bce"):
            scores, primary, curve = train_one(
                kind, probe_seed, epochs, data, device, evaluate_fn
            )
            all_scores[kind].append(scores)
            record = {
                "probe": len(history) + 1,
                "config": {"loss": kind, "seed": probe_seed, "epochs_cap": epochs},
                "primary": round(float(primary), 6),
                "curve": curve,
            }
            history.append(record)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({
                    "loss": kind,
                    "seed": probe_seed,
                    "primary": round(float(primary), 6),
                }, sort_keys=True) + "\n")

    groups = validation_groups(data["val_user"])
    ranked = {
        family: [rank_transform(scores, groups) for scores in members]
        for family, members in all_scores.items()
    }
    ensemble_history = []
    best_metrics = None
    best_scores = None
    best_config = None
    counts = candidate_counts(pair_count)
    for count in counts:
        gauge_mean = np.mean(np.stack(ranked["gauge_fixed_bce"][:count]), axis=0)
        control_mean = np.mean(np.stack(ranked["pointwise_bce_control"][:count]), axis=0)
        candidates = {
            "gauge_fixed_bce": gauge_mean,
            "pointwise_bce_control": control_mean,
            "half_gauge_half_control": 0.5 * gauge_mean + 0.5 * control_mean,
        }
        for family, scores in candidates.items():
            metrics = metric_values(evaluate_fn, data["val_user"], data["yv"], scores)
            entry = {
                "config": {"ensemble_family": family, "member_count": count},
                "gauc": round(metrics["gauc"], 6),
                "ndcg5": round(metrics["ndcg5"], 6),
                "primary": round(metrics["primary"], 6),
            }
            ensemble_history.append(entry)
            if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
                best_metrics = metrics
                best_scores = scores.copy()
                best_config = entry["config"]

    output_metrics = {
        "gauc": best_metrics["gauc"],
        "ndcg5": best_metrics["ndcg5"],
        "primary": best_metrics["primary"],
        "selected_config": best_config,
        "history": history,
        "ensemble_history": ensemble_history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{data['val_user'][i]},{data['val_video'][i]},{float(score):.7g}\n")


if __name__ == "__main__":
    main()
