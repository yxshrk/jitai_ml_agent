"""Validation-selected ensemble design sweep for the accepted DCN-lite champion."""
import argparse
import csv
import datetime
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        width = fields * k
        self.emb = torch.nn.Embedding(total_dim, k)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.emb_drop = torch.nn.Dropout(dropout)
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.cross_out = torch.nn.Linear(width, 1, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        torch.nn.init.xavier_uniform_(self.deep[0].weight)
        torch.nn.init.xavier_uniform_(self.deep[3].weight)
        torch.nn.init.xavier_uniform_(self.cross_out.weight)
        torch.nn.init.zeros_(self.deep[0].bias)
        torch.nn.init.zeros_(self.deep[3].bias)

    def forward(self, x):
        e = self.emb_drop(self.emb(x))
        x0 = e.reshape(e.shape[0], -1)
        cross = x0 * (x0 * self.cross_w).sum(1, keepdim=True) + self.cross_b + x0
        return (self.bias + self.linear(x).sum((1, 2)) +
                self.cross_out(cross).squeeze(1) + self.deep(x0).squeeze(1))


def encode_csv(data_dir):
    wanted = ["user_id", "video_id", "tab", "duration_ms", "date", "long_view"]

    def read(path):
        out = {key: [] for key in wanted}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                for key in wanted:
                    out[key].append(row[key])
        return out

    tr_raw = read(os.path.join(data_dir, "train.csv"))
    va_raw = read(os.path.join(data_dir, "val.csv"))
    users = {v: i for i, v in enumerate(sorted(set(tr_raw["user_id"])))}
    videos = {v: i for i, v in enumerate(sorted(set(tr_raw["video_id"])))}
    tabs = {v: i for i, v in enumerate(sorted(set(tr_raw["tab"])))}
    train_duration = np.asarray(tr_raw["duration_ms"], dtype=np.float64)
    cuts = np.unique(np.quantile(train_duration, np.linspace(0.1, 0.9, 9)))
    dims = np.asarray([len(users) + 1, len(videos) + 1, len(videos) + 1,
                       len(tabs) + 1, len(cuts) + 1], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1])))

    def make(raw):
        n = len(raw["user_id"])
        duration = np.asarray(raw["duration_ms"], dtype=np.float64)
        local = np.empty((n, 5), dtype=np.int64)
        local[:, 0] = [users.get(v, len(users)) for v in raw["user_id"]]
        local[:, 1] = [videos.get(v, len(videos)) for v in raw["video_id"]]
        local[:, 2] = local[:, 1]
        local[:, 3] = [tabs.get(v, len(tabs)) for v in raw["tab"]]
        local[:, 4] = np.searchsorted(cuts, duration, side="right")
        return {
            "X": local + offsets,
            "y": np.asarray(raw["long_view"], dtype=np.float32),
            "user": np.asarray(raw["user_id"]),
            "video": np.asarray(raw["video_id"]),
            "date": np.asarray(raw["date"]),
            "field_dims": dims,
        }

    return make(tr_raw), make(va_raw), False


def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.npz")
    val_path = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return encode_csv(data_dir)
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)
    tr = {key: train_npz[key] for key in train_npz.files}
    va = {key: val_npz[key] for key in val_npz.files}
    offsets = np.concatenate(([0], np.cumsum(np.asarray(tr["field_dims"])[:-1])))
    va["video"] = np.asarray(va["X"][:, 1], dtype=np.int64) - int(offsets[1])
    return tr, va, True


def get_evaluator(use_npz):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(result):
    return (float(result.get("GAUC", result.get("gauc"))),
            float(result.get("nDCG@5", result.get("ndcg5"))),
            float(result["primary"]))


def date_ordinals(values):
    arr = np.asarray(values).astype(str)
    unique, inverse = np.unique(arr, return_inverse=True)
    converted = []
    for value in unique:
        text = value.split(".")[0].replace("-", "")
        try:
            converted.append(datetime.datetime.strptime(text[:8], "%Y%m%d").date().toordinal())
        except ValueError:
            try:
                converted.append(int(float(text)))
            except ValueError:
                converted.append(0)
    return np.asarray(converted, dtype=np.float32)[inverse]


def recency_weights(dates, half_life):
    day = date_ordinals(dates)
    weights = np.exp2(-(float(day.max()) - day) / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-8)


def make_pair_pool(users, labels):
    users = np.asarray(users)
    positive_mask = np.asarray(labels) > 0.5
    negatives = np.flatnonzero(~positive_mask)
    if len(negatives) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    order = np.argsort(users[negatives], kind="stable")
    negative_sorted = negatives[order]
    negative_users = users[negative_sorted]
    unique, starts, counts = np.unique(negative_users, return_index=True, return_counts=True)
    positives = np.flatnonzero(positive_mask)
    locations = np.searchsorted(unique, users[positives])
    valid = locations < len(unique)
    exact = np.zeros(len(positives), dtype=bool)
    exact[valid] = unique[locations[valid]] == users[positives[valid]]
    positives = positives[exact]
    locations = locations[exact]
    return (positives.astype(np.int64), starts[locations].astype(np.int64),
            counts[locations].astype(np.int64), negative_sorted.astype(np.int64))


def predict_logits(model, Xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = Xv[start:start + 65536].to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks)


def train_variant(config, seed, epochs, X, y, dates, pair_data, Xv, vu, vy,
                  evaluate, device, fraction=1.0, half_checkpoints=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = DCNLite(int(config["total_dim"]), dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]))
    weights = recency_weights(dates, float(config["half_life"]))
    pos_pool, neg_starts, neg_counts, neg_sorted = pair_data
    n = len(y)
    subset_n = max(1, int(n * fraction))
    batch_size = int(config.get("batch_size", 16384))
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    curve = []

    def assess(checkpoint, last_loss):
        nonlocal best_primary, best_metrics, best_scores
        scores = predict_logits(model, Xv, device)
        metrics = metric_values(evaluate(vu, vy, scores))
        curve.append({
            "checkpoint": checkpoint,
            "train_loss": round(float(last_loss), 6),
            "val_gauc": round(metrics[0], 6),
            "val_ndcg5": round(metrics[1], 6),
            "val_primary": round(metrics[2], 6),
        })
        if metrics[2] > best_primary + 1e-8:
            best_primary = metrics[2]
            best_metrics = metrics
            best_scores = scores.copy()

    for epoch in range(epochs):
        model.train()
        if fraction < 0.999:
            permutation = rng.choice(n, size=subset_n, replace=False)
            rng.shuffle(permutation)
        else:
            permutation = rng.permutation(n)
        split = (len(permutation) + 1) // 2
        sections = [permutation[:split], permutation[split:]] if half_checkpoints else [permutation]
        last_loss = 0.0
        for section_id, section in enumerate(sections):
            for start in range(0, len(section), batch_size):
                idx_np = section[start:start + batch_size]
                idx = torch.from_numpy(idx_np.astype(np.int64, copy=False))
                xb = X[idx].to(device)
                target = y[idx].to(device)
                sample_weight = torch.from_numpy(weights[idx_np]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none")
                point_loss = ((point_raw * sample_weight).sum() /
                              sample_weight.sum().clamp_min(1e-8))
                pair_count = min(max(256, len(idx_np) // 8), len(pos_pool))
                if pair_count > 0:
                    choices = rng.integers(0, len(pos_pool), size=pair_count)
                    positive_np = pos_pool[choices]
                    offsets = (rng.random(pair_count) * neg_counts[choices]).astype(np.int64)
                    negative_np = neg_sorted[neg_starts[choices] + offsets]
                    positive_idx = torch.from_numpy(positive_np).to(device)
                    negative_idx = torch.from_numpy(negative_np).to(device)
                    positive_score = model(X[positive_idx].to(device))
                    negative_score = model(X[negative_idx].to(device))
                    pair_weight = torch.from_numpy(
                        (weights[positive_np] + weights[negative_np]) * 0.5).to(device)
                    pair_raw = torch.nn.functional.softplus(-(positive_score - negative_score))
                    pair_loss = ((pair_raw * pair_weight).sum() /
                                 pair_weight.sum().clamp_min(1e-8))
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            if half_checkpoints:
                assess("%d.%d" % (epoch + 1, 5 if section_id == 0 else 0), last_loss)
                model.train()
        if not half_checkpoints:
            assess(str(epoch + 1), last_loss)
        if (epoch + 1) % int(config["step_size"]) == 0:
            for group in optimizer.param_groups:
                group["lr"] *= float(config["gamma"])
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def clean_config(config):
    result = {}
    for key, value in config.items():
        if key in ("total_dim", "batch_size"):
            continue
        if isinstance(value, (float, np.floating)):
            result[key] = round(float(value), 8)
        else:
            result[key] = int(value)
    return result


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def per_user_ranks(scores, users):
    scores = np.asarray(scores)
    users = np.asarray(users)
    output = np.empty(len(scores), dtype=np.float32)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        local_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[local_order] = np.arange(len(indices), dtype=np.float32)
        if len(indices) > 1:
            ranks /= float(len(indices) - 1)
        else:
            ranks.fill(0.5)
        output[indices] = ranks
    return output


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
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, use_npz = load_data(args.data_dir)
    evaluate = get_evaluator(use_npz)
    X = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    y = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    vy = np.asarray(va["y"], dtype=np.int64)
    vu = np.asarray(va["user"])
    dates = np.asarray(tr["date"])
    train_users = np.asarray(tr["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    pair_data = make_pair_pool(train_users, np.asarray(tr["y"]))

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    coarse_epochs = min(2, smoke_cap) if smoke_cap is not None else 2
    refine_epochs = min(4, smoke_cap) if smoke_cap is not None else 4
    final_epochs = min(args.epochs, smoke_cap) if smoke_cap is not None else args.epochs

    coarse_specs = [
        (0.15, 0.00003, 0.72, 1, 3.5, 0.0010),
        (0.19, 0.00010, 0.55, 2, 7.0, 0.0010),
        (0.23, 0.00030, 0.70, 2, 14.0, 0.0008),
        (0.27, 0.00100, 0.50, 1, 7.0, 0.0008),
        (0.31, 0.00300, 0.68, 2, 3.5, 0.0006),
        (0.34, 0.00010, 0.42, 1, 14.0, 0.0008),
        (0.37, 0.00050, 0.60, 3, 7.0, 0.0006),
        (0.40, 0.00150, 0.78, 2, 14.0, 0.0005),
    ]
    history = []
    coarse_results = []
    for probe_id, spec in enumerate(coarse_specs):
        dropout, weight_decay, gamma, step_size, half_life, lr = spec
        config = {
            "dropout": dropout, "weight_decay": weight_decay, "gamma": gamma,
            "step_size": step_size, "half_life": half_life, "lr": lr,
            "total_dim": total_dim, "batch_size": 32768,
        }
        metrics, _, curve = train_variant(
            config, args.seed + 100, coarse_epochs, X, y, dates, pair_data,
            Xv, vu, vy, evaluate, device, fraction=0.22)
        coarse_results.append((metrics[2], config))
        history.append({
            "stage": "coarse", "probe": probe_id + 1,
            "config": clean_config(config), "best_gauc": round(metrics[0], 6),
            "best_ndcg5": round(metrics[1], 6), "best_primary": round(metrics[2], 6),
            "curve": curve,
        })

    coarse_results.sort(key=lambda item: item[0], reverse=True)
    center = coarse_results[0][1]
    refine_specs = [
        (-0.045, 0.55, -0.09, -3.0, 0.85),
        (-0.025, 0.75, -0.04, -1.5, 0.93),
        (-0.010, 0.90, 0.02, 0.0, 1.00),
        (0.010, 1.10, -0.02, 1.5, 1.00),
        (0.025, 1.35, 0.05, 3.0, 1.08),
        (0.045, 1.75, 0.09, 5.0, 1.15),
    ]
    refine_results = []
    for probe_id, spec in enumerate(refine_specs):
        delta_dropout, weight_multiplier, delta_gamma, delta_half_life, lr_multiplier = spec
        config = dict(center)
        config["dropout"] = float(np.clip(center["dropout"] + delta_dropout, 0.12, 0.45))
        config["weight_decay"] = float(np.clip(center["weight_decay"] * weight_multiplier, 2e-5, 5e-3))
        config["gamma"] = float(np.clip(center["gamma"] + delta_gamma, 0.35, 0.85))
        config["half_life"] = float(np.clip(center["half_life"] + delta_half_life, 2.5, 20.0))
        config["lr"] = float(np.clip(center["lr"] * lr_multiplier, 0.00035, 0.0013))
        config["batch_size"] = 32768
        metrics, _, curve = train_variant(
            config, args.seed + 200, refine_epochs, X, y, dates, pair_data,
            Xv, vu, vy, evaluate, device)
        refine_results.append((metrics[2], config))
        history.append({
            "stage": "refine", "probe": probe_id + 1,
            "config": clean_config(config), "best_gauc": round(metrics[0], 6),
            "best_ndcg5": round(metrics[1], 6), "best_primary": round(metrics[2], 6),
            "curve": curve,
        })

    refine_results.sort(key=lambda item: item[0], reverse=True)
    winner = dict(refine_results[0][1])
    winner["batch_size"] = 16384

    member_logits = []
    member_probabilities = []
    member_ranks = []
    for member_id in range(7):
        member_seed = args.seed + member_id
        metrics, scores, curve = train_variant(
            winner, member_seed, final_epochs, X, y, dates, pair_data,
            Xv, vu, vy, evaluate, device, half_checkpoints=True)
        probabilities = sigmoid(scores)
        ranks = per_user_ranks(scores, vu)
        member_logits.append(scores)
        member_probabilities.append(probabilities)
        member_ranks.append(ranks)
        history.append({
            "stage": "ensemble_member", "member": member_id + 1,
            "seed": member_seed, "config": clean_config(winner),
            "best_gauc": round(metrics[0], 6),
            "best_ndcg5": round(metrics[1], 6),
            "best_primary": round(metrics[2], 6), "curve": curve,
        })

    probability_matrix = np.stack(member_probabilities)
    rank_matrix = np.stack(member_ranks)
    design_history = []
    best_design = None
    best_scores = None
    best_metrics = None
    for member_count in (3, 5, 7):
        designs = (
            ("probability_average", probability_matrix[:member_count].mean(axis=0)),
            ("per_user_rank_average", rank_matrix[:member_count].mean(axis=0)),
        )
        for rule, scores in designs:
            metrics = metric_values(evaluate(vu, vy, scores))
            record = {
                "member_count": member_count, "combination_rule": rule,
                "seeds": [args.seed + i for i in range(member_count)],
                "gauc": round(metrics[0], 6), "ndcg5": round(metrics[1], 6),
                "primary": round(metrics[2], 6),
            }
            design_history.append(record)
            if best_metrics is None or metrics[2] > best_metrics[2] + 1e-12:
                best_metrics = metrics
                best_scores = np.asarray(scores, dtype=np.float32).copy()
                best_design = dict(record)

    final_result = evaluate(vu, vy, best_scores)
    gauc, ndcg5, primary = metric_values(final_result)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({
            "gauc": gauc, "ndcg5": ndcg5, "primary": primary,
            "selected_config": clean_config(winner),
            "selected_ensemble": best_design,
            "ensemble_design_history": design_history,
            "history": history,
        }, fh)

    videos = np.asarray(va["video"])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(best_scores):
            fh.write("%d,%s,%s,%.9g\n" %
                     (row_id, str(vu[row_id]), str(videos[row_id]), float(score)))


if __name__ == "__main__":
    main()
