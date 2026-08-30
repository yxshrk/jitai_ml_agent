import argparse
import csv
import datetime
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.drop(self.emb(x))
        s = e.sum(dim=1)
        pair = 0.5 * (s.square() - e.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.lin(x).sum(dim=(1, 2)) + pair


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_drop = torch.nn.Dropout(dropout)
        dim = fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(dim))
        self.cross_out = torch.nn.Linear(dim, 1)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x0 = self.emb_drop(self.emb(x)).flatten(1)
        xl = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b
        return self.bias + self.cross_out(xl).squeeze(1) + self.mlp(x0).squeeze(1)


def metric_values(m):
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    data = {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "ut": tr["user"],
        "date": tr["date"],
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "uv": va["user"],
        "field_dims": tr["field_dims"].astype(np.int64),
    }
    offset = int(data["field_dims"][0])
    data["video_out"] = data["Xv"][:, 1] - offset
    return data


def quantile_edges(values, buckets=10):
    q = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    return np.unique(np.quantile(values.astype(np.float64), q))


def load_csv_data(data_dir):
    train_rows = []
    durations = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rec = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            }
            train_rows.append(rec)
            durations.append(rec["duration_ms"])
    edges = quantile_edges(np.asarray(durations, dtype=np.float64), 10)
    vocab = [{}, {}, {"__author_unknown__": 0}, {}, {}]

    def token(row, field):
        if field == 0:
            return row["user_id"]
        if field == 1:
            return row["video_id"]
        if field == 2:
            return "__author_unknown__"
        if field == 3:
            return row["tab"]
        return str(int(np.searchsorted(edges, row["duration_ms"], side="right")))

    for row in train_rows:
        for f in (0, 1, 3, 4):
            value = token(row, f)
            if value not in vocab[f]:
                vocab[f][value] = len(vocab[f])
    field_dims = np.asarray([len(v) + 1 for v in vocab], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(row):
        out = np.empty(5, dtype=np.int64)
        for f in range(5):
            value = token(row, f)
            out[f] = offsets[f] + vocab[f].get(value, len(vocab[f]))
        return out

    Xt = np.stack([encode(r) for r in train_rows])
    yt = np.asarray([r["long_view"] for r in train_rows], dtype=np.float32)
    ut = np.asarray([r["user_id"] for r in train_rows])
    dates = np.asarray([r["date"] for r in train_rows])
    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })
    return {
        "Xt": Xt,
        "yt": yt,
        "ut": ut,
        "date": dates,
        "Xv": np.stack([encode(r) for r in val_rows]),
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "uv": np.asarray([r["user_id"] for r in val_rows]),
        "video_out": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
    }


def date_ordinal(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        digits = digits[:8]
        try:
            return datetime.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).toordinal()
        except ValueError:
            pass
    try:
        return int(float(text))
    except ValueError:
        return 0


def recency_weights(dates):
    unique, inverse = np.unique(dates, return_inverse=True)
    ordinals = np.asarray([date_ordinal(x) for x in unique], dtype=np.float64)
    ages = np.max(ordinals) - ordinals
    weights = np.exp(-math.log(2.0) * ages / 7.0)[inverse].astype(np.float32)
    weights /= max(float(weights.mean()), 1e-8)
    return weights


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        idx = order[left:right]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def predict(model, X, device, batch_size=65536):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device, non_blocking=True)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks)


def train_one(config, seed, epochs, data, evaluator, device, half_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dropout = 0.30 if config["regularization"] == "strong" else 0.0
    total_dim = int(data["field_dims"].sum())
    if config["architecture"] == "fm":
        model = FM(total_dim, k=16, dropout=dropout)
    else:
        model = DCNLite(total_dim, fields=data["Xt"].shape[1], k=16,
                        hidden=128, dropout=dropout)
    model.to(device)
    if config["regularization"] == "strong":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    if config["weighting"] == "recency":
        sample_weight = torch.from_numpy(data["recency"])
    else:
        sample_weight = torch.ones(len(yt), dtype=torch.float32)
    pair_pos, pair_neg = data["pairs"]
    pair_pos_t = torch.from_numpy(pair_pos)
    pair_neg_t = torch.from_numpy(pair_neg)
    n = len(yt)
    batch_size = 8192
    steps = int(math.ceil(n / batch_size))
    best_primary = -1.0
    best_scores = None
    curve = []
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17011)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        checkpoints = {steps - 1}
        if half_epoch and steps > 1:
            checkpoints.add(max(0, int(math.ceil(steps / 2.0)) - 1))
        running_loss = 0.0
        seen = 0
        for step, start in enumerate(range(0, n, batch_size)):
            idx = permutation[start:start + batch_size]
            xb = Xt[idx].to(device, non_blocking=True)
            yb = yt[idx].to(device, non_blocking=True)
            wb = sample_weight[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            point_loss = (bce(logits, yb) * wb).mean()
            if config["loss"] == "hybrid" and len(pair_pos_t):
                chosen = torch.randint(len(pair_pos_t), (len(idx),), generator=generator)
                pi = pair_pos_t[chosen]
                ni = pair_neg_t[chosen]
                pair_x = torch.cat((Xt[pi], Xt[ni]), dim=0).to(device, non_blocking=True)
                pair_logits = model(pair_x)
                pair_loss = torch.nn.functional.softplus(
                    -(pair_logits[:len(idx)] - pair_logits[len(idx):])
                ).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            if step in checkpoints:
                scores = predict(model, Xv, device)
                metrics = metric_values(evaluator(data["uv"], data["yv"], scores))
                fraction = 0.5 if step != steps - 1 else 1.0
                curve.append({
                    "epoch": epoch + fraction,
                    "train_loss": round(running_loss / max(seen, 1), 6),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-9:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
                model.train()
        if scheduler is not None:
            scheduler.step()
    result = {
        "best_primary": float(best_primary),
        "best_gauc": float(max(x["val_gauc"] for x in curve)),
        "best_epoch": float(max(curve, key=lambda x: x["val_primary"])["epoch"]),
        "curve": curve,
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, best_scores


def config_name(config):
    return "-".join([config["architecture"], config["loss"],
                     config["weighting"], config["regularization"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
    fast_path = (os.path.exists(os.path.join(args.data_dir, "train.npz")) and
                 os.path.exists(os.path.join(args.data_dir, "val.npz")))
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        data = load_csv_data(args.data_dir)
    data["recency"] = recency_weights(data["date"])
    data["pairs"] = build_pairs(data["ut"], data["yt"], args.seed)
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    final_epochs = max(1, args.epochs)
    probe_epochs = 3
    refine_epochs = min(10, final_epochs)
    repeats = 5
    refine_count = 6
    refine_repeats = 5
    if smoke_cap is not None:
        final_epochs = min(final_epochs, smoke_cap)
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        repeats = 1
        refine_count = 2
        refine_repeats = 1
    configs = []
    for architecture in ("fm", "dcn"):
        for loss in ("logloss", "hybrid"):
            for weighting in ("uniform", "recency"):
                for regularization in ("mild", "strong"):
                    configs.append({
                        "architecture": architecture,
                        "loss": loss,
                        "weighting": weighting,
                        "regularization": regularization,
                    })
    history = []
    cell_scores = {}
    progress_path = os.path.join(args.out_dir, "progress.log")
    for cell_index, config in enumerate(configs):
        name = config_name(config)
        cell_scores[name] = []
        for repeat in range(repeats):
            seed = args.seed + 1009 * repeat + 37 * cell_index
            result, _ = train_one(config, seed, probe_epochs, data, evaluator, device)
            cell_scores[name].append(result["best_primary"])
            record = {
                "phase": "matrix_probe",
                "config": dict(config),
                "seed": seed,
                "epochs": probe_epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "gauc": result["best_gauc"],
            }
            history.append(record)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
    summaries = []
    config_lookup = {config_name(c): c for c in configs}
    for name, scores in cell_scores.items():
        summaries.append({
            "config": dict(config_lookup[name]),
            "mean_primary": float(np.mean(scores)),
            "std_primary": float(np.std(scores)),
            "scores": [float(x) for x in scores],
        })
    summaries.sort(key=lambda x: x["mean_primary"], reverse=True)
    finalists = summaries[:refine_count]
    refined = []
    for rank, summary in enumerate(finalists):
        config = summary["config"]
        scores = []
        epochs_at_best = []
        for repeat in range(refine_repeats):
            seed = args.seed + 50021 + rank * 313 + repeat * 2017
            result, _ = train_one(config, seed, refine_epochs, data, evaluator, device)
            scores.append(result["best_primary"])
            epochs_at_best.append(result["best_epoch"])
            record = {
                "phase": "refinement",
                "config": dict(config),
                "seed": seed,
                "epochs": refine_epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "gauc": result["best_gauc"],
            }
            history.append(record)
            with open(progress_path, "a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        refined.append({
            "config": dict(config),
            "mean_primary": float(np.mean(scores)),
            "std_primary": float(np.std(scores)),
            "scores": [float(x) for x in scores],
            "mean_best_epoch": float(np.mean(epochs_at_best)),
        })
    refined.sort(key=lambda x: x["mean_primary"], reverse=True)
    winner = refined[0]["config"]
    final_result, best_scores = train_one(
        winner, args.seed, final_epochs, data, evaluator, device, half_epoch=True
    )
    final_metrics = metric_values(evaluator(data["uv"], data["yv"], best_scores))
    final_record = {
        "phase": "final",
        "config": dict(winner),
        "seed": args.seed,
        "epochs": final_epochs,
        "best_epoch": final_result["best_epoch"],
        "primary": final_metrics["primary"],
        "gauc": final_metrics["gauc"],
    }
    history.append(final_record)
    with open(progress_path, "a") as fh:
        fh.write(json.dumps(final_record, sort_keys=True) + "\n")
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "winner": winner,
        "matrix_summary": summaries,
        "refinement_summary": refined,
        "final_curve": final_result["curve"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{data['uv'][i]},{data['video_out'][i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
