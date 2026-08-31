"""Wide random search with successive halving for the official-parity FM baseline."""
import argparse
import csv
import gc
import json
import math
import os
import sys

import numpy as np
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = float(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.emb(x)
        if self.dropout > 0.0:
            e = torch.nn.functional.dropout(e, p=self.dropout, training=self.training)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def load_fast(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"))
    va = np.load(os.path.join(data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    video_offset = int(field_dims[0])
    video_ids = va["X"][:, 1].astype(np.int64) - video_offset
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "users": va["user"],
        "videos": video_ids,
        "field_dims": field_dims,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
            }
            if training or "long_view" in r:
                item["long_view"] = float(r["long_view"])
            rows.append(item)
    return rows


def make_map(values):
    uniq = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(uniq)}


def load_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    user_map = make_map([r["user_id"] for r in train_rows])
    video_map = make_map([r["video_id"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    durations = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    positive = durations[durations > 0]
    if len(positive) == 0:
        edges = np.asarray([], dtype=np.float64)
    else:
        edges = np.unique(np.quantile(np.log1p(positive), np.linspace(0.0, 1.0, 17)[1:-1]))
    dims = np.asarray([len(user_map) + 1, len(video_map) + 1, 1,
                       len(tab_map) + 1, len(edges) + 2], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(r["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(r["tab"], 0) + offsets[3]
            bucket = int(np.searchsorted(edges, np.log1p(max(0.0, r["duration_ms"])), side="right")) + 1
            x[i, 4] = bucket + offsets[4]
        return x

    return {
        "Xt": encode(train_rows),
        "yt": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "Xv": encode(val_rows),
        "yv": np.asarray([r["long_view"] for r in val_rows], dtype=np.int64),
        "users": np.asarray([int(r["user_id"]) for r in val_rows]),
        "videos": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": dims,
        "fast": False,
    }


def metric_values(m):
    return {
        "gauc": float(m["GAUC"] if "GAUC" in m else m["gauc"]),
        "ndcg5": float(m["nDCG@5"] if "nDCG@5" in m else m["ndcg5"]),
        "primary": float(m["primary"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    progress_path = os.path.join(a.out_dir, "progress.log")

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = max(1, int(smoke)) if smoke is not None else None
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast = (os.path.exists(os.path.join(a.data_dir, "train.npz")) and
            os.path.exists(os.path.join(a.data_dir, "val.npz")))
    data = load_fast(a.data_dir) if fast else load_csv_data(a.data_dir)
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate

    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    Xv = torch.from_numpy(data["Xv"])
    total_dim = int(data["field_dims"].sum())
    n = len(yt)
    bs = 8192
    rng = np.random.default_rng(a.seed + 7717)
    k_choices = np.asarray([8, 12, 16, 16, 24, 32, 48], dtype=np.int64)
    count = 4 if smoke_cap is not None else 72
    configs = []
    for i in range(count):
        configs.append({
            "id": i,
            "seed": int(a.seed + 1009 * (i + 1)),
            "k": int(rng.choice(k_choices)),
            "lr": float(10.0 ** rng.uniform(math.log10(2e-4), math.log10(5e-3))),
            "weight_decay": float(10.0 ** rng.uniform(-8.0, math.log10(3e-3))),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "lr_gamma": float(rng.uniform(0.72, 1.0)),
        })

    probe_history = []

    def run_training(config, requested_epochs, stage, keep_scores):
        epochs = min(int(requested_epochs), int(a.epochs))
        if smoke_cap is not None:
            epochs = min(epochs, smoke_cap)
        epochs = max(1, epochs)
        seed = int(config["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = FM(total_dim, config["k"], config["dropout"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                               weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=config["lr_gamma"])
        bce = torch.nn.BCEWithLogitsLoss()
        best_primary = -1.0
        best_scores = None
        best_metrics = None
        best_epoch = 0
        curve = []
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n)
            last_loss = 0.0
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                xb = Xt[idx].to(device, non_blocking=True)
                yb = yt[idx].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().cpu().item())
            scheduler.step()
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, len(Xv), 65536):
                    xb = Xv[start:start + 65536].to(device, non_blocking=True)
                    chunks.append(model(xb).detach().cpu().numpy())
            scores = np.concatenate(chunks)
            m = metric_values(evaluate(data["users"], data["yv"], scores))
            curve.append({
                "epoch": epoch + 1,
                "train_loss": round(last_loss, 6),
                "gauc": round(m["gauc"], 7),
                "ndcg5": round(m["ndcg5"], 7),
                "primary": round(m["primary"], 7),
            })
            if m["primary"] > best_primary + 1e-12:
                best_primary = m["primary"]
                best_epoch = epoch + 1
                best_metrics = m
                if keep_scores:
                    best_scores = scores.copy()
        record = {
            "stage": stage,
            "max_epochs": epochs,
            "config": dict(config),
            "best_epoch": best_epoch,
            "gauc": best_metrics["gauc"],
            "ndcg5": best_metrics["ndcg5"],
            "primary": best_metrics["primary"],
            "curve": curve,
        }
        probe_history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps({k: v for k, v in record.items() if k != "curve"}, sort_keys=True) + "\n")
        del model, opt, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_primary, best_scores, best_metrics, best_epoch

    if smoke_cap is not None:
        stage_plan = [(1, len(configs))]
    else:
        stage_plan = [(3, 26), (7, 8), (12, 1)]

    active = list(configs)
    for stage_index, (stage_epochs, survivors) in enumerate(stage_plan):
        results = []
        for config in active:
            score, _, _, _ = run_training(config, stage_epochs,
                                           "probe_%d" % (stage_index + 1), False)
            results.append((score, config))
        results.sort(key=lambda z: (-z[0], z[1]["id"]))
        active = [z[1] for z in results[:min(survivors, len(results))]]

    winner = active[0]
    final_epochs = min(a.epochs, 12)
    if smoke_cap is not None:
        final_epochs = min(final_epochs, smoke_cap)
    _, best_scores, final_metrics, best_epoch = run_training(
        winner, final_epochs, "final_full_fidelity", True)

    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": winner,
        "selected_checkpoint_epoch": best_epoch,
        "history": probe_history,
    }
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(a.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["users"][i], data["videos"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
