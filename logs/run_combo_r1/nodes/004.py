import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    Xtr = np.asarray(tr["X"], dtype=np.int64)
    Xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    aux = []
    aux_names = []

    if "click" in tr.files:
        aux.append(np.asarray(tr["click"], dtype=np.float32).clip(0, 1))
        aux_names.append("click")
    for name in ("like", "follow", "comment", "forward"):
        if name in tr.files:
            aux.append(np.asarray(tr[name], dtype=np.float32).clip(0, 1))
            aux_names.append(name)

    play = np.asarray(tr["play_time_ms"], dtype=np.float32)
    duration = np.asarray(tr["duration_ms"], dtype=np.float32)
    ratio = np.maximum(play, 0.0) / np.maximum(duration, 1.0)
    for threshold in (0.25, 0.50, 0.75, 1.00):
        aux.append((ratio >= threshold).astype(np.float32))
        aux_names.append("watch_ratio_ge_%g" % threshold)
    aux_y = np.stack(aux, axis=1).astype(np.float32)

    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1])))
    encoded_video = Xva[:, 1] - offsets[1]
    video_out = np.asarray(va["video"]) if "video" in va.files else encoded_video
    return {
        "Xtr": Xtr,
        "Xva": Xva,
        "ytr": ytr,
        "yva": yva,
        "utr": utr,
        "uva": uva,
        "video": video_out,
        "field_dims": field_dims,
        "aux": aux_y,
        "aux_names": aux_names,
        "fast": True,
    }


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user": r.get("user_id", ""),
                "video": r.get("video_id", ""),
                "author": r.get("author_id", ""),
                "tab": r.get("tab", ""),
                "duration": float(r.get("duration_ms", 0) or 0),
                "label": float(r.get("long_view", 0) or 0),
            }
            if training:
                item["click"] = float(r.get("click", 0) or 0)
                item["like"] = float(r.get("like", 0) or 0)
                item["play"] = float(r.get("play_time_ms", 0) or 0)
            rows.append(item)
    return rows


def load_csv(data_dir):
    train_rows = read_csv_rows(Path(data_dir) / "train.csv", True)
    val_rows = read_csv_rows(Path(data_dir) / "val.csv", False)
    durations = np.asarray([r["duration"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    fields = ("user", "video", "author", "tab")
    maps = {}
    dims = []
    for field in fields:
        values = sorted({str(r[field]) for r in train_rows})
        maps[field] = {v: i + 1 for i, v in enumerate(values)}
        dims.append(len(values) + 1)
    dims.append(10)
    offsets = np.concatenate(([0], np.cumsum(np.asarray(dims[:-1], dtype=np.int64))))

    def encode(rows):
        X = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            for j, field in enumerate(fields):
                X[i, j] = maps[field].get(str(r[field]), 0) + offsets[j]
            X[i, 4] = int(np.searchsorted(quantiles, r["duration"], side="right")) + offsets[4]
        return X

    Xtr = encode(train_rows)
    Xva = encode(val_rows)
    ytr = np.asarray([r["label"] for r in train_rows], dtype=np.float32)
    yva = np.asarray([r["label"] for r in val_rows], dtype=np.float32)
    utr = np.asarray([r["user"] for r in train_rows])
    uva = np.asarray([r["user"] for r in val_rows])
    video = np.asarray([r["video"] for r in val_rows])
    click = np.asarray([r["click"] for r in train_rows], dtype=np.float32).clip(0, 1)
    like = np.asarray([r["like"] for r in train_rows], dtype=np.float32).clip(0, 1)
    play = np.asarray([r["play"] for r in train_rows], dtype=np.float32)
    duration = np.asarray([r["duration"] for r in train_rows], dtype=np.float32)
    ratio = np.maximum(play, 0.0) / np.maximum(duration, 1.0)
    aux = [click, like]
    aux_names = ["click", "like"]
    for threshold in (0.25, 0.50, 0.75, 1.00):
        aux.append((ratio >= threshold).astype(np.float32))
        aux_names.append("watch_ratio_ge_%g" % threshold)
    return {
        "Xtr": Xtr,
        "Xva": Xva,
        "ytr": ytr,
        "yva": yva,
        "utr": utr,
        "uva": uva,
        "video": video,
        "field_dims": np.asarray(dims, dtype=np.int64),
        "aux": np.stack(aux, axis=1).astype(np.float32),
        "aux_names": aux_names,
        "fast": False,
    }


def get_evaluator(fast):
    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def normalize_metrics(result):
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            continue
        count = max(len(pos), len(neg))
        pos_parts.append(rng.choice(pos, size=count, replace=len(pos) < count))
        neg_parts.append(rng.choice(neg, size=count, replace=len(neg) < count))
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts), np.concatenate(neg_parts)


class DCNMultiTask(nn.Module):
    def __init__(self, total_features, n_fields, n_aux, dim=16, hidden=128, dropout=0.30):
        super().__init__()
        width = n_fields * dim
        self.embedding = nn.Embedding(total_features, dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.cross_w1 = nn.Parameter(torch.empty(width))
        self.cross_b1 = nn.Parameter(torch.zeros(width))
        self.cross_w2 = nn.Parameter(torch.empty(width))
        self.cross_b2 = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.cross_w1, std=0.01)
        nn.init.normal_(self.cross_w2, std=0.01)
        self.embed_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        rep_width = width + hidden
        self.main_head = nn.Linear(rep_width, 1)
        self.aux_heads = nn.ModuleList([nn.Linear(rep_width, 1) for _ in range(n_aux)])

    def representation(self, x):
        x0 = self.embed_dropout(self.embedding(x).flatten(1))
        x1 = x0 * torch.sum(x0 * self.cross_w1, dim=1, keepdim=True) + self.cross_b1 + x0
        x2 = x0 * torch.sum(x1 * self.cross_w2, dim=1, keepdim=True) + self.cross_b2 + x1
        deep = self.mlp(x0)
        return torch.cat((x2, deep), dim=1)

    def forward(self, x):
        rep = self.representation(x)
        main = self.main_head(rep).squeeze(1)
        if len(self.aux_heads):
            aux = torch.stack([head(rep).squeeze(1) for head in self.aux_heads], dim=1)
        else:
            aux = main.new_empty((len(main), 0))
        return main, aux

    def score(self, x):
        return self.main_head(self.representation(x)).squeeze(1)


def predict(model, X, device, batch_size):
    model.eval()
    output = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.long, device=device)
            output[start:start + len(xb)] = model.score(xb).detach().cpu().numpy()
    return output


def train_one(data, aux_weight, run_seed, epochs, device, evaluate):
    set_seed(run_seed)
    Xtr = data["Xtr"]
    Xva = data["Xva"]
    ytr = data["ytr"]
    aux_y = data["aux"]
    total_features = max(int(np.sum(data["field_dims"])), int(Xtr.max(initial=0)) + 1, int(Xva.max(initial=0)) + 1)
    model = DCNMultiTask(total_features, Xtr.shape[1], aux_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    batch_size = 4096 if device.type == "cuda" else 2048
    eval_batch = 16384 if device.type == "cuda" else 8192
    pair_pos, pair_neg = make_pairs(data["utr"], ytr, run_seed + 991)
    rng = np.random.default_rng(run_seed + 17)
    best_gauc = -1.0
    best_primary = -1.0
    best_scores = None
    best_epoch = 0
    n = len(Xtr)

    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(n)
        if len(pair_pos):
            pair_order = rng.permutation(len(pair_pos))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0
        for start in range(0, n, batch_size):
            ids = permutation[start:start + batch_size]
            xb = torch.as_tensor(Xtr[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[ids], dtype=torch.float32, device=device)
            ab = torch.as_tensor(aux_y[ids], dtype=torch.float32, device=device)
            logits, aux_logits = model(xb)
            main_loss = F.binary_cross_entropy_with_logits(logits, yb)

            if len(pair_pos):
                need = len(ids)
                if pair_cursor + need > len(pair_order):
                    pair_order = rng.permutation(len(pair_pos))
                    pair_cursor = 0
                chosen = pair_order[pair_cursor:pair_cursor + need]
                pair_cursor += len(chosen)
                pxb = torch.as_tensor(Xtr[pair_pos[chosen]], dtype=torch.long, device=device)
                nxb = torch.as_tensor(Xtr[pair_neg[chosen]], dtype=torch.long, device=device)
                bpr_loss = F.softplus(-(model.score(pxb) - model.score(nxb))).mean()
            else:
                bpr_loss = main_loss.new_zeros(())

            if aux_weight > 0 and aux_logits.numel():
                aux_loss = F.binary_cross_entropy_with_logits(aux_logits, ab)
            else:
                aux_loss = main_loss.new_zeros(())
            loss = 0.5 * main_loss + 0.5 * bpr_loss + aux_weight * aux_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        scores = predict(model, Xva, device, eval_batch)
        metrics = normalize_metrics(evaluate(data["uva"], data["yva"], scores))
        if metrics["gauc"] > best_gauc:
            best_gauc = metrics["gauc"]
            best_primary = metrics["primary"]
            best_scores = scores.copy()
            best_epoch = epoch + 1
    return best_scores, {"best_epoch": best_epoch, "best_gauc": best_gauc, "best_primary": best_primary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fast = (Path(args.data_dir) / "train.npz").exists() and (Path(args.data_dir) / "val.npz").exists()
    data = load_npz(args.data_dir) if fast else load_csv(args.data_dir)
    evaluate = get_evaluator(fast)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epoch_cap = int(smoke) if smoke is not None else None
    epochs = 7 if epoch_cap is None else max(1, min(7, epoch_cap))

    if smoke is not None:
        weights = [0.0, 0.05]
        seeds = [args.seed]
    else:
        weights = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
        seeds = [args.seed + 1009 * i for i in range(5)]

    history = []
    config_results = []
    progress_path = Path(args.out_dir) / "progress.log"
    for weight in weights:
        score_sum = np.zeros(len(data["Xva"]), dtype=np.float64)
        run_metrics = []
        for run_seed in seeds:
            scores, training_info = train_one(data, weight, run_seed, epochs, device, evaluate)
            score_sum += scores
            metrics = normalize_metrics(evaluate(data["uva"], data["yva"], scores))
            entry = {
                "aux_weight": weight,
                "seed": run_seed,
                "epochs": epochs,
                "best_epoch": training_info["best_epoch"],
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
            }
            history.append(entry)
            run_metrics.append(metrics)
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        ensemble_scores = (score_sum / len(seeds)).astype(np.float32)
        ensemble_metrics = normalize_metrics(evaluate(data["uva"], data["yva"], ensemble_scores))
        aggregate = {
            "aux_weight": weight,
            "seeds": seeds,
            "mean_single_primary": float(np.mean([m["primary"] for m in run_metrics])),
            "ensemble_gauc": ensemble_metrics["gauc"],
            "ensemble_ndcg5": ensemble_metrics["ndcg5"],
            "ensemble_primary": ensemble_metrics["primary"],
        }
        history.append({"aggregate": aggregate})
        config_results.append((ensemble_metrics["primary"], weight, ensemble_scores, ensemble_metrics))
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"aggregate": aggregate}, sort_keys=True) + "\n")

    config_results.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    _, selected_weight, final_scores, final_metrics = config_results[0]

    pred_path = Path(args.out_dir) / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(data["uva"], data["video"], final_scores)):
            writer.writerow([i, user.item() if isinstance(user, np.generic) else user,
                             video.item() if isinstance(video, np.generic) else video,
                             "%.9g" % float(score)])

    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_aux_weight": selected_weight,
        "aux_targets": data["aux_names"],
        "history": history,
    }
    with open(Path(args.out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, sort_keys=True)


if __name__ == "__main__":
    main()
