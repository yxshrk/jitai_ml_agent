import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RankModel(torch.nn.Module):
    def __init__(self, total_dim, architecture, dropout, k=16):
        super().__init__()
        self.architecture = architecture
        self.dropout = torch.nn.Dropout(dropout)
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        if architecture == "dcn-lite":
            d = 5 * k
            self.cross_w = torch.nn.ParameterList([
                torch.nn.Parameter(torch.empty(d)) for _ in range(2)
            ])
            self.cross_b = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(d)) for _ in range(2)
            ])
            for w in self.cross_w:
                torch.nn.init.normal_(w, std=0.01)
            self.cross_out = torch.nn.Linear(d, 1, bias=False)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 1),
            )

    def forward(self, x):
        e = self.dropout(self.emb(x))
        summed = e.sum(1)
        pair = 0.5 * (summed.square() - e.square().sum(1)).sum(1)
        fm = self.bias + self.lin(x).sum((1, 2)) + pair
        if self.architecture == "fm":
            return fm
        x0 = e.reshape(e.shape[0], -1)
        cross = x0
        for w, b in zip(self.cross_w, self.cross_b):
            cross = x0 * (cross * w).sum(1, keepdim=True) + b + cross
        return fm + self.cross_out(cross).squeeze(1) + self.mlp(x0).squeeze(1)


def date_number(value):
    s = str(value)
    if s.endswith(".0"):
        s = s[:-2]
    s = s.replace("-", "")
    try:
        d = datetime.datetime.strptime(s[:8], "%Y%m%d").date()
        return d.toordinal()
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def make_csv_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    train_rows = []
    val_rows = []
    with open(train_path, newline="") as fh:
        for r in csv.DictReader(fh):
            train_rows.append((r["user_id"], r["video_id"], r["tab"],
                               float(r["duration_ms"]), float(r["long_view"]), r["date"]))
    with open(val_path, newline="") as fh:
        for r in csv.DictReader(fh):
            val_rows.append((r["user_id"], r["video_id"], r["tab"],
                             float(r["duration_ms"]), float(r["long_view"]), r["date"]))
    durations = np.asarray([r[3] for r in train_rows], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    maps = []
    for column in (0, 1, 2):
        values = sorted(set(r[column] for r in train_rows))
        maps.append({v: i + 1 for i, v in enumerate(values)})
    dims = [len(m) + 1 for m in maps]
    dims = [dims[0], dims[1], dims[1], dims[2], 10]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            user_code = maps[0].get(r[0], 0)
            video_code = maps[1].get(r[1], 0)
            tab_code = maps[2].get(r[2], 0)
            bucket = min(int(np.searchsorted(edges, r[3], side="right")), 9)
            x[i] = np.asarray([user_code, video_code, video_code, tab_code, bucket]) + offsets
        return x

    tr = {
        "X": encode(train_rows),
        "y": np.asarray([r[4] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r[0] for r in train_rows]),
        "date": np.asarray([r[5] for r in train_rows]),
        "field_dims": np.asarray(dims, dtype=np.int64),
    }
    va = {
        "X": encode(val_rows),
        "y": np.asarray([r[4] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r[0] for r in val_rows]),
        "video_raw": np.asarray([r[1] for r in val_rows]),
    }
    return tr, va, False


def load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_npz = np.load(train_npz)
        va_npz = np.load(val_npz)
        tr = {k: tr_npz[k] for k in ("X", "y", "user", "date", "field_dims")}
        va = {k: va_npz[k] for k in ("X", "y", "user")}
        offset = int(tr["field_dims"][0])
        va["video_raw"] = va["X"][:, 1].astype(np.int64) - offset
        return tr, va, True
    return make_csv_data(data_dir)


def build_recency_weights(dates):
    ordinals = np.asarray([date_number(v) for v in dates], dtype=np.float64)
    latest = float(ordinals.max()) if len(ordinals) else 0.0
    age = np.maximum(0.0, latest - ordinals)
    weights = np.exp2(-age / 7.0)
    weights /= max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def user_index_groups(users):
    user_text = users.astype(str)
    order = np.argsort(user_text, kind="stable")
    sorted_users = user_text[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    return [order[left:right].astype(np.int64) for left, right in zip(boundaries[:-1], boundaries[1:])]


def build_pairs(users, labels, seed):
    groups = user_index_groups(users)
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for idx in groups:
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(rng.choice(neg, size=len(pos), replace=True))
    if not pos_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def make_complete_user_batches(group_tensors, target_rows, device):
    order = torch.randperm(len(group_tensors), device=device).detach().cpu().tolist()
    batches = []
    current = []
    current_rows = 0
    for group_id in order:
        group = group_tensors[group_id]
        group_rows = int(group.numel())
        if current and current_rows + group_rows > target_rows:
            batches.append(current)
            current = []
            current_rows = 0
        current.append(group)
        current_rows += group_rows
    if current:
        batches.append(current)
    return batches


def centered_logits(raw_logits, group_ids, group_count, global_bias):
    sums = torch.zeros(group_count, dtype=raw_logits.dtype, device=raw_logits.device)
    sums.scatter_add_(0, group_ids, raw_logits)
    counts = torch.bincount(group_ids, minlength=group_count).to(raw_logits.dtype)
    means = sums / counts.clamp_min(1.0)
    return raw_logits - means[group_ids] + global_bias


def metric_values(evaluator, users, labels, scores):
    m = evaluator(users, labels.astype(int), scores)
    return {
        "gauc": float(m.get("GAUC", m.get("gauc"))),
        "ndcg5": float(m.get("nDCG@5", m.get("ndcg5"))),
        "primary": float(m["primary"]),
    }


def train_once(config, seed, epochs, data, evaluator, device, keep_scores):
    (Xt, yt, Xv, users_v, labels_v, uniform_w, recency_w, pair_pos,
     pair_neg, total_dim, train_user_groups) = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RankModel(total_dim, config["architecture"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_gamma"])
    weights = recency_w if config["weighting"] == "recency-7d" else uniform_w
    bs = 8192
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    learning = []
    for epoch in range(epochs):
        model.train()
        batches = make_complete_user_batches(train_user_groups, bs, device)
        steps = len(batches)
        epoch_loss = 0.0
        seen = 0
        checkpoint_steps = {max(1, steps // 2), steps}
        for step, batch_groups in enumerate(batches):
            lengths = torch.as_tensor([g.numel() for g in batch_groups], dtype=torch.long, device=device)
            idx = torch.cat(batch_groups)
            group_ids = torch.repeat_interleave(
                torch.arange(len(batch_groups), device=device), lengths
            )
            raw_logits = model(Xt[idx])
            gauge_logits = centered_logits(raw_logits, group_ids, len(batch_groups), model.bias)
            point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gauge_logits, yt[idx], reduction="none"
            )
            bce_loss = (point_loss * weights[idx]).mean()
            if config["loss"] == "bpr-hybrid" and pair_pos.numel() > 0:
                pick = torch.randint(0, pair_pos.numel(), (idx.numel(),), device=device)
                pidx = pair_pos[pick]
                nidx = pair_neg[pick]
                margin = model(Xt[pidx]) - model(Xt[nidx])
                pair_loss = (torch.nn.functional.softplus(-margin) * weights[pidx]).mean()
                loss = 0.5 * bce_loss + 0.5 * pair_loss
            else:
                loss = bce_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item()) * int(idx.numel())
            seen += int(idx.numel())
            if step + 1 in checkpoint_steps:
                model.eval()
                chunks = []
                with torch.no_grad():
                    for j in range(0, Xv.shape[0], 65536):
                        chunks.append(model(Xv[j:j + 65536]).detach().cpu().numpy())
                scores = np.concatenate(chunks)
                mv = metric_values(evaluator, users_v, labels_v, scores)
                learning.append({
                    "epoch": epoch + 1,
                    "fraction": 0.5 if step + 1 < steps else 1.0,
                    "train_loss": round(epoch_loss / max(seen, 1), 6),
                    "val_gauc": round(mv["gauc"], 6),
                    "val_primary": round(mv["primary"], 6),
                })
                if mv["primary"] > best_primary + 1e-8:
                    best_primary = mv["primary"]
                    best_metrics = mv
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()
        scheduler.step()
    return best_metrics, best_scores, learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=14)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")

    tr, va, fast_path = load_data(args.data_dir)
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
    else:
        from harness.evaluate_provisional import evaluate as evaluator

    x_train = torch.from_numpy(tr["X"].astype(np.int64)).to(device)
    y_train_np = tr["y"].astype(np.float32)
    y_train = torch.from_numpy(y_train_np).to(device)
    x_val = torch.from_numpy(va["X"].astype(np.int64)).to(device)
    uniform = torch.ones(len(y_train_np), dtype=torch.float32, device=device)
    recency = torch.from_numpy(build_recency_weights(tr["date"])).to(device)
    pos_np, neg_np = build_pairs(tr["user"], y_train_np, args.seed + 271)
    pair_pos = torch.from_numpy(pos_np).to(device)
    pair_neg = torch.from_numpy(neg_np).to(device)
    train_user_groups = [
        torch.from_numpy(group).to(device) for group in user_index_groups(tr["user"])
    ]
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    train_data = (x_train, y_train, x_val, va["user"], va["y"], uniform,
                  recency, pair_pos, pair_neg, total_dim, train_user_groups)

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
    probe_epochs = 6
    refine_epochs = 8
    final_epochs = args.epochs
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)

    regularizations = {
        "mild": (0.10, 1e-5, 0.90),
        "strong": (0.30, 1e-3, 0.65),
    }
    configs = []
    for architecture in ("fm", "dcn-lite"):
        for loss in ("logloss", "bpr-hybrid"):
            for weighting in ("uniform", "recency-7d"):
                for reg_name in ("mild", "strong"):
                    dropout, wd, gamma = regularizations[reg_name]
                    configs.append({
                        "architecture": architecture,
                        "loss": loss,
                        "weighting": weighting,
                        "regularization": reg_name,
                        "dropout": dropout,
                        "weight_decay": wd,
                        "lr_gamma": gamma,
                        "point_objective": "complete-slate-user-centered-bce",
                    })

    history = []
    progress_path = os.path.join(args.out_dir, "progress.log")
    probe_seeds = [args.seed] if smoke_cap is not None else [args.seed, args.seed + 101, args.seed + 202]
    cell_results = []
    for cell_id, config in enumerate(configs):
        scores = []
        for run_seed in probe_seeds:
            mv, _, learning = train_once(config, run_seed, probe_epochs, train_data,
                                         evaluator, device, False)
            record = {
                "stage": "matrix_probe",
                "cell": cell_id,
                "seed": run_seed,
                "config": config,
                "epochs": probe_epochs,
                "gauc": mv["gauc"],
                "ndcg5": mv["ndcg5"],
                "primary": mv["primary"],
                "learning": learning,
            }
            history.append(record)
            scores.append(mv["primary"])
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({"cell": cell_id, "seed": run_seed,
                                     "config": config, "primary": mv["primary"]}) + "\n")
        cell_results.append((float(np.mean(scores)), float(np.std(scores)), config))
    cell_results.sort(key=lambda z: (-z[0], z[1]))
    stage_winner = dict(cell_results[0][2])

    refine_grid = [
        (0.05, 1e-6, 0.95),
        (0.10, 1e-5, 0.90),
        (0.20, 1e-4, 0.80),
        (0.30, 1e-3, 0.65),
        (0.40, 2e-3, 0.60),
    ]
    refine_seeds = [args.seed + 303] if smoke_cap is not None else [args.seed + 303, args.seed + 404]
    refined = []
    for refine_id, (dropout, wd, gamma) in enumerate(refine_grid):
        config = dict(stage_winner)
        config.update({"regularization": "refined", "dropout": dropout,
                       "weight_decay": wd, "lr_gamma": gamma})
        scores = []
        for run_seed in refine_seeds:
            mv, _, learning = train_once(config, run_seed, refine_epochs, train_data,
                                         evaluator, device, False)
            history.append({
                "stage": "regularization_refinement",
                "cell": refine_id,
                "seed": run_seed,
                "config": config,
                "epochs": refine_epochs,
                "gauc": mv["gauc"],
                "ndcg5": mv["ndcg5"],
                "primary": mv["primary"],
                "learning": learning,
            })
            scores.append(mv["primary"])
            with open(progress_path, "a") as fh:
                fh.write(json.dumps({"refine": refine_id, "seed": run_seed,
                                     "config": config, "primary": mv["primary"]}) + "\n")
        refined.append((float(np.mean(scores)), float(np.std(scores)), config))
    refined.sort(key=lambda z: (-z[0], z[1]))
    final_config = refined[0][2]
    final_metrics, final_scores, final_learning = train_once(
        final_config, args.seed, final_epochs, train_data, evaluator, device, True
    )
    history.append({
        "stage": "final",
        "seed": args.seed,
        "config": final_config,
        "epochs": final_epochs,
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "learning": final_learning,
    })

    metrics_payload = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_config": final_config,
        "matrix_summary": [
            {"mean_primary": mean, "std_primary": std, "config": config}
            for mean, std, config in cell_results
        ],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(f"{i},{va['user'][i]},{va['video_raw'][i]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
