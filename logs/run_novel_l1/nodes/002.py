import argparse
import csv
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc", 0.0))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5", 0.0))),
        "primary": float(metrics["primary"]),
    }


def load_npz(data_dir):
    from data.official.evaluate import evaluate

    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = tr["field_dims"].astype(np.int64)
    xtr = tr["X"].astype(np.int64)
    xva = va["X"].astype(np.int64)
    videos = va["video"] if "video" in va.files else xva[:, 1] - int(field_dims[0])
    dates = tr["date"] if "date" in tr.files else np.zeros(len(xtr), dtype=np.int64)
    return {
        "Xtr": xtr,
        "ytr": tr["y"].astype(np.float32),
        "utr": np.asarray(tr["user"]),
        "date": np.asarray(dates),
        "Xva": xva,
        "yva": va["y"].astype(np.int64),
        "uva": np.asarray(va["user"]),
        "vva": np.asarray(videos),
        "field_dims": field_dims,
        "evaluate": evaluate,
    }


def read_csv(path):
    rows = []
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row.get("date", "0"),
                "long_view": float(row["long_view"]),
            })
    return rows


def make_map(values):
    result = {}
    for value in values:
        if value not in result:
            result[value] = len(result) + 1
    return result


def load_csv(data_dir):
    from harness.evaluate_provisional import evaluate

    train_rows = read_csv(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv(os.path.join(data_dir, "val.csv"))
    user_map = make_map(row["user_id"] for row in train_rows)
    video_map = make_map(row["video_id"] for row in train_rows)
    tab_map = make_map(row["tab"] for row in train_rows)
    durations = np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float64)
    cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray(
        [len(user_map) + 1, len(video_map) + 1, 1, len(tab_map) + 1, 10],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0) + offsets[0]
            x[i, 1] = video_map.get(row["video_id"], 0) + offsets[1]
            x[i, 2] = offsets[2]
            x[i, 3] = tab_map.get(row["tab"], 0) + offsets[3]
            x[i, 4] = int(np.searchsorted(cuts, row["duration_ms"], side="right")) + offsets[4]
        return x

    return {
        "Xtr": encode(train_rows),
        "ytr": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "utr": np.asarray([row["user_id"] for row in train_rows]),
        "date": np.asarray([row["date"] for row in train_rows]),
        "Xva": encode(val_rows),
        "yva": np.asarray([row["long_view"] for row in val_rows], dtype=np.int64),
        "uva": np.asarray([row["user_id"] for row in val_rows]),
        "vva": np.asarray([row["video_id"] for row in val_rows]),
        "field_dims": field_dims,
        "evaluate": evaluate,
    }


def day_numbers(values):
    values = np.asarray(values)
    parsed = {}
    for value in np.unique(values):
        text = str(value)
        if text.endswith(".0"):
            text = text[:-2]
        try:
            parsed[value] = dt.datetime.strptime(text, "%Y%m%d").date().toordinal()
        except Exception:
            try:
                parsed[value] = int(float(text))
            except Exception:
                parsed[value] = 0
    return np.asarray([parsed[value] for value in values], dtype=np.float32)


def build_pairs(users, labels, seed):
    users = np.asarray(users)
    labels = np.asarray(labels)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.RandomState(seed)
    positives = []
    negatives = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group = order[start:end]
        pos = group[labels[group] > 0.5]
        neg = group[labels[group] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not positives:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, embed_dim=16, hidden=128, dropout=0.2):
        super().__init__()
        width = fields * embed_dim
        self.emb = torch.nn.Embedding(total_dim, embed_dim)
        self.linear = torch.nn.Embedding(total_dim, 1)
        self.emb_drop = torch.nn.Dropout(dropout)
        self.cross_w = torch.nn.Parameter(torch.empty(width))
        self.cross_b = torch.nn.Parameter(torch.zeros(width))
        self.deep = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.output = torch.nn.Linear(width + hidden // 2, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.normal_(self.cross_w, std=0.01)

    def forward(self, x):
        x0 = self.emb_drop(self.emb(x)).flatten(1)
        cross = x0 * (x0 * self.cross_w).sum(1, keepdim=True) + self.cross_b + x0
        deep = self.deep(x0)
        first = self.linear(x).sum((1, 2))
        return self.bias + first + self.output(torch.cat([cross, deep], dim=1)).squeeze(1)


def predict(model, x, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            outputs.append(model(xb).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def train_run(data, pair_pos, pair_neg, cfg, seed, epochs, fraction, device,
              checkpoints_per_epoch=1, keep_scores=False, eval_each_epoch=True):
    seed_all(seed)
    x = data["Xtr"]
    y = data["ytr"]
    n = len(y)
    model = DCNLite(
        int(data["field_dims"].sum()),
        embed_dim=16,
        hidden=128,
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.RandomState(seed + 17011)
    dates = day_numbers(data["date"])
    ages = float(dates.max()) - dates
    recency = np.power(0.5, ages / cfg["half_life"]).astype(np.float32)
    recency /= max(float(recency.mean()), 1e-8)
    batch_size = 16384 if device.type == "cuda" else 8192
    epoch_n = min(n, max(batch_size, int(round(n * fraction))))
    steps = int(math.ceil(epoch_n / batch_size))
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    curve = []

    for epoch in range(epochs):
        model.train()
        if epoch_n == n:
            indices = rng.permutation(n)
        else:
            indices = rng.choice(n, size=epoch_n, replace=False)
        checkpoint_steps = set()
        if eval_each_epoch or epoch == epochs - 1:
            for q in range(1, checkpoints_per_epoch + 1):
                checkpoint_steps.add(max(1, int(math.ceil(steps * q / checkpoints_per_epoch))))
        running_loss = 0.0
        seen = 0
        for step, start in enumerate(range(0, epoch_n, batch_size), 1):
            idx = indices[start:start + batch_size]
            batch_n = len(idx)
            if len(pair_pos):
                selected = rng.randint(0, len(pair_pos), size=batch_n)
                pos_idx = pair_pos[selected]
                neg_idx = pair_neg[selected]
            else:
                pos_idx = idx
                neg_idx = idx
            joined = np.concatenate([idx, pos_idx, neg_idx])
            logits = model(torch.from_numpy(x[joined]).to(device))
            point_logits = logits[:batch_n]
            pos_logits = logits[batch_n:2 * batch_n]
            neg_logits = logits[2 * batch_n:]
            target = torch.from_numpy(y[idx]).to(device)
            point_weight = torch.from_numpy(recency[idx]).to(device)
            pair_weight = torch.from_numpy(0.5 * (recency[pos_idx] + recency[neg_idx])).to(device)
            point_loss = (bce(point_logits, target) * point_weight).mean()
            pair_loss = (
                torch.nn.functional.softplus(-(pos_logits - neg_logits)) * pair_weight
            ).mean()
            loss = 0.5 * point_loss + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * batch_n
            seen += batch_n

            if step in checkpoint_steps:
                scores = predict(model, data["Xva"], device)
                metrics = metric_values(data["evaluate"](data["uva"], data["yva"], scores))
                curve.append({
                    "epoch": round(epoch + step / steps, 3),
                    "train_loss": round(running_loss / max(seen, 1), 6),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-12:
                    best_primary = metrics["primary"]
                    best_metrics = metrics
                    if keep_scores:
                        best_scores = scores.copy()
                model.train()

        if (epoch + 1) % cfg["step_size"] == 0:
            for group in optimizer.param_groups:
                group["lr"] *= cfg["gamma"]

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "primary": best_primary,
        "gauc": best_metrics["gauc"],
        "metrics": best_metrics,
        "scores": best_scores,
        "curve": curve,
    }


def coarse_configs(seed, count):
    rng = np.random.RandomState(seed + 913)
    configs = []
    half_lives = [3.5, 5.0, 7.0, 10.0, 14.0]
    gammas = [0.38, 0.50, 0.62, 0.74, 0.86]
    step_sizes = [1, 2, 3, 4]
    for i in range(count):
        dropout_stratum = ((i * 11) % count + rng.rand()) / count
        decay_stratum = ((i * 17) % count + rng.rand()) / count
        lr_stratum = ((i * 19) % count + rng.rand()) / count
        configs.append({
            "dropout": float(0.14 + 0.28 * dropout_stratum),
            "weight_decay": float(10 ** (-4.52 + 2.0 * decay_stratum)),
            "lr": float(10 ** (-3.42 + 0.72 * lr_stratum)),
            "gamma": float(gammas[rng.randint(len(gammas))]),
            "step_size": int(step_sizes[rng.randint(len(step_sizes))]),
            "half_life": float(half_lives[rng.randint(len(half_lives))]),
        })
    rng.shuffle(configs)
    return configs


def refined_configs(winner, seed, count):
    rng = np.random.RandomState(seed + 2719)
    configs = [dict(winner)]
    for _ in range(count - 1):
        configs.append({
            "dropout": float(np.clip(winner["dropout"] + rng.normal(0.0, 0.028), 0.10, 0.46)),
            "weight_decay": float(np.clip(
                winner["weight_decay"] * math.exp(rng.normal(0.0, 0.34)), 2e-5, 4e-3
            )),
            "lr": float(np.clip(winner["lr"] * math.exp(rng.normal(0.0, 0.16)), 3e-4, 2.2e-3)),
            "gamma": float(np.clip(winner["gamma"] + rng.normal(0.0, 0.055), 0.30, 0.92)),
            "step_size": int(np.clip(
                winner["step_size"] + rng.choice([-1, 0, 0, 0, 1]), 1, 4
            )),
            "half_life": float(np.clip(
                winner["half_life"] * math.exp(rng.normal(0.0, 0.17)), 3.0, 16.0
            )),
        })
    return configs


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def rank_average(score_list, users):
    users = np.asarray(users)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    result = np.zeros(len(users), dtype=np.float64)
    for scores in score_list:
        ranked = np.zeros(len(users), dtype=np.float64)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            idx = order[start:end]
            local_order = np.argsort(scores[idx], kind="mergesort")
            local_rank = np.empty(len(idx), dtype=np.float64)
            local_rank[local_order] = np.arange(len(idx), dtype=np.float64)
            if len(idx) > 1:
                local_rank /= len(idx) - 1
            ranked[idx] = local_rank
        result += ranked
    return result / len(score_list)


def record_result(history, progress_path, phase, candidate, cfg, result, epochs, fraction, seed):
    record = {
        "phase": phase,
        "candidate": candidate,
        "seed": seed,
        "epochs": epochs,
        "fraction": fraction,
        "config": cfg,
        "gauc": round(result["gauc"], 6),
        "primary": round(result["primary"], 6),
    }
    history.append(record)
    append_progress(progress_path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz")) and
        os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    data = load_npz(args.data_dir) if fast_path else load_csv(args.data_dir)
    pair_pos, pair_neg = build_pairs(data["utr"], data["ytr"], args.seed)
    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_value) if smoke_value is not None else None
    history = []

    if smoke_cap is not None:
        smoke_cfg = {
            "dropout": 0.24,
            "weight_decay": 2.5e-4,
            "lr": 8.0e-4,
            "gamma": 0.62,
            "step_size": 2,
            "half_life": 7.0,
        }
        final_epochs = max(1, min(args.epochs, smoke_cap))
        result = train_run(
            data, pair_pos, pair_neg, smoke_cfg, args.seed, final_epochs, 1.0,
            device, checkpoints_per_epoch=1, keep_scores=True, eval_each_epoch=True
        )
        record_result(
            history, progress_path, "smoke_final", 0, smoke_cfg, result,
            final_epochs, 1.0, args.seed
        )
        final_cfg = smoke_cfg
        member_scores = [result["scores"]]
        member_curves = [{"seed": args.seed, "curve": result["curve"]}]
    else:
        coarse = coarse_configs(args.seed, 24)
        survivors = list(enumerate(coarse))
        rung_specs = [
            ("coarse_rung1", 2, 0.35, 12, False),
            ("coarse_rung2", 3, 0.65, 6, False),
            ("coarse_full_fidelity", 5, 1.0, 3, True),
        ]
        for rung_index, (phase, epochs, fraction, keep_n, eval_each_epoch) in enumerate(rung_specs):
            scored = []
            for candidate_id, cfg in survivors:
                run_seed = args.seed + 1000 * (rung_index + 1) + candidate_id
                result = train_run(
                    data, pair_pos, pair_neg, cfg, run_seed, epochs, fraction,
                    device, checkpoints_per_epoch=1, keep_scores=False,
                    eval_each_epoch=eval_each_epoch
                )
                record_result(
                    history, progress_path, phase, candidate_id, cfg, result,
                    epochs, fraction, run_seed
                )
                scored.append((result["primary"], result["gauc"], candidate_id, cfg))
            scored.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
            survivors = [(candidate_id, cfg) for _, _, candidate_id, cfg in scored[:keep_n]]

        coarse_winner = survivors[0][1]
        refined = refined_configs(coarse_winner, args.seed, 10)
        refinement_scores = []
        for candidate_id, cfg in enumerate(refined):
            run_seed = args.seed + 5000 + candidate_id
            result = train_run(
                data, pair_pos, pair_neg, cfg, run_seed, 6, 1.0,
                device, checkpoints_per_epoch=1, keep_scores=False,
                eval_each_epoch=True
            )
            record_result(
                history, progress_path, "local_refinement", candidate_id, cfg,
                result, 6, 1.0, run_seed
            )
            refinement_scores.append((result["primary"], result["gauc"], candidate_id, cfg))
        refinement_scores.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
        final_cfg = refinement_scores[0][3]

        final_epochs = max(1, args.epochs)
        member_scores = []
        member_curves = []
        for member in range(5):
            member_seed = args.seed + member
            result = train_run(
                data, pair_pos, pair_neg, final_cfg, member_seed, final_epochs,
                1.0, device, checkpoints_per_epoch=2, keep_scores=True,
                eval_each_epoch=True
            )
            member_scores.append(result["scores"])
            member_curves.append({"seed": member_seed, "curve": result["curve"]})
            record_result(
                history, progress_path, "matched_seed_final", member, final_cfg,
                result, final_epochs, 1.0, member_seed
            )

    ensemble_trials = []
    best_scores = None
    best_metrics = None
    best_count = 1
    for count in range(1, len(member_scores) + 1):
        scores = rank_average(member_scores[:count], data["uva"])
        metrics = metric_values(data["evaluate"](data["uva"], data["yva"], scores))
        ensemble_trials.append({
            "members": count,
            "gauc": round(metrics["gauc"], 6),
            "ndcg5": round(metrics["ndcg5"], 6),
            "primary": round(metrics["primary"], 6),
        })
        if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
            best_metrics = metrics
            best_scores = scores.copy()
            best_count = count

    final_metrics = metric_values(
        data["evaluate"](data["uva"], data["yva"], best_scores)
    )
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "history": history,
        "winning_config": final_cfg,
        "selected_ensemble_members": best_count,
        "ensemble_trials": ensemble_trials,
        "final_learning_curves": member_curves,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, score in enumerate(best_scores):
            writer.writerow([i, data["uva"][i], data["vva"][i], format(float(score), ".9g")])


if __name__ == "__main__":
    main()
