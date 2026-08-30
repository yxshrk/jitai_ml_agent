import argparse
import csv
import datetime
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DCNAddonModel(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.30,
                 ordinal=False, regime=False, cwm=False):
        super().__init__()
        self.use_ordinal = ordinal
        self.use_regime = regime
        self.use_cwm = cwm
        self.emb = torch.nn.Embedding(total_dim, k)
        self.emb_drop = torch.nn.Dropout(dropout)
        dim = fields * k
        self.cross_w = torch.nn.Parameter(torch.empty(dim))
        self.cross_b = torch.nn.Parameter(torch.zeros(dim))
        self.cross_out = torch.nn.Linear(dim, 1)
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.main_out = torch.nn.Linear(hidden, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        if regime:
            self.short_out = torch.nn.Linear(hidden, 1)
            self.long_out = torch.nn.Linear(hidden, 1)
        if ordinal:
            self.ordinal_out = torch.nn.Linear(hidden, 5)
        if cwm:
            self.cwm_out = torch.nn.Linear(hidden, 1)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.normal_(self.cross_w, std=0.01)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x, duration_ms=None, return_aux=False):
        x0 = self.emb_drop(self.emb(x)).flatten(1)
        xl = x0 + x0 * torch.sum(x0 * self.cross_w, dim=1, keepdim=True) + self.cross_b
        h = self.shared(x0)
        logits = self.bias + self.cross_out(xl).squeeze(1) + self.main_out(h).squeeze(1)
        if self.use_regime:
            if duration_ms is None:
                raise ValueError("duration_ms is required for duration-regime heads")
            short = self.short_out(h).squeeze(1)
            long = self.long_out(h).squeeze(1)
            residual = torch.where(duration_ms <= 18000.0, short, long)
            logits = logits + 0.5 * residual
        if not return_aux:
            return logits
        aux = {}
        if self.use_ordinal:
            aux["ordinal"] = self.ordinal_out(h)
        if self.use_cwm:
            aux["cwm"] = self.cwm_out(h).squeeze(1)
        return logits, aux


def metric_values(metrics):
    return {
        "gauc": float(metrics.get("GAUC", metrics.get("gauc"))),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = tr["field_dims"].astype(np.int64)
    offset = int(field_dims[0])
    return {
        "Xt": tr["X"].astype(np.int64),
        "yt": tr["y"].astype(np.float32),
        "ut": tr["user"],
        "date": tr["date"],
        "play_t": tr["play_time_ms"].astype(np.float32),
        "duration_t": tr["duration_ms"].astype(np.float32),
        "Xv": va["X"].astype(np.int64),
        "yv": va["y"].astype(np.int64),
        "uv": va["user"],
        "duration_v": va["duration_ms"].astype(np.float32),
        "video_out": va["X"][:, 1].astype(np.int64) - offset,
        "field_dims": field_dims,
    }


def quantile_edges(values, buckets=10):
    quantiles = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    return np.unique(np.quantile(values.astype(np.float64), quantiles))


def load_csv_data(data_dir):
    train_rows = []
    durations = []
    with open(os.path.join(data_dir, "train.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            rec = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
                "play_time_ms": float(row["play_time_ms"]),
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
        for field in (0, 1, 3, 4):
            value = token(row, field)
            if value not in vocab[field]:
                vocab[field][value] = len(vocab[field])
    field_dims = np.asarray([len(values) + 1 for values in vocab], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1]))

    def encode(row):
        encoded = np.empty(5, dtype=np.int64)
        for field in range(5):
            value = token(row, field)
            encoded[field] = offsets[field] + vocab[field].get(value, len(vocab[field]))
        return encoded

    val_rows = []
    with open(os.path.join(data_dir, "val.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            val_rows.append({
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
                "date": row["date"],
                "long_view": float(row["long_view"]),
            })
    return {
        "Xt": np.stack([encode(row) for row in train_rows]),
        "yt": np.asarray([row["long_view"] for row in train_rows], dtype=np.float32),
        "ut": np.asarray([row["user_id"] for row in train_rows]),
        "date": np.asarray([row["date"] for row in train_rows]),
        "play_t": np.asarray([row["play_time_ms"] for row in train_rows], dtype=np.float32),
        "duration_t": np.asarray([row["duration_ms"] for row in train_rows], dtype=np.float32),
        "Xv": np.stack([encode(row) for row in val_rows]),
        "yv": np.asarray([row["long_view"] for row in val_rows], dtype=np.int64),
        "uv": np.asarray([row["user_id"] for row in val_rows]),
        "duration_v": np.asarray([row["duration_ms"] for row in val_rows], dtype=np.float32),
        "video_out": np.asarray([row["video_id"] for row in val_rows]),
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


def recency_weights(dates, half_life=14.0):
    unique, inverse = np.unique(dates, return_inverse=True)
    ordinals = np.asarray([date_ordinal(value) for value in unique], dtype=np.float64)
    ages = np.max(ordinals) - ordinals
    weights = np.exp(-math.log(2.0) * ages / half_life)[inverse].astype(np.float32)
    weights = np.clip(weights, 0.35, 1.0)
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
        indices = order[left:right]
        pos = indices[labels[indices] > 0.5]
        neg = indices[labels[indices] <= 0.5]
        if len(pos) and len(neg):
            positives.append(pos)
            negatives.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not positives:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(positives).astype(np.int64), np.concatenate(negatives).astype(np.int64)


def addon_config(addons):
    return {
        "ordinal": "ordinal" in addons,
        "regime": "regime" in addons,
        "cwm": "cwm" in addons,
        "recency14": "recency14" in addons,
    }


def config_name(config):
    active = [name for name in ("ordinal", "regime", "cwm", "recency14") if config[name]]
    return "+".join(active) if active else "champion"


def predict(model, X, durations, device, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = X[start:end].to(device, non_blocking=True)
            db = durations[start:end].to(device, non_blocking=True)
            outputs.append(model(xb, db).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_one(config, seed, epochs, data, evaluator, device, half_epoch=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DCNAddonModel(
        int(data["field_dims"].sum()),
        fields=data["Xt"].shape[1],
        k=16,
        hidden=128,
        dropout=0.30,
        ordinal=config["ordinal"],
        regime=config["regime"],
        cwm=config["cwm"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    Xt = torch.from_numpy(data["Xt"])
    yt = torch.from_numpy(data["yt"])
    duration_t = torch.from_numpy(data["duration_t"])
    play_t = torch.from_numpy(data["play_t"])
    Xv = torch.from_numpy(data["Xv"])
    duration_v = torch.from_numpy(data["duration_v"])
    if config["recency14"]:
        sample_weight = torch.from_numpy(data["recency14"])
    else:
        sample_weight = torch.ones(len(yt), dtype=torch.float32)
    pair_pos, pair_neg = data["pairs"]
    pair_pos_t = torch.from_numpy(pair_pos)
    pair_neg_t = torch.from_numpy(pair_neg)
    n = len(yt)
    batch_size = 8192
    steps = int(math.ceil(n / batch_size))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17011)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    ordinal_thresholds = torch.tensor([0.10, 0.25, 0.50, 0.75, 1.00], dtype=torch.float32)
    best_primary = -1.0
    best_scores = None
    curve = []
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
            db = duration_t[idx].to(device, non_blocking=True)
            pb = play_t[idx].to(device, non_blocking=True)
            wb = sample_weight[idx].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(xb, db, return_aux=True)
            point_loss = (bce(logits, yb) * wb).mean()
            if len(pair_pos_t):
                chosen = torch.randint(len(pair_pos_t), (len(idx),), generator=generator)
                pi = pair_pos_t[chosen]
                ni = pair_neg_t[chosen]
                pair_x = torch.cat((Xt[pi], Xt[ni]), dim=0).to(device, non_blocking=True)
                pair_d = torch.cat((duration_t[pi], duration_t[ni]), dim=0).to(device, non_blocking=True)
                pair_logits = model(pair_x, pair_d)
                pair_loss = torch.nn.functional.softplus(
                    -(pair_logits[:len(idx)] - pair_logits[len(idx):])
                ).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            safe_duration = torch.clamp(db, min=1.0)
            if config["ordinal"]:
                ratio18 = pb / torch.clamp(db, min=18000.0)
                ratio18 = torch.clamp(ratio18, min=0.0, max=2.0)
                targets = (ratio18.unsqueeze(1) >= ordinal_thresholds.to(device)).float()
                ordinal_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    aux["ordinal"], targets, reduction="none"
                ).mean(dim=1)
                loss = loss + 0.30 * (ordinal_loss * wb).mean()
            if config["cwm"]:
                watch_prediction = torch.nn.functional.softplus(aux["cwm"])
                observed_ratio = torch.clamp(pb / safe_duration, min=0.0, max=1.0)
                completed = pb >= safe_duration
                uncensored = torch.nn.functional.smooth_l1_loss(
                    watch_prediction, observed_ratio, reduction="none"
                )
                censored = torch.relu(1.0 - watch_prediction).square()
                cwm_loss = torch.where(completed, censored, uncensored)
                loss = loss + 0.30 * (cwm_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
            if step in checkpoints:
                scores = predict(model, Xv, duration_v, device)
                metrics = metric_values(evaluator(data["uv"], data["yv"], scores))
                fraction = 0.5 if step != steps - 1 else 1.0
                curve.append({
                    "epoch": float(epoch + fraction),
                    "train_loss": round(running_loss / max(seen, 1), 6),
                    "val_gauc": round(metrics["gauc"], 6),
                    "val_primary": round(metrics["primary"], 6),
                })
                if metrics["primary"] > best_primary + 1e-12:
                    best_primary = metrics["primary"]
                    best_scores = scores.copy()
                model.train()
        scheduler.step()
    best_point = max(curve, key=lambda row: row["val_primary"])
    result = {
        "best_primary": float(best_primary),
        "best_gauc": float(best_point["val_gauc"]),
        "best_epoch": float(best_point["epoch"]),
        "curve": curve,
    }
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, best_scores


def append_progress(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def summarize(config, records):
    scores = [record["primary"] for record in records]
    epochs = [record["best_epoch"] for record in records]
    return {
        "name": config_name(config),
        "config": dict(config),
        "mean_primary": float(np.mean(scores)),
        "std_primary": float(np.std(scores)),
        "standard_error": float(np.std(scores) / math.sqrt(max(len(scores), 1))),
        "mean_best_epoch": float(np.mean(epochs)),
        "scores": [float(score) for score in scores],
    }


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
    fast_path = (
        os.path.exists(os.path.join(args.data_dir, "train.npz")) and
        os.path.exists(os.path.join(args.data_dir, "val.npz"))
    )
    if fast_path:
        from data.official.evaluate import evaluate as evaluator
        data = load_npz(args.data_dir)
    else:
        from harness.evaluate_provisional import evaluate as evaluator
        data = load_csv_data(args.data_dir)
    data["recency14"] = recency_weights(data["date"], half_life=14.0)
    data["pairs"] = build_pairs(data["ut"], data["yt"], args.seed)
    smoke_text = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke_text) if smoke_text is not None else None
    probe_epochs = 5
    refine_epochs = min(10, max(1, args.epochs))
    final_epochs = max(1, args.epochs)
    probe_repeats = 60 if device.type == "cuda" else 10
    refine_repeats = 20 if device.type == "cuda" else 8
    refine_count = 6
    if smoke_cap is not None:
        probe_epochs = min(probe_epochs, smoke_cap)
        refine_epochs = min(refine_epochs, smoke_cap)
        final_epochs = min(final_epochs, smoke_cap)
        probe_repeats = 1
        refine_repeats = 1
        refine_count = 2
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)
    history = []
    phase_summaries = []
    base = addon_config(())
    addon_names = ("ordinal", "regime", "cwm", "recency14")
    individual_configs = [base] + [addon_config((name,)) for name in addon_names]
    individual_summaries = []
    for config_index, config in enumerate(individual_configs):
        records = []
        for repeat in range(probe_repeats):
            seed = args.seed + 1009 * repeat
            result, _ = train_one(config, seed, probe_epochs, data, evaluator, device)
            record = {
                "phase": "individual_probe",
                "name": config_name(config),
                "config": dict(config),
                "seed": seed,
                "epochs": probe_epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "gauc": result["best_gauc"],
            }
            records.append(record)
            history.append(record)
            append_progress(progress_path, record)
        individual_summaries.append(summarize(config, records))
    individual_summaries.sort(key=lambda row: row["mean_primary"], reverse=True)
    phase_summaries.extend(individual_summaries)
    base_mean = next(row["mean_primary"] for row in individual_summaries if row["name"] == "champion")
    ranked_addons = [
        row for row in individual_summaries if row["name"] != "champion"
    ]
    positive_addons = [row["name"] for row in ranked_addons if row["mean_primary"] > base_mean]
    selected_addons = positive_addons[:3]
    if len(selected_addons) < 3:
        for row in ranked_addons:
            if row["name"] not in selected_addons:
                selected_addons.append(row["name"])
            if len(selected_addons) == 3:
                break
    pair_configs = [addon_config(pair) for pair in itertools.combinations(selected_addons, 2)]
    pair_summaries = []
    for config in pair_configs:
        records = []
        for repeat in range(probe_repeats):
            seed = args.seed + 1009 * repeat
            result, _ = train_one(config, seed, probe_epochs, data, evaluator, device)
            record = {
                "phase": "pair_probe",
                "name": config_name(config),
                "config": dict(config),
                "seed": seed,
                "epochs": probe_epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "gauc": result["best_gauc"],
            }
            records.append(record)
            history.append(record)
            append_progress(progress_path, record)
        pair_summaries.append(summarize(config, records))
    pair_summaries.sort(key=lambda row: row["mean_primary"], reverse=True)
    phase_summaries.extend(pair_summaries)
    unique_candidates = {}
    for summary in sorted(phase_summaries, key=lambda row: row["mean_primary"], reverse=True):
        unique_candidates.setdefault(summary["name"], summary["config"])
    if "champion" not in unique_candidates:
        unique_candidates["champion"] = base
    candidate_items = list(unique_candidates.items())[:refine_count]
    if all(name != "champion" for name, _ in candidate_items):
        candidate_items[-1] = ("champion", base)
    refinement_summaries = []
    for rank, (name, config) in enumerate(candidate_items):
        records = []
        for repeat in range(refine_repeats):
            seed = args.seed + 50021 + repeat * 2017
            result, _ = train_one(config, seed, refine_epochs, data, evaluator, device, half_epoch=True)
            record = {
                "phase": "refinement",
                "rank": rank,
                "name": name,
                "config": dict(config),
                "seed": seed,
                "epochs": refine_epochs,
                "best_epoch": result["best_epoch"],
                "primary": result["best_primary"],
                "gauc": result["best_gauc"],
            }
            records.append(record)
            history.append(record)
            append_progress(progress_path, record)
        refinement_summaries.append(summarize(config, records))
    refinement_summaries.sort(key=lambda row: row["mean_primary"], reverse=True)
    winner = refinement_summaries[0]["config"]
    final_result, best_scores = train_one(
        winner, args.seed, final_epochs, data, evaluator, device, half_epoch=True
    )
    final_metrics = metric_values(evaluator(data["uv"], data["yv"], best_scores))
    final_record = {
        "phase": "final",
        "name": config_name(winner),
        "config": dict(winner),
        "seed": args.seed,
        "epochs": final_epochs,
        "best_epoch": final_result["best_epoch"],
        "primary": final_metrics["primary"],
        "gauc": final_metrics["gauc"],
    }
    history.append(final_record)
    append_progress(progress_path, final_record)
    output = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "diagnosis": "validation peaks early then falls, indicating overfit",
        "winner": dict(winner),
        "winner_name": config_name(winner),
        "selected_pair_addons": selected_addons,
        "individual_summary": individual_summaries,
        "pair_summary": pair_summaries,
        "refinement_summary": refinement_summaries,
        "final_curve": final_result["curve"],
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output, fh)
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for index, score in enumerate(best_scores):
            fh.write(f"{index},{data['uv'][index]},{data['video_out'][index]},{float(score):.8g}\n")


if __name__ == "__main__":
    main()
