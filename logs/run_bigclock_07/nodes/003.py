"""Two-stage dial search for a regularized DCN-lite hybrid-ranking package."""
import argparse
import csv
import datetime
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
        self.fields = fields
        self.k = k
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
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    wanted = ["user_id", "video_id", "tab", "duration_ms", "date", "long_view"]

    def read(path):
        out = {k: [] for k in wanted}
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                for k in wanted:
                    out[k].append(row[k])
        return out

    tr = read(train_path)
    va = read(val_path)
    users = {v: i for i, v in enumerate(sorted(set(tr["user_id"])))}
    videos = {v: i for i, v in enumerate(sorted(set(tr["video_id"])))}
    tabs = {v: i for i, v in enumerate(sorted(set(tr["tab"])))}
    train_duration = np.asarray(tr["duration_ms"], dtype=np.float64)
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

    return make(tr), make(va), False


def load_data(data_dir):
    tp = os.path.join(data_dir, "train.npz")
    vp = os.path.join(data_dir, "val.npz")
    if not (os.path.exists(tp) and os.path.exists(vp)):
        return encode_csv(data_dir)
    trn = np.load(tp)
    val = np.load(vp)
    tr = {k: trn[k] for k in trn.files}
    va = {k: val[k] for k in val.files}
    offsets = np.concatenate(([0], np.cumsum(tr["field_dims"][:-1])))
    va["video"] = va["X"][:, 1].astype(np.int64) - int(offsets[1])
    return tr, va, True


def date_ordinals(values):
    arr = np.asarray(values).astype(str)
    unique, inverse = np.unique(arr, return_inverse=True)
    converted = []
    for text in unique:
        text = text.split(".")[0].replace("-", "")
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
    weight = np.exp2(-(float(day.max()) - day) / float(half_life)).astype(np.float32)
    return weight / max(float(weight.mean()), 1e-8)


def make_pair_pool(users, labels):
    users = np.asarray(users)
    labels = np.asarray(labels) > 0.5
    neg = np.flatnonzero(~labels)
    if len(neg) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(users[neg], kind="stable")
    neg_sorted = neg[order]
    neg_users = users[neg_sorted]
    uniq, starts, counts = np.unique(neg_users, return_index=True, return_counts=True)
    pos = np.flatnonzero(labels)
    loc = np.searchsorted(uniq, users[pos])
    valid = loc < len(uniq)
    exact = np.zeros(len(pos), dtype=bool)
    exact[valid] = uniq[loc[valid]] == users[pos[valid]]
    pos = pos[exact]
    loc = loc[exact]
    return pos.astype(np.int64), starts[loc].astype(np.int64), counts[loc].astype(np.int64), neg_sorted.astype(np.int64)


def evaluator(use_npz):
    if use_npz:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    return evaluate


def metric_values(result):
    return (float(result.get("GAUC", result.get("gauc"))),
            float(result.get("nDCG@5", result.get("ndcg5"))),
            float(result["primary"]))


def predict(model, Xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = Xv[start:start + 65536].to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks)


def train_variant(config, seed, epochs, X, y, users, dates, pair_data, Xv, vu, vy,
                  evaluate, device, fraction=1.0, half_checkpoints=False, keep_scores=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = DCNLite(int(config["total_dim"]), dropout=float(config["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    weights = recency_weights(dates, float(config["half_life"]))
    pos_pool, neg_starts, neg_counts, neg_sorted = pair_data
    n = len(y)
    subset_n = max(1, int(n * fraction))
    bs = int(config.get("batch_size", 16384))
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    curve = []

    def assess(mark, last_loss):
        nonlocal best_primary, best_metrics, best_scores
        scores = predict(model, Xv, device)
        result = evaluate(vu, vy, scores)
        gauc, ndcg, primary = metric_values(result)
        curve.append({"checkpoint": mark, "train_loss": round(float(last_loss), 6),
                      "val_gauc": round(gauc, 6), "val_ndcg5": round(ndcg, 6),
                      "val_primary": round(primary, 6)})
        if primary > best_primary + 1e-8:
            best_primary = primary
            best_metrics = (gauc, ndcg, primary)
            if keep_scores:
                best_scores = scores.copy()

    for epoch in range(epochs):
        model.train()
        if fraction < 0.999:
            perm = rng.choice(n, size=subset_n, replace=False)
            rng.shuffle(perm)
        else:
            perm = rng.permutation(n)
        split = (len(perm) + 1) // 2
        sections = [perm[:split], perm[split:]] if half_checkpoints else [perm]
        last_loss = 0.0
        for section_id, section in enumerate(sections):
            for start in range(0, len(section), bs):
                idx_np = section[start:start + bs]
                idx = torch.from_numpy(idx_np.astype(np.int64, copy=False))
                xb = X[idx].to(device)
                target = y[idx].to(device)
                sw = torch.from_numpy(weights[idx_np]).to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                point = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none")
                point_loss = (point * sw).sum() / sw.sum().clamp_min(1e-8)
                pair_count = min(max(256, len(idx_np) // 8), len(pos_pool))
                if pair_count > 0:
                    choose = rng.integers(0, len(pos_pool), size=pair_count)
                    pi_np = pos_pool[choose]
                    offset = (rng.random(pair_count) * neg_counts[choose]).astype(np.int64)
                    ni_np = neg_sorted[neg_starts[choose] + offset]
                    pi = torch.from_numpy(pi_np).to(device)
                    ni = torch.from_numpy(ni_np).to(device)
                    ps = model(X[pi].to(device))
                    ns = model(X[ni].to(device))
                    pw = torch.from_numpy((weights[pi_np] + weights[ni_np]) * 0.5).to(device)
                    pair_loss_raw = torch.nn.functional.softplus(-(ps - ns))
                    pair_loss = (pair_loss_raw * pw).sum() / pw.sum().clamp_min(1e-8)
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                last_loss = float(loss.detach().cpu())
            if half_checkpoints:
                assess("%d.%d" % (epoch + 1, 5 if section_id == 0 else 0), last_loss)
                model.train()
        if not half_checkpoints:
            assess(str(epoch + 1), last_loss)
        if (epoch + 1) % int(config["step_size"]) == 0:
            for group in opt.param_groups:
                group["lr"] *= float(config["gamma"])
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, curve


def clean_config(config):
    return {k: (round(float(v), 8) if isinstance(v, (float, np.floating)) else int(v))
            for k, v in config.items() if k not in ("total_dim", "batch_size")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

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
    evaluate = evaluator(use_npz)
    X = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    y = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    vy = np.asarray(va["y"], dtype=np.int64)
    vu = np.asarray(va["user"])
    dates = np.asarray(tr["date"])
    users = np.asarray(tr["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    pair_data = make_pair_pool(users, np.asarray(tr["y"]))

    smoke = os.environ.get("SMOKE_EPOCHS")
    smoke_cap = int(smoke) if smoke is not None else None
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
        drop, wd, gamma, step, half, lr = spec
        cfg = {"dropout": drop, "weight_decay": wd, "gamma": gamma,
               "step_size": step, "half_life": half, "lr": lr,
               "total_dim": total_dim, "batch_size": 32768}
        metrics, _, curve = train_variant(
            cfg, args.seed + 100, coarse_epochs, X, y, users, dates, pair_data,
            Xv, vu, vy, evaluate, device, fraction=0.22,
            half_checkpoints=False, keep_scores=False)
        coarse_results.append((metrics[2], cfg))
        history.append({"stage": "coarse", "probe": probe_id + 1,
                        "config": clean_config(cfg), "best_gauc": round(metrics[0], 6),
                        "best_ndcg5": round(metrics[1], 6),
                        "best_primary": round(metrics[2], 6), "curve": curve})

    coarse_results.sort(key=lambda z: z[0], reverse=True)
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
        dd, wm, dg, dh, lm = spec
        cfg = dict(center)
        cfg["dropout"] = float(np.clip(center["dropout"] + dd, 0.12, 0.45))
        cfg["weight_decay"] = float(np.clip(center["weight_decay"] * wm, 2e-5, 5e-3))
        cfg["gamma"] = float(np.clip(center["gamma"] + dg, 0.35, 0.85))
        cfg["half_life"] = float(np.clip(center["half_life"] + dh, 2.5, 20.0))
        cfg["lr"] = float(np.clip(center["lr"] * lm, 0.00035, 0.0013))
        cfg["batch_size"] = 32768
        metrics, _, curve = train_variant(
            cfg, args.seed + 200, refine_epochs, X, y, users, dates, pair_data,
            Xv, vu, vy, evaluate, device, fraction=1.0,
            half_checkpoints=False, keep_scores=False)
        refine_results.append((metrics[2], cfg))
        history.append({"stage": "refine", "probe": probe_id + 1,
                        "config": clean_config(cfg), "best_gauc": round(metrics[0], 6),
                        "best_ndcg5": round(metrics[1], 6),
                        "best_primary": round(metrics[2], 6), "curve": curve})

    refine_results.sort(key=lambda z: z[0], reverse=True)
    winner = dict(refine_results[0][1])
    winner["batch_size"] = 16384
    final_metrics, best_scores, final_curve = train_variant(
        winner, args.seed, final_epochs, X, y, users, dates, pair_data,
        Xv, vu, vy, evaluate, device, fraction=1.0,
        half_checkpoints=True, keep_scores=True)
    history.append({"stage": "final", "config": clean_config(winner),
                    "best_gauc": round(final_metrics[0], 6),
                    "best_ndcg5": round(final_metrics[1], 6),
                    "best_primary": round(final_metrics[2], 6), "curve": final_curve})

    result = evaluate(vu, vy, best_scores)
    gauc, ndcg, primary = metric_values(result)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg, "primary": primary,
                   "selected_config": clean_config(winner), "history": history}, fh)
    videos = np.asarray(va["video"])
    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write("%d,%s,%s,%.9g\n" % (i, str(vu[i]), str(videos[i]), float(score)))


if __name__ == "__main__":
    main()
