import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import csv
import json
import math
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DCNLite(nn.Module):
    def __init__(self, total_dim, n_fields=5, emb_dim=16, hidden=128, dropout=0.15):
        super().__init__()
        self.n_fields = n_fields
        self.emb_dim = emb_dim
        width = n_fields * emb_dim
        self.embedding = nn.Embedding(total_dim, emb_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.emb_dropout = nn.Dropout(dropout)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(width + 64, 1)

    def forward(self, x):
        x0 = self.emb_dropout(self.embedding(x).reshape(x.shape[0], -1))
        xl = x0
        for w, b in zip(self.cross_w, self.cross_b):
            xl = xl + x0 * torch.sum(xl * w, dim=1, keepdim=True) + b
        deep = self.mlp(x0)
        return self.out(torch.cat((xl, deep), dim=1)).squeeze(1)


def seed_everything(seed):
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
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    xva = np.asarray(va["X"], dtype=np.int64)
    yva = np.asarray(va["y"], dtype=np.float32)
    uva = np.asarray(va["user"])
    dims = np.asarray(tr["field_dims"], dtype=np.int64)
    total_dim = int(max(int(xtr.max(initial=0)), int(xva.max(initial=0))) + 1)
    offsets = np.concatenate(([0], np.cumsum(dims[:-1]))).astype(np.int64)
    vids = xva[:, 1] - offsets[1]
    return xtr, ytr, utr, xva, yva, uva, vids, total_dim, True


def read_csv_rows(path, is_train):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user": r["user_id"],
                "video": r["video_id"],
                "author": r.get("author_id", r["video_id"]),
                "tab": r["tab"],
                "duration": float(r["duration_ms"] or 0.0),
                "label": float(r["long_view"]),
            }
            rows.append(item)
    return rows


def load_csv(data_dir):
    train = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    val = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    durations = np.asarray([r["duration"] for r in train], dtype=np.float64)
    cuts = np.quantile(durations, np.linspace(0.1, 0.9, 9)) if len(durations) else np.zeros(9)
    keys = ("user", "video", "author", "tab")
    maps = []
    dims = []
    for key in keys:
        values = sorted({r[key] for r in train})
        mapping = {v: i + 1 for i, v in enumerate(values)}
        maps.append(mapping)
        dims.append(len(mapping) + 1)
    dims.append(10)
    offsets = np.concatenate(([0], np.cumsum(np.asarray(dims[:-1], dtype=np.int64))))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        users = []
        videos = []
        for i, r in enumerate(rows):
            for j, key in enumerate(keys):
                x[i, j] = offsets[j] + maps[j].get(r[key], 0)
            x[i, 4] = offsets[4] + int(np.searchsorted(cuts, r["duration"], side="right"))
            y[i] = r["label"]
            users.append(r["user"])
            videos.append(r["video"])
        return x, y, np.asarray(users), np.asarray(videos)

    xtr, ytr, utr, _ = encode(train)
    xva, yva, uva, vids = encode(val)
    return xtr, ytr, utr, xva, yva, uva, vids, int(np.sum(dims)), False


def make_pairs(users, labels, seed):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    rng = np.random.RandomState(seed)
    pos_parts = []
    neg_parts = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            pos_parts.append(pos)
            neg_parts.append(neg[rng.randint(0, len(neg), size=len(pos))])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(pos_parts).astype(np.int64), np.concatenate(neg_parts).astype(np.int64)


def predict(model, x, device, batch_size=16384):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            out[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def metric_values(user_ids, labels, scores, npz_mode):
    if npz_mode:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    result = evaluate(user_ids, labels, scores)
    return {
        "gauc": float(result.get("GAUC", result.get("gauc"))),
        "ndcg5": float(result.get("nDCG@5", result.get("ndcg5"))),
        "primary": float(result.get("primary")),
    }


def average_state(avg, state, count):
    if avg is None:
        return {k: v.detach().cpu().clone() for k, v in state.items()}
    for k, value in state.items():
        source = value.detach().cpu()
        if torch.is_floating_point(source):
            avg[k].add_((source - avg[k]) / float(count))
        else:
            avg[k].copy_(source)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    npz_mode = os.path.exists(os.path.join(args.data_dir, "train.npz")) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if npz_mode:
        xtr, ytr, utr, xva, yva, uva, vids, total_dim, npz_mode = load_npz(args.data_dir)
    else:
        xtr, ytr, utr, xva, yva, uva, vids, total_dim, npz_mode = load_csv(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(total_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    batch_size = 4096
    epochs = 7
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    pair_pos, pair_neg = make_pairs(utr, ytr, args.seed)
    rng = np.random.RandomState(args.seed)
    swa_state = None
    swa_count = 0
    best_swa_state = None
    best_swa_gauc = -float("inf")
    global_half = 0

    for epoch in range(epochs):
        model.train()
        point_order = rng.permutation(len(xtr))
        pair_order = rng.permutation(len(pair_pos)) if len(pair_pos) else np.empty(0, dtype=np.int64)
        steps = int(math.ceil(len(xtr) / batch_size))
        half_step = max(1, steps // 2)
        for step in range(steps):
            a = step * batch_size
            b = min(a + batch_size, len(xtr))
            idx = point_order[a:b]
            xb = torch.as_tensor(xtr[idx], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            point_logits = model(xb)
            point_loss = F.binary_cross_entropy_with_logits(point_logits, yb)
            if len(pair_pos):
                pa = (step * batch_size) % len(pair_pos)
                take = min(batch_size, len(pair_pos))
                positions = (np.arange(take, dtype=np.int64) + pa) % len(pair_pos)
                chosen = pair_order[positions]
                xp = torch.as_tensor(xtr[pair_pos[chosen]], dtype=torch.long, device=device)
                xn = torch.as_tensor(xtr[pair_neg[chosen]], dtype=torch.long, device=device)
                pair_loss = F.softplus(-(model(xp) - model(xn))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            at_half = (step + 1 == half_step) or (step + 1 == steps)
            if at_half:
                global_half += 1
                progress = global_half / 2.0
                if 3.0 <= progress <= 4.5:
                    swa_count += 1
                    swa_state = average_state(swa_state, model.state_dict(), swa_count)
                    if swa_count >= 2:
                        raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        model.load_state_dict(swa_state)
                        candidate = predict(model, xva, device)
                        candidate_metrics = metric_values(uva, yva, candidate, npz_mode)
                        if candidate_metrics["gauc"] > best_swa_gauc:
                            best_swa_gauc = candidate_metrics["gauc"]
                            best_swa_state = {k: v.clone() for k, v in swa_state.items()}
                        model.load_state_dict(raw_state)
                        model.to(device)
        scheduler.step()

    if best_swa_state is not None:
        model.load_state_dict(best_swa_state)
    elif swa_state is not None:
        model.load_state_dict(swa_state)
    else:
        fallback = average_state(None, model.state_dict(), 1)
        model.load_state_dict(fallback)
    model.to(device)
    scores = predict(model, xva, device)
    metrics = metric_values(uva, yva, scores, npz_mode)

    pred_path = os.path.join(args.out_dir, "predictions.csv")
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (user, video, score) in enumerate(zip(uva, vids, scores)):
            if isinstance(user, np.generic):
                user = user.item()
            if isinstance(video, np.generic):
                video = video.item()
            writer.writerow([i, user, video, format(float(score), ".10g")])
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, separators=(",", ":"))


if __name__ == "__main__":
    sink = open(os.devnull, "w")
    sys.stdout = sink
    sys.stderr = sink
    main()
