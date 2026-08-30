import argparse
import csv
import datetime
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_offsets(field_dims):
    field_dims = np.asarray(field_dims, dtype=np.int64)
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(field_dims[:-1])])


def load_npz(data_dir):
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=False)
    va = np.load(os.path.join(data_dir, "val.npz"), allow_pickle=False)
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    offsets = make_offsets(field_dims)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    dates = np.asarray(tr["date"])
    if "video" in va.files:
        videos = np.asarray(va["video"])
    else:
        videos = xva[:, 1] - offsets[1]
    row_ids = np.arange(len(xva), dtype=np.int64)
    return xtr, ytr, utr, dates, xva, yva, uva, videos, row_ids, field_dims, True


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            item = {
                "row_id": r.get("row_id", str(i)),
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "author_id": r.get("author_id", "__missing_author__"),
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"]),
                "long_view": float(r["long_view"]),
            }
            if training:
                item["date"] = r["date"]
            rows.append(item)
    return rows


def build_map(values):
    unique = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(unique)}


def duration_edges(values):
    q = np.linspace(0.1, 0.9, 9)
    return np.unique(np.quantile(np.asarray(values, dtype=np.float64), q))


def load_csv_data(data_dir):
    tr = read_csv_rows(os.path.join(data_dir, "train.csv"), True)
    va = read_csv_rows(os.path.join(data_dir, "val.csv"), False)
    maps = [
        build_map([r["user_id"] for r in tr]),
        build_map([r["video_id"] for r in tr]),
        build_map([r["author_id"] for r in tr]),
        build_map([r["tab"] for r in tr]),
    ]
    edges = duration_edges([r["duration_ms"] for r in tr])
    field_dims = np.asarray([len(m) + 1 for m in maps] + [len(edges) + 1], dtype=np.int64)
    offsets = make_offsets(field_dims)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            vals = [r["user_id"], r["video_id"], r["author_id"], r["tab"]]
            for j in range(4):
                x[i, j] = maps[j].get(vals[j], 0) + offsets[j]
            x[i, 4] = int(np.searchsorted(edges, r["duration_ms"], side="right")) + offsets[4]
        return x

    xtr = encode(tr)
    xva = encode(va)
    ytr = np.asarray([r["long_view"] for r in tr], dtype=np.float32)
    yva = np.asarray([r["long_view"] for r in va], dtype=np.float32)
    utr = np.asarray([r["user_id"] for r in tr])
    uva = np.asarray([r["user_id"] for r in va])
    videos = np.asarray([r["video_id"] for r in va])
    dates = np.asarray([r["date"] for r in tr])
    row_ids = np.asarray([r["row_id"] for r in va])
    return xtr, ytr, utr, dates, xva, yva, uva, videos, row_ids, field_dims, False


def recency_weights(dates):
    text = np.asarray(dates).astype(str)
    unique = np.unique(text)
    ordinal = {}
    for d in unique:
        s = d.replace("-", "")[:8]
        try:
            ordinal[d] = datetime.datetime.strptime(s, "%Y%m%d").date().toordinal()
        except ValueError:
            ordinal[d] = 0
    vals = np.asarray([ordinal[d] for d in text], dtype=np.float32)
    latest = float(vals.max()) if len(vals) else 0.0
    w = np.exp2(-(latest - vals) / 7.0).astype(np.float32)
    return w / max(float(w.mean()), 1e-6)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dropout=0.30):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.linear = nn.Linear(width, 1)
        nn.init.normal_(self.embedding.weight, std=0.01)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

    def forward(self, x):
        x0 = self.embed_dropout(self.embedding(x).flatten(1))
        z = x0
        for w, b in zip(self.cross_w, self.cross_b):
            z = x0 * torch.sum(z * w, dim=1, keepdim=True) + b + z
        return (self.linear(z) + self.mlp(z)).squeeze(1)


def pair_groups(users, labels):
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    groups = []
    for a, b in zip(starts, ends):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if len(pos) and len(neg):
            groups.append((pos, neg))
    return groups


def sample_pairs(groups, rng, cap):
    pos_parts = []
    neg_parts = []
    for pos, neg in groups:
        n = min(max(len(pos), len(neg)), 32)
        pos_parts.append(pos[rng.integers(0, len(pos), size=n)])
        neg_parts.append(neg[rng.integers(0, len(neg), size=n)])
    if not pos_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    p = np.concatenate(pos_parts)
    n = np.concatenate(neg_parts)
    if len(p) > cap:
        keep = rng.choice(len(p), size=cap, replace=False)
        p, n = p[keep], n[keep]
    perm = rng.permutation(len(p))
    return p[perm], n[perm]


def predict(model, x, device, batch_size=16384):
    model.eval()
    out = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(x), batch_size):
            b = min(a + batch_size, len(x))
            xb = torch.from_numpy(x[a:b]).to(device)
            out[a:b] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def metric_values(result):
    g = result.get("GAUC", result.get("gauc"))
    n = result.get("nDCG@5", result.get("ndcg5"))
    p = result.get("primary")
    return float(g), float(n), float(p)


def average_states(states):
    keys = states[0].keys()
    averaged = {}
    for k in keys:
        tensors = [s[k] for s in states]
        if tensors[0].is_floating_point():
            averaged[k] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            averaged[k] = tensors[-1].clone()
    return averaged


def train_member(member_seed, xtr, ytr, users, weights, xva, yva, val_users, field_dims, evaluator, epochs, device):
    seed_all(member_seed)
    rng = np.random.default_rng(member_seed)
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    groups = pair_groups(users, ytr)
    checkpoints = []
    gaucs = []
    batch_size = 8192
    n = len(xtr)

    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        pair_pos, pair_neg = sample_pairs(groups, rng, min(n, 400000))
        pair_cursor = 0
        for a in range(0, n, batch_size):
            idx = order[a:min(a + batch_size, n)]
            xb = torch.from_numpy(xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            wb = torch.from_numpy(weights[idx]).to(device)
            logits = model(xb)
            point_loss = (F.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).mean()
            if len(pair_pos):
                count = min(len(idx), len(pair_pos))
                if pair_cursor + count > len(pair_pos):
                    pair_cursor = 0
                pi = pair_pos[pair_cursor:pair_cursor + count]
                ni = pair_neg[pair_cursor:pair_cursor + count]
                pair_cursor += count
                px = torch.from_numpy(xtr[pi]).to(device)
                nx = torch.from_numpy(xtr[ni]).to(device)
                pair_loss = F.softplus(-(model(px) - model(nx))).mean()
                loss = 0.5 * point_loss + 0.5 * pair_loss
            else:
                loss = point_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_scores = predict(model, xva, device)
        result = evaluator(val_users, yva, val_scores)
        gauc, _, _ = metric_values(result)
        gaucs.append(gauc)
        checkpoints.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    best = int(np.argmax(np.asarray(gaucs)))
    selected = [checkpoints[best]]
    if best > 0 and gaucs[best] - gaucs[best - 1] <= 0.0015:
        selected.insert(0, checkpoints[best - 1])
    model.load_state_dict(average_states(selected))
    model.to(device)
    return predict(model, xva, device)


def rank_average(scores_list, users):
    total = np.zeros(len(users), dtype=np.float64)
    order = np.argsort(users, kind="mergesort")
    sorted_users = users[order]
    cuts = np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    for scores in scores_list:
        ranked = np.empty(len(scores), dtype=np.float64)
        for a, b in zip(starts, ends):
            idx = order[a:b]
            if len(idx) == 1:
                ranked[idx] = 0.5
            else:
                local = np.argsort(scores[idx], kind="mergesort")
                values = np.empty(len(idx), dtype=np.float64)
                values[local] = np.arange(len(idx), dtype=np.float64) / float(len(idx) - 1)
                ranked[idx] = values
        total += ranked
    return (total / len(scores_list)).astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.data_dir, "train.npz")
    use_npz = os.path.exists(npz_path) and os.path.exists(os.path.join(args.data_dir, "val.npz"))
    if use_npz:
        data = load_npz(args.data_dir)
        from data.official.evaluate import evaluate as evaluator
    else:
        data = load_csv_data(args.data_dir)
        from harness.evaluate_provisional import evaluate as evaluator
    xtr, ytr, utr, dates, xva, yva, uva, videos, row_ids, field_dims, _ = data
    weights = recency_weights(dates)
    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = 6
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_scores = []
    for i in range(5):
        member_scores.append(train_member(args.seed + i, xtr, ytr, utr, weights, xva, yva, uva, field_dims, evaluator, epochs, device))
    scores = rank_average(member_scores, uva)
    result = evaluator(uva, yva, scores)
    gauc, ndcg5, primary = metric_values(result)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for rid, uid, vid, score in zip(row_ids, uva, videos, scores):
            writer.writerow([rid, uid, vid, format(float(score), ".9g")])

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
