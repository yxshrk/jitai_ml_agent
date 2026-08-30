import argparse
import csv
import json
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_npz(data_dir):
    tr = np.load(Path(data_dir) / "train.npz", allow_pickle=False)
    va = np.load(Path(data_dir) / "val.npz", allow_pickle=False)
    xtr = np.asarray(tr["X"], dtype=np.int64)
    xva = np.asarray(va["X"], dtype=np.int64)
    ytr = np.asarray(tr["y"], dtype=np.float32)
    yva = np.asarray(va["y"], dtype=np.float32)
    utr = np.asarray(tr["user"])
    uva = np.asarray(va["user"])
    field_dims = np.asarray(tr["field_dims"], dtype=np.int64).reshape(-1)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    video = xva[:, 1] - offsets[1]
    return xtr, ytr, utr, xva, yva, uva, video, field_dims


def read_csv_rows(path, training):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            item = {
                "user_id": r["user_id"],
                "video_id": r["video_id"],
                "tab": r["tab"],
                "duration_ms": float(r["duration_ms"] or 0.0),
                "long_view": float(r["long_view"]),
            }
            rows.append(item)
    return rows


def make_mapping(values):
    uniq = sorted(set(values))
    return {v: i + 1 for i, v in enumerate(uniq)}


def load_csv(data_dir):
    train = read_csv_rows(Path(data_dir) / "train.csv", True)
    val = read_csv_rows(Path(data_dir) / "val.csv", False)
    user_map = make_mapping([r["user_id"] for r in train])
    video_map = make_mapping([r["video_id"] for r in train])
    tab_map = make_mapping([r["tab"] for r in train])
    durations = np.asarray([r["duration_ms"] for r in train], dtype=np.float64)
    edges = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    field_dims = np.asarray([
        len(user_map) + 1,
        len(video_map) + 1,
        1,
        len(tab_map) + 1,
        len(edges) + 2,
    ], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, 0] = user_map.get(r["user_id"], 0)
            x[i, 1] = video_map.get(r["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(r["tab"], 0)
            x[i, 4] = int(np.searchsorted(edges, r["duration_ms"], side="right")) + 1
        x += offsets[None, :]
        return x

    xtr = encode(train)
    xva = encode(val)
    ytr = np.asarray([r["long_view"] for r in train], dtype=np.float32)
    yva = np.asarray([r["long_view"] for r in val], dtype=np.float32)
    utr = np.asarray([r["user_id"] for r in train])
    uva = np.asarray([r["user_id"] for r in val])
    video = np.asarray([r["video_id"] for r in val])
    return xtr, ytr, utr, xva, yva, uva, video, field_dims


def build_pairs(users, labels, seed):
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_users[1:] != sorted_users[:-1], True])
    rng = np.random.default_rng(seed)
    positives = []
    negatives = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        idx = order[a:b]
        pos = idx[labels[idx] > 0.5]
        neg = idx[labels[idx] <= 0.5]
        if pos.size and neg.size:
            positives.append(pos)
            negatives.append(rng.choice(neg, size=pos.size, replace=True))
    if not positives:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(positives), np.concatenate(negatives)


class DCNLite(nn.Module):
    def __init__(self, field_dims, embed_dim=16):
        super().__init__()
        total = int(np.sum(field_dims))
        width = len(field_dims) * embed_dim
        self.embedding = nn.Embedding(total, embed_dim)
        self.linear_embedding = nn.Embedding(total, 1)
        self.embed_dropout = nn.Dropout(0.15)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.empty(width)) for _ in range(2)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(width)) for _ in range(2)])
        self.mlp = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )
        self.cross_out = nn.Linear(width, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear_embedding.weight)
        for w in self.cross_w:
            nn.init.normal_(w, std=0.01)

    def forward(self, x):
        emb = self.embed_dropout(self.embedding(x)).flatten(1)
        cross = emb
        for w, b in zip(self.cross_w, self.cross_b):
            cross = emb * torch.sum(cross * w, dim=1, keepdim=True) + b + cross
        linear = self.linear_embedding(x).sum(dim=1).squeeze(1)
        return linear + self.cross_out(cross).squeeze(1) + self.mlp(emb).squeeze(1) + self.bias


def copy_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def add_to_swa(swa, state, count):
    if swa is None:
        return {k: v.detach().clone() for k, v in state.items()}
    alpha = 1.0 / float(count + 1)
    for k in swa:
        if torch.is_floating_point(swa[k]):
            swa[k].mul_(1.0 - alpha).add_(state[k], alpha=alpha)
        else:
            swa[k].copy_(state[k])
    return swa


def predict(model, x, device, batch_size):
    model.eval()
    result = np.empty(len(x), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.long, device=device)
            result[start:end] = torch.sigmoid(model(xb)).cpu().numpy()
    return result


def main():
    args = parse_args()
    seed_all(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fast = (data_dir / "train.npz").exists() and (data_dir / "val.npz").exists()
    if fast:
        xtr, ytr, utr, xva, yva, uva, video, field_dims = load_npz(data_dir)
    else:
        xtr, ytr, utr, xva, yva, uva, video, field_dims = load_csv(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNLite(field_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    pos_idx, neg_idx = build_pairs(utr, ytr, args.seed)
    rng = np.random.default_rng(args.seed)
    batch_size = 8192 if device.type == "cuda" else 4096

    smoke = os.environ.get("SMOKE_EPOCHS")
    max_epochs = 6.5
    if smoke is not None:
        max_epochs = min(max_epochs, float(max(1, int(smoke))))
    half_steps = int(round(max_epochs * 2.0))
    swa_state = None
    swa_count = 0

    for half in range(1, half_steps + 1):
        model.train()
        take = (len(xtr) + 1) // 2
        impression_order = rng.permutation(len(xtr))[:take]
        if len(pos_idx):
            pair_order = rng.permutation(len(pos_idx))
        else:
            pair_order = np.empty(0, dtype=np.int64)
        pair_cursor = 0

        for start in range(0, take, batch_size):
            ids = impression_order[start:min(start + batch_size, take)]
            xb = torch.as_tensor(xtr[ids], dtype=torch.long, device=device)
            yb = torch.as_tensor(ytr[ids], dtype=torch.float32, device=device)
            logits = model(xb)
            point_loss = bce(logits, yb)

            pair_loss = torch.zeros((), device=device)
            if pair_order.size:
                need = min(len(ids), pair_order.size)
                if pair_cursor + need > pair_order.size:
                    pair_order = rng.permutation(len(pos_idx))
                    pair_cursor = 0
                psel = pair_order[pair_cursor:pair_cursor + need]
                pair_cursor += need
                px = torch.as_tensor(xtr[pos_idx[psel]], dtype=torch.long, device=device)
                nx = torch.as_tensor(xtr[neg_idx[psel]], dtype=torch.long, device=device)
                pair_loss = torch.nn.functional.softplus(-(model(px) - model(nx))).mean()

            loss = 0.5 * point_loss + 0.5 * pair_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        epoch_value = half / 2.0
        in_window = 3.5 <= epoch_value <= 4.5
        if max_epochs < 3.5:
            in_window = half == half_steps
        if in_window:
            swa_state = add_to_swa(swa_state, copy_state(model), swa_count)
            swa_count += 1

    if swa_state is None:
        swa_state = copy_state(model)
    model.load_state_dict(swa_state)
    scores = predict(model, xva, device, batch_size)

    pred_path = out_dir / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for i, (u, v, s) in enumerate(zip(uva, video, scores)):
            writer.writerow([i, u.item() if hasattr(u, "item") else u, v.item() if hasattr(v, "item") else v, float(s)])

    if fast:
        from data.official.evaluate import evaluate
    else:
        from harness.evaluate_provisional import evaluate
    measured = evaluate(uva, yva, scores)
    metrics = {
        "gauc": float(measured["GAUC"]),
        "ndcg5": float(measured["nDCG@5"]),
        "primary": float(measured["primary"]),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
