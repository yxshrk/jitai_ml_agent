"""Train the BigClock-07 node-006 champion and build the label-free test submission."""

import csv
import datetime
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_PATH = ROOT / "logs/run_bigclock_07/workspace/train.npz"
VAL_PATH = ROOT / "logs/run_bigclock_07/workspace/val.npz"
TEST_PATH = ROOT / "data/test_features/test.npz"
OUTPUT_PATH = ROOT / "evidence/test_submission_pure.csv"
EXPECTED_TEST_ROWS = 170588


class DCNLite(torch.nn.Module):
    def __init__(self, total_dim, fields=5, k=16, hidden=128, dropout=0.25):
        super().__init__()
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


def metric_values(result):
    return (float(result.get("GAUC", result.get("gauc"))),
            float(result.get("nDCG@5", result.get("ndcg5"))),
            float(result["primary"]))


def date_ordinals(values):
    arr = np.asarray(values).astype(str)
    unique, inverse = np.unique(arr, return_inverse=True)
    converted = []
    for value in unique:
        text = value.split(".")[0].replace("-", "")
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
    weights = np.exp2(-(float(day.max()) - day) / float(half_life)).astype(np.float32)
    return weights / max(float(weights.mean()), 1e-8)


def make_pair_pool(users, labels):
    users = np.asarray(users)
    positive_mask = np.asarray(labels) > 0.5
    negatives = np.flatnonzero(~positive_mask)
    if len(negatives) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty, empty
    order = np.argsort(users[negatives], kind="stable")
    negative_sorted = negatives[order]
    negative_users = users[negative_sorted]
    unique, starts, counts = np.unique(negative_users, return_index=True, return_counts=True)
    positives = np.flatnonzero(positive_mask)
    locations = np.searchsorted(unique, users[positives])
    valid = locations < len(unique)
    exact = np.zeros(len(positives), dtype=bool)
    exact[valid] = unique[locations[valid]] == users[positives[valid]]
    positives = positives[exact]
    locations = locations[exact]
    return (positives.astype(np.int64), starts[locations].astype(np.int64),
            counts[locations].astype(np.int64), negative_sorted.astype(np.int64))


def predict_logits(model, Xv, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(Xv), 65536):
            xb = Xv[start:start + 65536].to(device)
            chunks.append(model(xb).detach().cpu().numpy())
    return np.concatenate(chunks)


def train_variant(config, seed, epochs, X, y, dates, pair_data, Xv, vu, vy,
                  evaluate, device, Xtest, fraction=1.0, half_checkpoints=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = DCNLite(int(config["total_dim"]), dropout=float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]))
    weights = recency_weights(dates, float(config["half_life"]))
    pos_pool, neg_starts, neg_counts, neg_sorted = pair_data
    n = len(y)
    subset_n = max(1, int(n * fraction))
    batch_size = int(config.get("batch_size", 16384))
    best_primary = -1.0
    best_metrics = None
    best_scores = None
    best_state = None
    curve = []

    def assess(checkpoint, last_loss):
        nonlocal best_primary, best_metrics, best_scores, best_state
        scores = predict_logits(model, Xv, device)
        metrics = metric_values(evaluate(vu, vy, scores))
        curve.append({
            "checkpoint": checkpoint,
            "train_loss": round(float(last_loss), 6),
            "val_gauc": round(metrics[0], 6),
            "val_ndcg5": round(metrics[1], 6),
            "val_primary": round(metrics[2], 6),
        })
        if metrics[2] > best_primary + 1e-8:
            best_primary = metrics[2]
            best_metrics = metrics
            best_scores = scores.copy()
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        if fraction < 0.999:
            permutation = rng.choice(n, size=subset_n, replace=False)
            rng.shuffle(permutation)
        else:
            permutation = rng.permutation(n)
        split = (len(permutation) + 1) // 2
        sections = [permutation[:split], permutation[split:]] if half_checkpoints else [permutation]
        last_loss = 0.0
        for section_id, section in enumerate(sections):
            for start in range(0, len(section), batch_size):
                idx_np = section[start:start + batch_size]
                idx = torch.from_numpy(idx_np.astype(np.int64, copy=False))
                xb = X[idx].to(device)
                target = y[idx].to(device)
                sample_weight = torch.from_numpy(weights[idx_np]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                point_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none")
                point_loss = ((point_raw * sample_weight).sum() /
                              sample_weight.sum().clamp_min(1e-8))
                pair_count = min(max(256, len(idx_np) // 8), len(pos_pool))
                if pair_count > 0:
                    choices = rng.integers(0, len(pos_pool), size=pair_count)
                    positive_np = pos_pool[choices]
                    offsets = (rng.random(pair_count) * neg_counts[choices]).astype(np.int64)
                    negative_np = neg_sorted[neg_starts[choices] + offsets]
                    positive_idx = torch.from_numpy(positive_np).to(device)
                    negative_idx = torch.from_numpy(negative_np).to(device)
                    positive_score = model(X[positive_idx].to(device))
                    negative_score = model(X[negative_idx].to(device))
                    pair_weight = torch.from_numpy(
                        (weights[positive_np] + weights[negative_np]) * 0.5).to(device)
                    pair_raw = torch.nn.functional.softplus(-(positive_score - negative_score))
                    pair_loss = ((pair_raw * pair_weight).sum() /
                                 pair_weight.sum().clamp_min(1e-8))
                    loss = 0.5 * point_loss + 0.5 * pair_loss
                else:
                    loss = point_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            if half_checkpoints:
                assess("%d.%d" % (epoch + 1, 5 if section_id == 0 else 0), last_loss)
                model.train()
        if not half_checkpoints:
            assess(str(epoch + 1), last_loss)
        if (epoch + 1) % int(config["step_size"]) == 0:
            for group in optimizer.param_groups:
                group["lr"] *= float(config["gamma"])

    if best_state is None:
        raise RuntimeError("training produced no validation checkpoint")
    model.load_state_dict(best_state)
    test_scores = predict_logits(model, Xtest, device)
    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_metrics, best_scores, test_scores, curve


def per_user_ranks(scores, users):
    scores = np.asarray(scores)
    users = np.asarray(users)
    output = np.empty(len(scores), dtype=np.float32)
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1, len(order)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        local_order = np.argsort(scores[indices], kind="stable")
        ranks = np.empty(len(indices), dtype=np.float32)
        ranks[local_order] = np.arange(len(indices), dtype=np.float32)
        if len(indices) > 1:
            ranks /= float(len(indices) - 1)
        else:
            ranks.fill(0.5)
        output[indices] = ranks
    return output


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def resembles_label_key(key):
    normalized = key.lower().strip()
    tokens = normalized.replace("-", "_").split("_")
    return ("label" in normalized or "y" in tokens or
            normalized in {"target", "targets", "long_view"})


def load_label_free_test(path):
    with np.load(path, allow_pickle=False) as archive:
        suspicious = [key for key in archive.files if resembles_label_key(key)]
        assert not suspicious, f"test archive contains label-like keys: {suspicious}"
        required = {"X", "user_id", "video_id"}
        missing = required.difference(archive.files)
        assert not missing, f"test archive missing required keys: {sorted(missing)}"
        # Only these explicitly label-free arrays are read from the hidden-test archive.
        Xtest = np.asarray(archive["X"], dtype=np.int64)
        users = np.asarray(archive["user_id"])
        videos = np.asarray(archive["video_id"])
    assert Xtest.ndim == 2 and Xtest.shape[1] == 5
    assert len(Xtest) == len(users) == len(videos) == EXPECTED_TEST_ROWS
    return Xtest, users, videos


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = load_npz(TRAIN_PATH)
    va = load_npz(VAL_PATH)
    Xtest_np, test_users, test_videos = load_label_free_test(TEST_PATH)

    from data.official.evaluate import evaluate

    X = torch.from_numpy(np.asarray(tr["X"], dtype=np.int64))
    y = torch.from_numpy(np.asarray(tr["y"], dtype=np.float32))
    Xv = torch.from_numpy(np.asarray(va["X"], dtype=np.int64))
    Xtest = torch.from_numpy(Xtest_np)
    vy = np.asarray(va["y"], dtype=np.int64)
    vu = np.asarray(va["user"])
    dates = np.asarray(tr["date"])
    train_users = np.asarray(tr["user"])
    total_dim = int(np.asarray(tr["field_dims"]).sum())
    pair_data = make_pair_pool(train_users, np.asarray(tr["y"]))

    config = {
        "dropout": 0.18,
        "weight_decay": 9e-05,
        "gamma": 0.57,
        "step_size": 2,
        "half_life": 7.0,
        "lr": 0.001,
        "total_dim": total_dim,
        "batch_size": 16384,
    }

    validation_ranks = []
    test_ranks = []
    for member_seed in (42, 43, 44):
        metrics, validation_scores, test_scores, _ = train_variant(
            config, member_seed, 12, X, y, dates, pair_data,
            Xv, vu, vy, evaluate, device, Xtest, half_checkpoints=True)
        print(f"seed {member_seed} validation: GAUC={metrics[0]:.9f} "
              f"nDCG@5={metrics[1]:.9f} primary={metrics[2]:.9f}", flush=True)
        validation_ranks.append(per_user_ranks(validation_scores, vu))
        test_ranks.append(per_user_ranks(test_scores, test_users))

    ensemble_validation = np.stack(validation_ranks).mean(axis=0)
    ensemble_test = np.stack(test_ranks).mean(axis=0)
    ensemble_metrics = metric_values(evaluate(vu, vy, ensemble_validation))
    print(f"ensemble validation: GAUC={ensemble_metrics[0]:.9f} "
          f"nDCG@5={ensemble_metrics[1]:.9f} primary={ensemble_metrics[2]:.9f}",
          flush=True)

    with open(OUTPUT_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (user_id, video_id, score) in enumerate(
                zip(test_users, test_videos, ensemble_test)):
            writer.writerow([row_id, int(user_id), int(video_id), f"{float(score):.9g}"])
    print(f"wrote {OUTPUT_PATH}: {len(ensemble_test)} rows", flush=True)

    command = [sys.executable, str(ROOT / "evidence/submission.py"), "--check",
               str(OUTPUT_PATH), str(TEST_PATH)]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print("submission checker stdout:", flush=True)
    print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print("submission checker stderr:", file=sys.stderr, flush=True)
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    completed.check_returncode()


if __name__ == "__main__":
    main()
