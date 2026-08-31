"""FM with frequency-adaptive embedding regularization.

The architecture, inputs, optimizer, batch size, and validation checkpointing match
node_000. A full-length paired dial search varies only the strength and frequency
exponent of embedding regularization, using identical initialization and minibatch
orders for every candidate.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x, return_embeddings=False):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        logits = self.bias + self.lin(x).sum((1, 2)) + pair
        if return_embeddings:
            return logits, e
        return logits


def _read_csv(path, need_label):
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            item = {
                "user_id": row["user_id"],
                "video_id": row["video_id"],
                "author_id": row.get("author_id", row["video_id"]),
                "tab": row["tab"],
                "duration_ms": float(row["duration_ms"]),
            }
            if need_label:
                item["long_view"] = float(row["long_view"])
            rows.append(item)
    return rows


def _build_csv_arrays(data_dir):
    train_rows = _read_csv(os.path.join(data_dir, "train.csv"), True)
    val_rows = _read_csv(os.path.join(data_dir, "val.csv"), True)
    train_duration = np.asarray([r["duration_ms"] for r in train_rows], dtype=np.float64)
    quantiles = np.quantile(train_duration, np.linspace(0.1, 0.9, 9))

    def duration_bucket(value):
        return str(int(np.searchsorted(quantiles, value, side="right")))

    fields = ["user_id", "video_id", "author_id", "tab"]
    mappings = []
    field_dims = []
    for field in fields:
        values = sorted({r[field] for r in train_rows})
        mapping = {value: i for i, value in enumerate(values)}
        mappings.append(mapping)
        field_dims.append(len(mapping) + 1)
    field_dims.append(10)
    field_dims = np.asarray(field_dims, dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))

    def encode(rows):
        x = np.empty((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, field in enumerate(fields):
                mapping = mappings[j]
                x[i, j] = offsets[j] + mapping.get(row[field], len(mapping))
            x[i, 4] = offsets[4] + int(duration_bucket(row["duration_ms"]))
        return x

    train = {
        "X": encode(train_rows),
        "y": np.asarray([r["long_view"] for r in train_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in train_rows]),
        "video": np.asarray([r["video_id"] for r in train_rows]),
        "field_dims": field_dims,
    }
    val = {
        "X": encode(val_rows),
        "y": np.asarray([r["long_view"] for r in val_rows], dtype=np.float32),
        "user": np.asarray([r["user_id"] for r in val_rows]),
        "video": np.asarray([r["video_id"] for r in val_rows]),
        "field_dims": field_dims,
    }
    return train, val, False


def _load_data(data_dir):
    train_npz = os.path.join(data_dir, "train.npz")
    val_npz = os.path.join(data_dir, "val.npz")
    if os.path.exists(train_npz) and os.path.exists(val_npz):
        tr_file = np.load(train_npz)
        va_file = np.load(val_npz)
        field_dims = tr_file["field_dims"].astype(np.int64)
        video_offset = int(field_dims[0])
        train = {
            "X": tr_file["X"].astype(np.int64),
            "y": tr_file["y"].astype(np.float32),
            "user": np.asarray(tr_file["user"]),
            "video": tr_file["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
        }
        val = {
            "X": va_file["X"].astype(np.int64),
            "y": va_file["y"].astype(np.float32),
            "user": np.asarray(va_file["user"]),
            "video": va_file["X"][:, 1].astype(np.int64) - video_offset,
            "field_dims": field_dims,
        }
        return train, val, True
    return _build_csv_arrays(data_dir)


def _make_evaluator(fast_path):
    if fast_path:
        from data.official.evaluate import evaluate
        return evaluate
    from harness.evaluate_provisional import evaluate
    return evaluate


def _frequency_weights(x, field_dims, alpha):
    total_dim = int(field_dims.sum())
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(field_dims)[:-1]))
    weights = np.ones(total_dim, dtype=np.float32)
    for field in range(3):
        start = int(offsets[field])
        end = start + int(field_dims[field])
        local = x[:, field] - start
        counts = np.bincount(local, minlength=int(field_dims[field])).astype(np.float64)
        present = counts > 0
        if not np.any(present):
            continue
        reference = float(np.median(counts[present]))
        raw = np.ones_like(counts)
        raw[present] = np.power(reference / counts[present], alpha)
        raw = np.clip(raw, 0.2, 8.0)
        occurrence_mean = float(np.sum(raw[present] * counts[present]) / np.sum(counts[present]))
        raw /= max(occurrence_mean, 1e-12)
        weights[start:end] = raw.astype(np.float32)
    return weights


def _metric_values(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def _train_candidate(x_train, y_train, x_val, val_user, val_y, field_dims,
                     alpha, reg_lambda, epochs, seed, device, evaluate):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = FM(int(field_dims.sum()), k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()
    reg_weights = torch.from_numpy(_frequency_weights(x_train.cpu().numpy(), field_dims, alpha)).to(device)
    n = len(y_train)
    batch_size = 8192
    best_primary = -1.0
    best_scores = None
    patience = 0
    epoch_history = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            index = permutation[start:start + batch_size]
            xb = x_train[index]
            optimizer.zero_grad(set_to_none=True)
            logits, embeddings = model(xb, return_embeddings=True)
            adaptive = reg_weights[xb]
            penalty = (adaptive.unsqueeze(-1) * embeddings.square()).sum(2).mean()
            loss = bce(logits, y_train[index]) + reg_lambda * penalty
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        model.eval()
        score_parts = []
        with torch.no_grad():
            for start in range(0, len(x_val), 65536):
                score_parts.append(model(x_val[start:start + 65536]).detach().cpu().numpy())
        scores = np.concatenate(score_parts)
        metrics = evaluate(val_user, val_y.astype(int), scores)
        values = _metric_values(metrics)
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(last_loss, 6),
            "val_gauc": round(values["gauc"], 6),
            "val_primary": round(values["primary"], 6),
        })
        if values["primary"] > best_primary + 1e-6:
            best_primary = values["primary"]
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    final_metrics = _metric_values(evaluate(val_user, val_y.astype(int), best_scores))
    return best_scores, final_metrics, epoch_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train, val, fast_path = _load_data(args.data_dir)
    evaluate = _make_evaluator(fast_path)
    epochs = args.epochs
    smoke = os.environ.get("SMOKE_EPOCHS")
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    x_train = torch.from_numpy(train["X"]).to(device)
    y_train = torch.from_numpy(train["y"]).to(device)
    x_val = torch.from_numpy(val["X"]).to(device)
    field_dims = train["field_dims"]

    alphas = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    lambdas = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    candidates = [(alpha, reg_lambda) for alpha in alphas for reg_lambda in lambdas]
    if smoke is not None:
        candidates = [(1.0, 0.03)]

    history = []
    best_scores = None
    best_metrics = None
    best_config = None
    best_epoch_history = None

    with open(progress_path, "w") as progress:
        for alpha, reg_lambda in candidates:
            scores, metrics, epoch_history = _train_candidate(
                x_train, y_train, x_val, val["user"], val["y"], field_dims,
                alpha, reg_lambda, epochs, args.seed, device, evaluate
            )
            record = {
                "config": {"frequency_alpha": alpha, "embedding_reg_lambda": reg_lambda},
                "seed": args.seed,
                "epochs_run": len(epoch_history),
                "gauc": metrics["gauc"],
                "ndcg5": metrics["ndcg5"],
                "primary": metrics["primary"],
                "epochs": epoch_history,
            }
            history.append(record)
            progress.write(json.dumps({
                "frequency_alpha": alpha,
                "embedding_reg_lambda": reg_lambda,
                "primary": metrics["primary"],
            }, sort_keys=True) + "\n")
            progress.flush()
            if best_metrics is None or metrics["primary"] > best_metrics["primary"] + 1e-12:
                best_scores = scores
                best_metrics = metrics
                best_config = record["config"]
                best_epoch_history = epoch_history

    output_metrics = {
        "gauc": best_metrics["gauc"],
        "ndcg5": best_metrics["ndcg5"],
        "primary": best_metrics["primary"],
        "best_config": best_config,
        "best_epoch_history": best_epoch_history,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{val['user'][i]},{val['video'][i]},{float(score):.9g}\n")


if __name__ == "__main__":
    main()
