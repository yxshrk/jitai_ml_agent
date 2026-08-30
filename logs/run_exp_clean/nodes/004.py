"""Validation-selected lower-dimensional FM with matched-seed rank ensembling."""
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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

    def forward(self, x):
        e = self.emb(x)
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def read_csv_rows(path):
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def build_csv_data(data_dir):
    train_rows = read_csv_rows(os.path.join(data_dir, "train.csv"))
    val_rows = read_csv_rows(os.path.join(data_dir, "val.csv"))

    def make_map(values):
        result = {"__UNK__": 0}
        for value in values:
            if value not in result:
                result[value] = len(result)
        return result

    user_map = make_map([r["user_id"] for r in train_rows])
    video_map = make_map([r["video_id"] for r in train_rows])
    tab_map = make_map([r["tab"] for r in train_rows])
    durations = np.asarray(
        [float(r["duration_ms"]) for r in train_rows], dtype=np.float64
    )
    if len(durations):
        cuts = np.unique(np.quantile(durations, np.linspace(0.1, 0.9, 9)))
    else:
        cuts = np.asarray([], dtype=np.float64)

    field_dims = np.asarray(
        [len(user_map), len(video_map), 1, len(tab_map), len(cuts) + 1],
        dtype=np.int64,
    )
    offsets = np.concatenate(([0], np.cumsum(field_dims)[:-1])).astype(np.int64)

    def encode(rows):
        x = np.zeros((len(rows), 5), dtype=np.int64)
        for i, row in enumerate(rows):
            x[i, 0] = user_map.get(row["user_id"], 0)
            x[i, 1] = video_map.get(row["video_id"], 0)
            x[i, 2] = 0
            x[i, 3] = tab_map.get(row["tab"], 0)
            x[i, 4] = int(
                np.searchsorted(cuts, float(row["duration_ms"]), side="right")
            )
        x += offsets
        return x

    train_x = encode(train_rows)
    val_x = encode(val_rows)
    train_y = np.asarray(
        [float(r["long_view"]) for r in train_rows], dtype=np.float32
    )
    val_y = np.asarray(
        [float(r["long_view"]) for r in val_rows], dtype=np.float32
    )
    val_users = np.asarray([r["user_id"] for r in val_rows])
    val_videos = np.asarray([r["video_id"] for r in val_rows])
    return train_x, train_y, val_x, val_y, val_users, val_videos, field_dims


def rank_within_user(users, scores):
    n = len(scores)
    result = np.empty(n, dtype=np.float32)
    order = np.lexsort((np.arange(n, dtype=np.int64), scores, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for start, end in zip(starts, ends):
        size = end - start
        if size == 1:
            result[order[start]] = 0.5
        else:
            result[order[start:end]] = (
                np.arange(size, dtype=np.float32) / float(size - 1)
            )
    return result


def metric_values(metrics):
    return {
        "gauc": float(metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"]),
        "ndcg5": float(metrics.get("nDCG@5", metrics.get("ndcg5"))),
        "primary": float(metrics["primary"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, "progress.log")
    if os.path.exists(progress_path):
        os.remove(progress_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    fast_path = os.path.exists(train_npz) and os.path.exists(val_npz)

    if fast_path:
        from data.official.evaluate import evaluate as evaluate_fn

        tr = np.load(train_npz)
        va = np.load(val_npz)
        train_x_np = tr["X"].astype(np.int64, copy=False)
        train_y_np = tr["y"].astype(np.float32, copy=False)
        val_x_np = va["X"].astype(np.int64, copy=False)
        val_y = va["y"].astype(np.int64, copy=False)
        val_users = np.asarray(va["user"])
        val_videos = np.zeros(len(val_y), dtype=np.int64)
        field_dims = np.asarray(tr["field_dims"], dtype=np.int64)
    else:
        from harness.evaluate_provisional import evaluate as evaluate_fn

        (
            train_x_np,
            train_y_np,
            val_x_np,
            val_y_float,
            val_users,
            val_videos,
            field_dims,
        ) = build_csv_data(args.data_dir)
        val_y = val_y_float.astype(np.int64)

    total_dim = int(field_dims.sum())
    train_x = torch.from_numpy(train_x_np)
    train_y = torch.from_numpy(train_y_np)
    val_x = torch.from_numpy(val_x_np)
    n_train = len(train_y)
    batch_size = 8192
    predict_batch_size = 65536

    smoke_raw = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_raw) if smoke_raw is not None else None
    epochs = args.epochs if smoke_epochs is None else min(
        args.epochs, max(1, smoke_epochs)
    )

    if smoke_epochs is not None:
        candidate_dims = [6, 12]
        probe_seed_count = 1
        final_member_count = 3
    else:
        candidate_dims = [3, 4, 6, 8, 10, 12]
        probe_seed_count = 24
        final_member_count = 96

    probe_epochs = min(epochs, 8)
    history = []

    def train_member(k, member_index, epoch_limit, phase):
        member_seed = args.seed + 1009 * member_index
        np.random.seed(member_seed)
        torch.manual_seed(member_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(member_seed)

        model = FM(total_dim, k=k).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        best_primary = -1.0
        best_scores = None
        best_metrics = None
        best_epoch = 0
        patience = 0
        curve = []

        for epoch in range(epoch_limit):
            model.train()
            permutation = torch.randperm(n_train)
            last_loss = 0.0
            for start in range(0, n_train, batch_size):
                index = permutation[start:start + batch_size]
                xb = train_x[index].to(
                    device, non_blocking=(device.type == "cuda")
                )
                yb = train_y[index].to(
                    device, non_blocking=(device.type == "cuda")
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

            model.eval()
            pieces = []
            with torch.no_grad():
                for start in range(0, len(val_x), predict_batch_size):
                    xb = val_x[start:start + predict_batch_size].to(
                        device, non_blocking=(device.type == "cuda")
                    )
                    pieces.append(model(xb).detach().cpu().numpy())
            scores = np.concatenate(pieces).astype(np.float32, copy=False)
            values = metric_values(evaluate_fn(val_users, val_y, scores))
            curve.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(last_loss, 5),
                    "val_gauc": round(values["gauc"], 6),
                    "val_ndcg5": round(values["ndcg5"], 6),
                    "val_primary": round(values["primary"], 6),
                }
            )
            if values["primary"] > best_primary + 1e-6:
                best_primary = values["primary"]
                best_scores = scores.copy()
                best_metrics = values
                best_epoch = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= 2:
                    break

        ranks = rank_within_user(val_users, best_scores)
        record = {
            "type": phase,
            "member": member_index,
            "seed": member_seed,
            "config": {
                "k": k,
                "lr": 0.001,
                "batch_size": batch_size,
                "epoch_limit": epoch_limit,
            },
            "best_epoch": best_epoch,
            "gauc": best_metrics["gauc"],
            "ndcg5": best_metrics["ndcg5"],
            "primary": best_metrics["primary"],
            "curve": curve,
        }
        history.append(record)
        with open(progress_path, "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "phase": phase,
                        "k": k,
                        "member": member_index,
                        "seed": member_seed,
                        "best_epoch": best_epoch,
                        "primary": best_primary,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        del model, optimizer, best_scores
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return ranks, best_primary

    probe_ranks = {}
    probe_primaries = {}
    dimension_results = []
    best_dimension = None
    best_dimension_primary = -1.0

    for k in candidate_dims:
        ranks_for_k = []
        primaries_for_k = []
        for member_index in range(probe_seed_count):
            ranks, primary = train_member(
                k, member_index, probe_epochs, "dimension_probe"
            )
            ranks_for_k.append(ranks)
            primaries_for_k.append(primary)
        probe_ranks[k] = ranks_for_k
        probe_primaries[k] = primaries_for_k
        ensemble_scores = np.mean(
            np.stack(ranks_for_k, axis=0), axis=0, dtype=np.float64
        ).astype(np.float32)
        values = metric_values(evaluate_fn(val_users, val_y, ensemble_scores))
        summary = {
            "type": "dimension_summary",
            "config": {
                "k": k,
                "matched_seed_count": probe_seed_count,
                "epoch_limit": probe_epochs,
            },
            "mean_individual_primary": float(np.mean(primaries_for_k)),
            "gauc": values["gauc"],
            "ndcg5": values["ndcg5"],
            "primary": values["primary"],
        }
        dimension_results.append(summary)
        history.append(summary)
        with open(progress_path, "a") as fh:
            fh.write(json.dumps(summary, sort_keys=True) + "\n")
        if values["primary"] > best_dimension_primary + 1e-8:
            best_dimension_primary = values["primary"]
            best_dimension = k

    member_ranks = list(probe_ranks[best_dimension])
    member_primary = list(probe_primaries[best_dimension])

    for member_index in range(probe_seed_count, final_member_count):
        ranks, primary = train_member(
            best_dimension, member_index, epochs, "final_seed_member"
        )
        member_ranks.append(ranks)
        member_primary.append(primary)

    rank_matrix = np.stack(member_ranks, axis=0)
    sorted_members = np.argsort(-np.asarray(member_primary), kind="stable")
    best_individual = float(np.max(member_primary))
    competitive = [
        int(i)
        for i in sorted_members
        if member_primary[int(i)] >= best_individual - 0.003
    ]
    if len(competitive) < 2:
        competitive = [
            int(i) for i in sorted_members[: min(2, final_member_count)]
        ]

    candidate_counts = {
        2,
        3,
        4,
        5,
        8,
        12,
        16,
        24,
        32,
        48,
        64,
        80,
        96,
        len(competitive),
    }
    candidate_counts = sorted(
        count for count in candidate_counts if 1 <= count <= len(competitive)
    )
    if not candidate_counts:
        candidate_counts = [len(competitive)]

    cumulative = np.zeros(len(val_y), dtype=np.float64)
    candidate_set = set(candidate_counts)
    best_ensemble_primary = -1.0
    final_scores = None
    final_members = None

    for position, member_index in enumerate(competitive, start=1):
        cumulative += rank_matrix[member_index]
        if position in candidate_set:
            ensemble_scores = (cumulative / float(position)).astype(np.float32)
            values = metric_values(evaluate_fn(val_users, val_y, ensemble_scores))
            record = {
                "type": "rank_ensemble",
                "k": best_dimension,
                "count": position,
                "members": competitive[:position],
                "gauc": values["gauc"],
                "ndcg5": values["ndcg5"],
                "primary": values["primary"],
            }
            history.append(record)
            if values["primary"] > best_ensemble_primary + 1e-8:
                best_ensemble_primary = values["primary"]
                final_scores = ensemble_scores.copy()
                final_members = competitive[:position]

    final_metrics = metric_values(evaluate_fn(val_users, val_y, final_scores))
    output_metrics = {
        "gauc": final_metrics["gauc"],
        "ndcg5": final_metrics["ndcg5"],
        "primary": final_metrics["primary"],
        "selected_embedding_dim": best_dimension,
        "dimension_probe_primary": best_dimension_primary,
        "selected_member_count": len(final_members),
        "selected_members": final_members,
        "history": history,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(output_metrics, fh)

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(final_scores):
            fh.write(
                f"{i},{val_users[i]},{val_videos[i]},{float(score):.7g}\n"
            )


if __name__ == "__main__":
    main()
