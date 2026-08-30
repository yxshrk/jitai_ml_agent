"""FM baseline with a CWM-style censored watch-time auxiliary objective.

Uses the five offset-encoded fields from the NPZ fast path. The primary head is
trained on long_view, while a shared-representation auxiliary FM head predicts
log truncated watch time. Plays reaching the video duration are treated as
right-censored lower bounds and incur loss only when underpredicted.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class CensoredWatchFM(torch.nn.Module):
    def __init__(self, total_dim, k=16):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.watch_lin = torch.nn.Embedding(total_dim, 1)
        self.watch_pair_weight = torch.nn.Parameter(torch.zeros(k))
        self.watch_bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)
        torch.nn.init.zeros_(self.watch_lin.weight)

    def forward(self, x):
        e = self.emb(x)
        summed = e.sum(1)
        pair_vector = 0.5 * (summed * summed - (e * e).sum(1))
        main = self.bias + self.lin(x).sum((1, 2)) + pair_vector.sum(1)
        watch = (
            self.watch_bias
            + self.watch_lin(x).sum((1, 2))
            + (pair_vector * self.watch_pair_weight).sum(1)
        )
        return main, watch


def censored_watch_loss(prediction, target, censored):
    residual = torch.where(
        censored,
        torch.relu(target - prediction),
        prediction - target,
    )
    absolute = residual.abs()
    per_row = torch.where(
        absolute < 1.0,
        0.5 * residual * residual,
        absolute - 0.5,
    )
    return per_row.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = np.load(os.path.join(args.data_dir, "train.npz"))
    val = np.load(os.path.join(args.data_dir, "val.npz"))

    total_dim = int(train["field_dims"].sum())
    train_x = torch.from_numpy(train["X"].astype(np.int64))
    train_y = torch.from_numpy(train["y"].astype(np.float32))
    val_x = torch.from_numpy(val["X"].astype(np.int64))

    play_ms = train["play_time_ms"].astype(np.float32)
    duration_ms = train["duration_ms"].astype(np.float32)
    valid_duration = np.maximum(duration_ms, 1.0)
    observed_ms = np.minimum(np.maximum(play_ms, 0.0), valid_duration)
    watch_target = np.log1p(observed_ms / 1000.0).astype(np.float32)
    completed = ((duration_ms > 0.0) & (play_ms >= duration_ms)).astype(np.bool_)
    train_watch = torch.from_numpy(watch_target)
    train_censored = torch.from_numpy(completed)

    model = CensoredWatchFM(total_dim, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    binary_loss = torch.nn.BCEWithLogitsLoss()

    n = len(train_y)
    batch_size = 8192
    auxiliary_weight = 0.3
    best_primary = -1.0
    best_scores = None
    patience = 0

    for _ in range(args.epochs):
        model.train()
        permutation = torch.randperm(n)
        for start in range(0, n, batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad()
            logits, watch_prediction = model(train_x[indices])
            main_loss = binary_loss(logits, train_y[indices])
            auxiliary_loss = censored_watch_loss(
                watch_prediction,
                train_watch[indices],
                train_censored[indices],
            )
            loss = main_loss + auxiliary_weight * auxiliary_loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            score_parts = []
            for start in range(0, len(val_x), 65536):
                logits, _ = model(val_x[start:start + 65536])
                score_parts.append(logits.numpy())
            scores = np.concatenate(score_parts)

        metrics = evaluate(val["user"], val["y"].astype(int), scores)
        primary = metrics["primary"]
        if primary > best_primary + 1e-6:
            best_primary = primary
            best_scores = scores.copy()
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break

    os.makedirs(args.out_dir, exist_ok=True)
    metrics = evaluate(val["user"], val["y"].astype(int), best_scores)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as handle:
        json.dump(
            {
                "gauc": metrics["GAUC"] if "GAUC" in metrics else metrics["gauc"],
                "ndcg5": metrics.get("nDCG@5", metrics.get("ndcg5")),
                "primary": metrics["primary"],
            },
            handle,
        )

    with open(os.path.join(args.out_dir, "predictions.csv"), "w") as handle:
        handle.write("row_id,user_id,video_id,score\n")
        for row_id, score in enumerate(best_scores):
            handle.write(f"{row_id},{val['user'][row_id]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
