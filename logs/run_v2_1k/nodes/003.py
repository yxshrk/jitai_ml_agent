#!/usr/bin/env python3
"""Run the fixed polish stack with frequency-adaptive stochastic shared embeddings.

The only training change relative to node_000 is applied at embedding lookup time:
rare IDs are replaced by an in-batch donor ID more often than frequent IDs. Donors
remain in the same field when the stack performs a [batch, fields] lookup.
"""

import argparse
import contextlib
import importlib.util
import os
import runpy
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def install_frequency_adaptive_sse(base_probability=0.03, min_probability=0.003,
                                   max_probability=0.15):
    original_forward = torch.nn.Embedding.forward
    frequency_state = {}

    def adaptive_forward(module, indices):
        if not module.training or indices.numel() == 0:
            return original_forward(module, indices)

        key = id(module)
        counts = frequency_state.get(key)
        if (counts is None or counts.device != indices.device or
                counts.numel() != module.num_embeddings):
            counts = torch.zeros(
                module.num_embeddings, device=indices.device, dtype=torch.float32
            )
            frequency_state[key] = counts

        flat = indices.detach().reshape(-1)
        prior_frequency = counts.index_select(0, flat).reshape_as(indices)
        local_reference = prior_frequency.float().mean() + 1.0
        replacement_probability = base_probability * torch.sqrt(
            local_reference / (prior_frequency + 1.0)
        )
        replacement_probability = replacement_probability.clamp(
            min=min_probability, max=max_probability
        )

        if indices.shape[0] > 1:
            donor_rows = torch.randperm(indices.shape[0], device=indices.device)
            donors = indices.index_select(0, donor_rows)
            replace = torch.rand(
                indices.shape, device=indices.device, dtype=torch.float32
            ) < replacement_probability
            lookup_indices = torch.where(replace, donors, indices)
        else:
            lookup_indices = indices

        with torch.no_grad():
            batch_counts = torch.bincount(flat, minlength=module.num_embeddings)
            counts.add_(batch_counts.to(dtype=counts.dtype))

        return original_forward(module, lookup_indices)

    torch.nn.Embedding.forward = adaptive_forward


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    smoke = os.environ.get("SMOKE_EPOCHS")
    epochs = 1
    if smoke is not None:
        epochs = min(epochs, max(1, int(smoke)))

    spec = importlib.util.find_spec("zoo.polish_stack")
    if spec is None or not spec.origin:
        raise RuntimeError("cannot locate zoo.polish_stack on PYTHONPATH")

    install_frequency_adaptive_sse()

    sys.argv = [
        spec.origin,
        "--lr", "0.00168",
        "--dropout", "0.21",
        "--weight-decay", "0.000037",
        "--k", "24",
        "--recency-half-life", "7.0",
        "--epochs", str(epochs),
        "--data-dir", args.data_dir,
        "--out-dir", args.out_dir,
        "--seed", str(args.seed),
    ]

    with open(os.devnull, "w") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                runpy.run_path(spec.origin, run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError("zoo.polish_stack failed") from exc


if __name__ == "__main__":
    main()
