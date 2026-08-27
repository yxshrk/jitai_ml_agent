"""Generate a small, learnable KuaiRand-Pure-shaped fixture dataset."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np


FIELDS = [
    "user_id",
    "video_id",
    "tab",
    "hourmin",
    "date",
    "duration_ms",
    "long_view",
    "click",
    "like",
    "play_time_ms",
]

# The frozen docs name the full period but omit boundaries. These contiguous
# windows preserve the common early/middle/late temporal evaluation convention.
SPLITS = {
    "train": (20220408, 20220421),
    "val": (20220422, 20220428),
    "test": (20220429, 20220508),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    output_dir: Path,
    seed: int = 42,
    n_impressions: int = 5_000,
    n_users: int = 200,
    n_videos: int = 150,
) -> dict[str, int]:
    """Write deterministic date splits and return their row counts."""
    if min(n_impressions, n_users, n_videos) <= 0:
        raise ValueError("all size arguments must be positive")

    rng = np.random.default_rng(seed)
    users = rng.integers(0, n_users, size=n_impressions)
    videos = rng.integers(0, n_videos, size=n_impressions)
    tabs = rng.integers(0, 5, size=n_impressions)
    day_offsets = rng.integers(0, 31, size=n_impressions)
    minutes = rng.integers(0, 24 * 60, size=n_impressions)

    # A planted low-rank user-taste x video-appeal signal. The additional video
    # quality and observable context make the fixture realistic and learnable by FM.
    latent_dim = 3
    user_taste = rng.normal(0.0, 1.0, size=(n_users, latent_dim))
    video_taste = rng.normal(0.0, 1.0, size=(n_videos, latent_dim))
    user_bias = rng.normal(0.0, 0.35, size=n_users)
    video_quality = rng.normal(0.0, 0.75, size=n_videos)
    tab_effect = np.array([-0.30, 0.15, 0.30, -0.10, 0.05])

    raw_duration = 43_000 + 8_000 * video_quality + rng.normal(0, 7_000, n_videos)
    video_duration = np.clip(raw_duration, 8_000, 120_000).astype(int)
    durations = video_duration[videos]
    hour = minutes // 60
    evening = ((hour >= 18) & (hour <= 23)).astype(float)
    affinity = np.sum(user_taste[users] * video_taste[videos], axis=1) / np.sqrt(latent_dim)
    logits = (
        1.45 * affinity
        + 1.65 * video_quality[videos]
        + user_bias[users]
        + tab_effect[tabs]
        + 0.30 * evening
        - 0.15
        + rng.normal(0.0, 0.25, size=n_impressions)
    )
    long_view = rng.binomial(1, _sigmoid(logits))

    # Auxiliary outcomes share the engagement cause, but retain independent noise.
    click = rng.binomial(1, _sigmoid(0.80 * logits + 0.45 * long_view - 0.25))
    like = rng.binomial(1, _sigmoid(0.75 * logits + 1.00 * long_view - 1.80))
    watched_fraction = np.clip(
        0.12 + 0.68 * long_view + 0.10 * click + rng.normal(0, 0.12, n_impressions),
        0.0,
        1.25,
    )
    play_time = np.rint(durations * watched_fraction).astype(int)

    start = date(2022, 4, 8)
    dates = np.array(
        [int((start + timedelta(days=int(offset))).strftime("%Y%m%d")) for offset in day_offsets]
    )
    hourmin = (hour * 100 + minutes % 60).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, (lower, upper) in SPLITS.items():
        indices = np.flatnonzero((dates >= lower) & (dates <= upper))
        # Stable date ordering makes temporal files easy to inspect and reproduce.
        indices = indices[np.argsort(dates[indices], kind="stable")]
        counts[split] = len(indices)
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            for i in indices:
                writer.writerow(
                    [
                        int(users[i]), int(videos[i]), int(tabs[i]), int(hourmin[i]),
                        int(dates[i]), int(durations[i]), int(long_view[i]),
                        int(click[i]), int(like[i]), int(play_time[i]),
                    ]
                )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--out-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-impressions", type=int, default=5_000)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--n-videos", type=int, default=150)
    args = parser.parse_args()
    counts = generate(args.output_dir, args.seed, args.n_impressions, args.n_users, args.n_videos)
    print("generated " + ", ".join(f"{name}={count}" for name, count in counts.items()))


if __name__ == "__main__":
    main()
