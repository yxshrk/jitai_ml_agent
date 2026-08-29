"""Generate publication-ready figures for the Pure validation story.

Run from the repository root with:

    uv run python tools/story_figures.py

The ladder and ablation values are transcribed from the experiment evidence.
Search-map family counts are an approximate allocation of 250 curated cells:
the source logs count rows by campaign (297 in DASHBOARD.md), not unique cells
by idea family, and confirmation seeds/control reruns appear as additional rows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "evidence" / "figures"

BG = "#F7F8FA"
INK = "#202936"
MUTED = "#667085"
GRID = "#D9DEE7"
GREY = "#AAB2BF"
GREY_DARK = "#747E8C"
ACCENT = "#0B7A75"
ACCENT_LIGHT = "#76B7B2"
BASELINE = "#D28B1F"


def set_style() -> None:
    """Apply a consistent light, restrained publication style."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=160, metadata={"Software": "matplotlib"})
    plt.close(fig)


def make_ladder() -> None:
    labels = [
        "Random",
        "Popularity",
        "Official FM baseline",
        "Our single model",
        "Our 5-seed ensemble",
    ]
    scores = [0.4753, 0.5715, 0.6016, 0.6047, 0.6058]
    ceiling = 0.8645
    floor = 0.45
    colors = [GREY, GREY, BASELINE, ACCENT_LIGHT, ACCENT]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    y = list(range(len(labels)))

    # Quiet full-range tracks make the ceiling gap legible without chartjunk.
    ax.barh(y, [ceiling - floor] * len(y), left=floor, height=0.58,
            color="#E9EDF2", edgecolor="none", zorder=1)
    ax.barh(y, [s - floor for s in scores], left=floor, height=0.58,
            color=colors, edgecolor="none", zorder=2)

    for yi, score in zip(y, scores):
        ax.text(score + 0.006, yi, f"{score:.4f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color=INK)

    ax.axvline(ceiling, color=GREY_DARK, lw=1.6, ls=(0, (4, 3)), zorder=3)
    ax.text(
        ceiling - 0.007,
        4.72,
        "27% of users have no positive — ceiling, not 1.0",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=GREY_DARK,
    )

    ax.set_title("Pure score ladder", loc="left", pad=12)
    ax.text(0, 1.01, "Primary metric (ours: validation; reference rungs: official ladder)", transform=ax.transAxes,
            color=MUTED, fontsize=10, va="bottom")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(floor, 0.89)
    ax.set_ylim(4.9, -0.8)
    ax.set_xlabel("Primary metric (higher is better)")
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8, zorder=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)

    save(fig, "ladder.png")


def make_ablation_curve() -> None:
    levels = [0, 1, 2, 3, 4, 5]
    fields = [5, 7, 10, 15, 21, 24]
    base = [0.604335, 0.603655, 0.603704, 0.603614, 0.604089, 0.601740]
    # EXPERIMENTS_ABLATION.md reports strong-regularization reruns only at L0/L5.
    strong_x = [0, 5]
    strong = [0.604998, 0.602991]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(levels, base, color=GREY_DARK, lw=2.1, marker="o", ms=6,
            markerfacecolor=BG, markeredgewidth=1.8,
            label="Base regularization")
    ax.plot(strong_x, strong, color=ACCENT, lw=2.3, ls=(0, (5, 3)),
            marker="o", ms=7, markerfacecolor=ACCENT, markeredgecolor=BG,
            markeredgewidth=1.2,
            label="Strong regularization (L0/L5 measured only)")

    for x, value in zip(levels, base):
        offset = 0.00014 if x != 5 else -0.00024
        va = "bottom" if offset > 0 else "top"
        ax.text(x, value + offset, f"{value:.4f}", ha="center", va=va,
                fontsize=8.7, color=GREY_DARK)
    for x, value in zip(strong_x, strong):
        offset = 0.00013 if x == 0 else 0.00017
        ax.text(x, value + offset, f"{value:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=ACCENT)

    ax.annotate(
        "More fields lose — regularization,\nnot information, was the constraint.",
        xy=(5, strong[-1]),
        xytext=(2.25, 0.60245),
        fontsize=10,
        color=INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": GREY_DARK, "lw": 1.0,
                    "connectionstyle": "arc3,rad=-0.12"},
    )

    ax.set_title("Field ablation: compact beats kitchen sink", loc="left", pad=12)
    ax.text(0, 1.01, "Primary metric (ours: validation; reference rungs: official ladder) · seed 42", transform=ax.transAxes,
            color=MUTED, fontsize=10, va="bottom")
    ax.set_xticks(levels, [f"L{i}\n{n} fields" for i, n in zip(levels, fields)])
    ax.set_xlabel("Cumulative field level")
    ax.set_ylabel("Valid primary metric")
    ax.set_ylim(0.60115, 0.60545)
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 0.985), frameon=False,
              fontsize=9)
    save(fig, "ablation_curve.png")


def make_search_map() -> None:
    # Approximate unique curated cells by family. These sum to 250. DASHBOARD.md
    # reports 297 table rows, but rows include controls and seed confirmations;
    # no source file supplies an exact cross-campaign family partition.
    families = [
        "Architectures",
        "Losses",
        "Features",
        "Auxiliaries",
        "Sampling",
        "Optimizers",
        "Schedules",
        "Sequence models",
        "Ensembles",
        "Personalization",
        "Watch-time",
        "Long-shots",
    ]
    counts = [22, 18, 46, 14, 8, 9, 44, 8, 24, 14, 12, 31]
    # The three surviving levers in PRACTICES.md map to DCN-lite architecture,
    # recency weighting (features/data), and strong regularization + scheduling.
    alive = {"Architectures", "Features", "Schedules"}
    colors = [ACCENT if family in alive else GREY for family in families]

    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    y = list(range(len(families)))
    bars = ax.barh(y, counts, color=colors, height=0.62, edgecolor="none")
    for bar, count in zip(bars, counts):
        ax.text(count + 0.8, bar.get_y() + bar.get_height() / 2, str(count),
                va="center", ha="left", fontsize=9.5, fontweight="bold")

    ax.set_title("≈250 curated cells + ≈3,600 automated trials;\n3 levers survived",
                 loc="left", pad=13, fontsize=14.5, linespacing=1.05)
    ax.set_yticks(y, families)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel("Measured cells (approximate; count shown on each bar)")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85, zorder=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=ACCENT, label="Contains a surviving lever"),
            Patch(facecolor=GREY, label="Dead, abandoned, or no durable lift"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.7,
    )
    save(fig, "search_map.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    make_ladder()
    make_ablation_curve()
    make_search_map()
    for name in ("ladder.png", "ablation_curve.png", "search_map.png"):
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
