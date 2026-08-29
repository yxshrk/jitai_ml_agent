"""Generate publication-ready figures for the Pure validation story.

Run from the repository root with:

    uv run python tools/story_figures.py

The ladder and ablation values are transcribed from the experiment evidence.
Search-map family counts are an approximate allocation of 250 curated cells:
the source logs count rows by campaign (297 in DASHBOARD.md), not unique cells
by idea family, and confirmation seeds/control reruns appear as additional rows.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch, Rectangle
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
DEAD = "#B55A54"

PURE_BASELINE = 0.6016


def read_source(relative_path: str) -> str:
    """Read an allowed evidence source; figure constants cite these files below."""
    return (ROOT / relative_path).read_text(encoding="utf-8")


def require_source_value(relative_path: str, literal: str) -> None:
    """Fail loudly if a transcribed source value disappears from its ledger."""
    if literal not in read_source(relative_path):
        raise ValueError(f"{literal!r} not found in {relative_path}")


def official_summaries() -> list[dict]:
    """Load the complete official-run summary set used by autonomy_cost.png."""
    paths = sorted((ROOT / "logs").glob("run_official_*/summary.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def calibration_rows() -> list[tuple[float, float, str]]:
    """Extract every populated expected/realized pair from official journals."""
    rows: list[tuple[float, float, str]] = []
    for path in sorted((ROOT / "logs").glob("run_official_*/journal.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            expected = row.get("expected_delta")
            realized = row.get("realized_delta")
            if expected is not None and realized is not None:
                rows.append((float(expected), float(realized), path.parent.name))
    return rows


def cumulative_llm_spend() -> float:
    """Read the maximum cumulative dollar-ledger value in official journals."""
    totals: list[float] = []
    for path in sorted((ROOT / "logs").glob("run_official_*/journal.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line).get("usd_total")
            if value is not None:
                totals.append(float(value))
    if not totals:
        raise ValueError("no usd_total values in official journals")
    return max(totals)


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


def make_levers() -> None:
    """Plot representative measured deltas for every major tried lever family."""
    # Alive values: PRACTICES.md. Strong-reg is the confirmed three-seed
    # difference 0.604660 - 0.603684 from EXPERIMENTS_ABLATION.md.
    require_source_value("PRACTICES.md", "+0.0025 over FM")
    require_source_value("PRACTICES.md", "+0.0027 ± 0.0012 confirmed")
    require_source_value("zoo/EXPERIMENTS_ABLATION.md", "0.603684 +/- 0.000461")
    require_source_value("zoo/EXPERIMENTS_ABLATION.md", "0.604660 +/- 0.000309")

    # Dead-family representatives use each campaign's explicit comparator.
    # This mirrors the ledgers rather than forcing unlike campaigns to one base.
    require_source_value("zoo/EXPERIMENTS_SEQ.md", "0.581807875")
    require_source_value("zoo/EXPERIMENTS_SEQ.md", "0.604998355")
    require_source_value("zoo/EXPERIMENTS_AUDIT.md", "-0.003392")
    require_source_value("zoo/EXPERIMENTS_AUDIT.md", "-0.002354")
    require_source_value("zoo/EXPERIMENTS.md", "| LightGBM alone | 0.6621 | 0.5328 | 0.5974")
    require_source_value("zoo/EXPERIMENTS.md", "| k=32 | 0.6710 | 0.5367 | 0.6039")
    require_source_value("zoo/EXPERIMENTS_ABLATION.md", "losing 0.002349 versus L4 and 0.002595 versus L0")
    require_source_value("zoo/EXPERIMENTS_DIMS.md", "| S2-play | play-time fraction MSE")
    require_source_value("zoo/EXPERIMENTS_DIMS.md", "0.603264")
    require_source_value("zoo/EXPERIMENTS_DIMS.md", "0.605425")
    require_source_value("zoo/EXPERIMENTS_RECHECK.md", "**-0.000275**")
    require_source_value("zoo/EXPERIMENTS_RECHECK.md", "**-0.000597**")

    labels = [
        "DCN-lite architecture",
        "7-day recency weighting",
        "Strong-reg + schedule",
        "Causal session fields",
        "Click auxiliary head",
        "Capacity increase (k=32)",
        "LightGBM alone",
        "Kitchen-sink feature family",
        "Watch-time auxiliary",
        "Duration-regime heads",
        "LambdaRank (0.3 mix)",
        "Sequence model (SASRec)",
    ]
    values = [
        0.0025,
        0.0027,
        0.604660 - 0.603684,
        -0.000275,
        -0.000597,
        0.6039 - 0.6047,
        0.5974 - PURE_BASELINE,
        -0.002595,
        0.603264 - 0.605425,
        -0.002354,
        -0.003392,
        0.581807875 - 0.604998355,
    ]
    alive = 3
    colors = [ACCENT] * alive + [GREY if value > -0.001 else DEAD for value in values[alive:]]

    fig, ax = plt.subplots(figsize=(8, 5.9), constrained_layout=True)
    y = list(range(len(labels)))
    bars = ax.barh(y, values, color=colors, edgecolor="none", height=0.62)
    ax.axvline(0, color=GREY_DARK, lw=1.0)
    for bar, value in zip(bars, values):
        if value < -0.015:
            x, align, text_color = value + 0.00045, "left", BG
        else:
            x = value + (0.00030 if value >= 0 else -0.00030)
            align, text_color = ("left" if value >= 0 else "right"), INK
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            va="center",
            ha=align,
            fontsize=8.8,
            fontweight="bold",
            color=text_color,
        )

    ax.set_title("Every lever, measured. Three survived.", loc="left", pad=23)
    ax.text(
        0,
        1.005,
        "Validation-primary delta vs each campaign's named control",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        va="bottom",
    )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.0255, 0.0050)
    ax.set_xlabel("Measured primary delta (higher is better)")
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%+.3f"))
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.85, zorder=0)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9.2)
    ax.spines["left"].set_visible(False)
    save(fig, "levers.png")


def make_weekend_arc() -> None:
    """Use exact commit timestamps as approximate discovery timestamps."""
    require_source_value("logs/RUNS.md", "node_002 (0.6042)")
    require_source_value("PRACTICES.md", "0.6047 ± 0.0003")
    require_source_value("SUBMISSION_RECIPE.md", "0.60577")
    require_source_value("SUBMISSION_RECIPE.md", "0.60602")

    # Times come from git log --date=iso-local. Baseline uses the initial Thu
    # commit; later positions use the evidence commits named beside each point.
    times = [
        datetime(2026, 8, 27, 23, 49),  # 460bc06
        datetime(2026, 8, 28, 1, 40),   # a74b3d2, DCN campaign
        datetime(2026, 8, 28, 10, 21),  # 12cc1b6, recency frozen
        datetime(2026, 8, 28, 14, 0),   # 674e316, strong L0
        datetime(2026, 8, 29, 2, 36),   # fadca6c, 5-seed recipe
        datetime(2026, 8, 29, 19, 47),  # 9da7190, extended greedy
    ]
    require_source_value("zoo/EXPERIMENTS_HIST.md", "0.6043165 ± 0.0011685")
    scores = [0.6016, 0.6042, 0.6043165, 0.6047, 0.60577, 0.60602]
    labels = [
        "baseline reproduced\n0.6016",
        "agent finds DCN\n0.6042",
        "recency +0.0027\n0.6043 confirmed",
        "strong-L0\n0.6047",
        "5-seed ensemble\n0.60577",
        "extended greedy\n0.60602",
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.step(times, scores, where="post", color=ACCENT, lw=2.4)
    ax.scatter(times, scores, s=55, color=ACCENT, edgecolor=BG, linewidth=1.2, zorder=3)
    offsets = [(5, -34), (5, 13), (5, -38), (5, 13), (5, -38), (-5, 13)]
    aligns = ["left", "left", "left", "left", "left", "right"]
    for x, yv, label, offset, align in zip(times, scores, labels, offsets, aligns):
        ax.annotate(label, (x, yv), xytext=offset, textcoords="offset points",
                    ha=align, va="bottom", fontsize=8.7, color=INK)

    sunday = datetime(2026, 8, 30, 9, 0)
    ax.step([times[-1], sunday], [scores[-1], scores[-1]], where="post",
            color=ACCENT, lw=2.4)
    ax.scatter([sunday], [scores[-1]], s=70, facecolor=BG, edgecolor=GREY_DARK,
               linewidth=1.6, zorder=4)
    ax.annotate("(final Sun)", (sunday, scores[-1]), xytext=(-4, -25),
                textcoords="offset points", ha="right", color=GREY_DARK, fontsize=9)

    ax.set_title("One weekend, one rising best-known score", loc="left", pad=23)
    ax.text(0, 1.01, "Pure validation primary · timestamps from evidence commits",
            transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")
    ax.set_ylabel("Best-known primary")
    ax.set_ylim(0.6008, 0.6070)
    ax.yaxis.set_major_locator(MultipleLocator(0.001))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xticks([
        datetime(2026, 8, 27, 22),
        datetime(2026, 8, 28, 6),
        datetime(2026, 8, 28, 12),
        datetime(2026, 8, 28, 18),
        datetime(2026, 8, 29, 0),
        datetime(2026, 8, 29, 6),
        datetime(2026, 8, 29, 12),
        datetime(2026, 8, 29, 18),
        datetime(2026, 8, 30, 0),
        datetime(2026, 8, 30, 6),
        datetime(2026, 8, 30, 12),
    ])
    ax.xaxis.set_major_formatter(DateFormatter("%a\n%H:%M"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_xlim(datetime(2026, 8, 27, 21), datetime(2026, 8, 30, 12))
    save(fig, "weekend_arc.png")


def make_ensemble_effect() -> None:
    """Plot only seed scores individually present in the allowed ledgers."""
    # Pure: exact strong-stack seeds 42/43/44 from EXPERIMENTS_ABLATION.md.
    pure_singles = [0.604998, 0.604250, 0.604730]
    for value in pure_singles:
        require_source_value("zoo/EXPERIMENTS_ABLATION.md", f"{value:.6f}")
    require_source_value("SUBMISSION_RECIPE.md", "0.60602")
    # 1K: SUBMISSION_RECIPE.md logs only the single-seed range endpoints.
    one_k_singles = [0.6073, 0.6216]
    require_source_value("SUBMISSION_RECIPE.md", "singles 0.6073-0.6216")
    require_source_value("SUBMISSION_RECIPE.md", "0.6323 valid primary")

    panels = [
        ("Pure", pure_singles, 0.60602, "3 individually logged seeds"),
        ("1K", one_k_singles, 0.6323, "documented range endpoints"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.8))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.22, wspace=0.20)
    for ax, (title, singles, ensemble, note) in zip(axes, panels):
        center = 0.0
        jitter = [(-0.10 + 0.20 * i / max(len(singles) - 1, 1)) for i in range(len(singles))]
        ax.scatter(jitter, singles, s=58, facecolor=BG, edgecolor=GREY_DARK,
                   linewidth=1.6, zorder=3, label="Single seeds")
        ax.scatter([center], [ensemble], marker="D", s=82, color=ACCENT,
                   edgecolor=BG, linewidth=1.2, zorder=4, label="Rank ensemble")
        ax.text(center + 0.025, ensemble, f"{ensemble:.5f}" if title == "Pure" else f"{ensemble:.4f}",
                ha="left", va="center", fontsize=9, fontweight="bold", color=ACCENT)
        ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color=MUTED)
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xlim(-0.25, 0.35)
        ax.set_xticks([])
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_color(GRID)
    axes[0].set_ylim(0.6037, 0.6065)
    axes[1].set_ylim(0.604, 0.636)
    axes[0].set_ylabel("Validation primary")
    fig.suptitle("Seed diversity becomes ensemble lift", x=0.02, y=0.97, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(0.02, 0.035,
             "same recipe, different random seeds — averaging their rankings beats every individual.",
             color=MUTED, fontsize=9.2)
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.5)
    save(fig, "ensemble_effect.png")


def make_bonus_arc() -> None:
    require_source_value("SUBMISSION_RECIPE.md", "0.6134 -> tuned single")
    require_source_value("SUBMISSION_RECIPE.md", "0.6214 -> ensemble 0.6323")
    values = [0.6134, 0.6214, 0.6323]
    labels = ["Default transfer", "Tuned single", "5-seed ensemble"]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    x = [0, 1, 2]
    bars = ax.bar(x, values, width=0.62, color=[GREY, ACCENT_LIGHT, ACCENT], edgecolor="none")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0012, f"{value:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    placeholder_x = 3
    ax.add_patch(Rectangle((placeholder_x - 0.31, 0.60), 0.62, 0.035,
                           facecolor="none", edgecolor=GREY_DARK, linewidth=1.5,
                           linestyle=(0, (4, 3))))
    ax.text(placeholder_x, 0.637, "(final Sun)", ha="center", va="bottom",
            fontsize=9, color=GREY_DARK)
    ax.text(0.985, 0.12, "27K result\nnot yet logged", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.3, color=GREY_DARK,
            bbox={"facecolor": BG, "edgecolor": GREY_DARK, "linewidth": 1.0,
                  "linestyle": (0, (4, 3)), "boxstyle": "square,pad=0.45"})

    ax.set_title("1K bonus: transfer, tune, ensemble", loc="left", pad=12)
    ax.text(0, 1.01, "Absolute validation primary · no official 1K baseline",
            transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")
    ax.set_xticks([0, 1, 2, 3], labels + ["Pending"])
    ax.set_ylabel("Validation primary")
    ax.set_ylim(0.60, 0.64)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.85, zorder=0)
    ax.tick_params(axis="x", length=0, pad=8)
    save(fig, "bonus_arc.png")


def make_calibration() -> None:
    rows = calibration_rows()
    if not rows:
        raise ValueError("no populated expected/realized delta pairs")
    expected = [row[0] for row in rows]
    realized = [row[1] for row in rows]
    pure = [not row[2].startswith("run_official_1k") for row in rows]
    colors = [ACCENT if flag else BASELINE for flag in pure]

    lo = min(expected + realized)
    hi = max(expected + realized)
    padding = (hi - lo) * 0.08
    limits = (lo - padding, hi + padding)

    fig, ax = plt.subplots(figsize=(8, 5.1), constrained_layout=True)
    ax.scatter(expected, realized, s=52, c=colors, edgecolor=BG,
               linewidth=1.0, alpha=0.92, zorder=3)
    ax.plot(limits, limits, color=GREY_DARK, lw=1.4, ls=(0, (4, 3)), label="perfect calibration")
    ax.axhline(0, color=GRID, lw=0.9)
    ax.axvline(0, color=GRID, lw=0.9)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("the agent predicts its own results -\nand we measure how well",
                 loc="left", pad=22, fontsize=14.2, linespacing=1.0)
    ax.text(0, 1.01, f"{len(rows)} iterations with both fields populated",
            transform=ax.transAxes, color=MUTED, fontsize=10, va="bottom")
    ax.set_xlabel("Expected primary delta")
    ax.set_ylabel("Realized primary delta")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%+.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%+.3f"))
    ax.grid(color=GRID, linewidth=0.7, alpha=0.55)
    ax.legend(handles=[
        Patch(facecolor=ACCENT, label="Pure"),
        Patch(facecolor=BASELINE, label="1K"),
        plt.Line2D([0], [0], color=GREY_DARK, ls=(0, (4, 3)), label="y = x"),
    ], loc="lower right", frameon=False, fontsize=8.8)
    save(fig, "calibration.png")


def make_autonomy_cost() -> None:
    require_source_value("logs/RUNS.md", "All runs: 0 interventions")
    require_source_value("zoo/EXPERIMENTS_NIGHT.md", "CPU-only")
    summaries = official_summaries()
    if not summaries:
        raise ValueError("no logs/run_official_*/summary.json files")
    median_wall_s = statistics.median(float(row["wall_s"]) for row in summaries)
    median_tokens = statistics.median(int(row["tokens_total"]) for row in summaries)
    spend = cumulative_llm_spend()

    cards = [
        ("0", "interventions", "all logged runs"),
        (f"{median_wall_s / 60:.1f}m", "median wall-clock", f"{len(summaries)} official summaries"),
        (f"{median_tokens / 1000:.1f}k", "median tokens / run", f"{len(summaries)} official summaries"),
        ("0", "GPU-hours", "CPU-only benchmark"),
        (f"${spend:.2f}", "total LLM spend", "cumulative official journal ledger"),
    ]

    fig = plt.figure(figsize=(8, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    fig.suptitle("Autonomy, with the meter running", x=0.02, ha="left",
                 fontsize=16, fontweight="bold")
    for ax, (value, label, note) in zip(axes, cards):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
            spine.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.61, value, transform=ax.transAxes, ha="center", va="center",
                fontsize=25, fontweight="bold", color=ACCENT)
        ax.text(0.5, 0.35, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=10.2, fontweight="bold", color=INK)
        ax.text(0.5, 0.17, note, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.0, color=MUTED)
    save(fig, "autonomy_cost.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    make_ladder()
    make_ablation_curve()
    make_search_map()
    make_levers()
    make_weekend_arc()
    make_ensemble_effect()
    make_bonus_arc()
    make_calibration()
    make_autonomy_cost()
    names = (
        "ladder.png",
        "ablation_curve.png",
        "search_map.png",
        "levers.png",
        "weekend_arc.png",
        "ensemble_effect.png",
        "bonus_arc.png",
        "calibration.png",
        "autonomy_cost.png",
    )
    for name in names:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
