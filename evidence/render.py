"""Render the judged evidence pack for one agent run.

Input : a run directory logs/run_<id>/ containing journal.jsonl
        (one JSON line per iteration, schema in CONTRACTS.md section 2).
Output: logs/run_<id>/report/{trajectory.png, results.md, RUNLOG.md}

Usage : uv run python evidence/render.py logs/run_<id> [--baseline 0.6016]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OFFICIAL_VAL_BASELINE = 0.6016


# ---------------------------------------------------------------- loading

def load_journal(run_dir: Path) -> list[dict]:
    path = run_dir / "journal.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no journal.jsonl in {run_dir}")
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty")
    return sorted(records, key=lambda r: r["n"])


# ---------------------------------------------------------------- aggregates

def aggregate(records: list[dict]) -> dict:
    accepted = [r for r in records if r.get("accepted")]
    errors = [r for r in records if r.get("error")]
    rejected = [r for r in records if not r.get("accepted") and not r.get("error")]

    best = max(accepted, key=lambda r: r["metrics"]["primary"]) if accepted else None

    per_role: dict[str, dict[str, int]] = {}
    for r in records:
        for role, tok in (r.get("tokens_by_role") or {}).items():
            slot = per_role.setdefault(role, {"in": 0, "out": 0})
            slot["in"] += int(tok.get("in", 0))
            slot["out"] += int(tok.get("out", 0))

    return {
        "n_iterations": len(records),
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "n_errors": len(errors),
        "n_interventions": sum(1 for r in records if r.get("intervention")),
        "tokens_in": sum(int(r.get("tokens_in", 0)) for r in records),
        "tokens_out": sum(int(r.get("tokens_out", 0)) for r in records),
        "wall_clock_s": sum(float(r.get("duration_s", 0.0)) for r in records),
        "best": best,
        "per_role": per_role,
    }


# ---------------------------------------------------------------- trajectory.png

def render_trajectory(records: list[dict], out_path: Path, baseline: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in records if r.get("metrics") and not r.get("error")]
    acc = [r for r in scored if r.get("accepted")]
    rej = [r for r in scored if not r.get("accepted")]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(baseline, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.annotate(f"official FM baseline ({baseline:g})",
                xy=(records[0]["n"], baseline), xytext=(2, 4),
                textcoords="offset points", fontsize=8, color="#666666")

    xs = [r["n"] for r in records]
    ax.plot([r["n"] for r in records], [r.get("val_best_so_far") for r in records],
            color="#2b6cb0", lw=0.9, alpha=0.7, label="val best so far", zorder=2)

    if rej:
        ax.scatter([r["n"] for r in rej], [r["metrics"]["primary"] for r in rej],
                   facecolors="none", edgecolors="#c05621", s=45, lw=1.3,
                   label="rejected / reverted", zorder=3)
    if acc:
        ax.scatter([r["n"] for r in acc], [r["metrics"]["primary"] for r in acc],
                   color="#2f855a", s=48, label="accepted", zorder=4)

    # annotate each accepted jump with a short hypothesis tag
    for r in acc:
        tag = (r.get("hypothesis") or "").strip()
        if len(tag) > 32:
            tag = tag[:29] + "..."
        ax.annotate(tag, xy=(r["n"], r["metrics"]["primary"]),
                    xytext=(4, 7), textcoords="offset points",
                    fontsize=7, color="#2f855a")

    ax.set_xlabel("iteration")
    ax.set_ylabel("validation primary score")
    ax.set_title("Agent run: validation score vs iteration")
    ax.set_xticks(xs)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- results.md

def render_results(records: list[dict], agg: dict, out_path: Path, baseline: float) -> None:
    best = agg["best"]
    lines = ["# Run results", ""]
    if best is not None:
        m = best["metrics"]
        lines += [
            "| metric | value |",
            "|---|---|",
            f"| final valid best GAUC | {m['gauc']:.4f} |",
            f"| final valid best nDCG@5 | {m['ndcg5']:.4f} |",
            f"| final valid best primary | {m['primary']:.4f} |",
            f"| official FM baseline (valid primary) | {baseline:.4f} |",
            f"| delta vs baseline | {m['primary'] - baseline:+.4f} |",
            f"| best node | {best['node_id']} (iteration {best['n']}) |",
        ]
    else:
        lines.append("No accepted iteration - run produced no improvement over baseline.")
    lines += [
        "",
        "## Run accounting",
        "",
        "| | |",
        "|---|---|",
        f"| iterations | {agg['n_iterations']} |",
        f"| accepted | {agg['n_accepted']} |",
        f"| rejected | {agg['n_rejected']} |",
        f"| errors | {agg['n_errors']} |",
        f"| human interventions | {agg['n_interventions']} |",
        f"| tokens in | {agg['tokens_in']:,} |",
        f"| tokens out | {agg['tokens_out']:,} |",
        f"| total wall-clock | {agg['wall_clock_s']:.0f} s ({agg['wall_clock_s'] / 3600:.2f} h) |",
    ]
    if agg["per_role"]:
        lines += ["", "## Token split by role", "", "| role | tokens in | tokens out |", "|---|---|---|"]
        for role, tok in sorted(agg["per_role"].items()):
            lines.append(f"| {role} | {tok['in']:,} | {tok['out']:,} |")
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- RUNLOG.md

def render_runlog(records: list[dict], out_path: Path) -> None:
    lines = [
        "# Run log (per iteration)",
        "",
        "| n | action | hypothesis | change | primary | accepted | error / recovery |",
        "|---|---|---|---|---|---|---|",
    ]

    def esc(s: str) -> str:
        return (s or "").replace("|", "\\|").replace("\n", " ").strip()

    for r in records:
        primary = r["metrics"]["primary"] if r.get("metrics") else None
        status = "yes" if r.get("accepted") else "no"
        if r.get("intervention"):
            status += " (HUMAN INTERVENTION)"
        err = "-"
        if r.get("error"):
            err = esc(r["error"])
            if r.get("recovery"):
                err += f" -> {r['recovery']}"
        elif r.get("recovery"):
            err = f"-> {r['recovery']}"
        lines.append(
            f"| {r['n']} | {r.get('action', '?')} | {esc(r.get('hypothesis'))} "
            f"| {esc(r.get('change_summary'))} "
            f"| {'-' if primary is None else f'{primary:.4f}'} "
            f"| {status} | {err} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- entry point

def render(run_dir: Path, baseline: float = OFFICIAL_VAL_BASELINE) -> Path:
    run_dir = Path(run_dir)
    records = load_journal(run_dir)
    agg = aggregate(records)
    report = run_dir / "report"
    report.mkdir(exist_ok=True)
    render_trajectory(records, report / "trajectory.png", baseline)
    render_results(records, agg, report / "results.md", baseline)
    render_runlog(records, report / "RUNLOG.md")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="logs/run_<id> directory containing journal.jsonl")
    ap.add_argument("--baseline", type=float, default=OFFICIAL_VAL_BASELINE)
    args = ap.parse_args()
    report = render(args.run_dir, args.baseline)
    print(f"report written to {report}")


if __name__ == "__main__":
    main()
