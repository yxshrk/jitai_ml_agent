"""Robustness audit of the designated champion (run_bigclock_07 node_006).

Analysis only — no training, no model selection. Computes, from saved validation
predictions:
  1. paired per-user bootstrap CI of the champion-vs-baseline primary delta
  2. per-day (temporal slice) deltas across the validation week
  3. user-activity-bin deltas
Writes evidence/bc07_robustness.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.official.evaluate import evaluate  # noqa: E402

RUN = ROOT / "logs/run_bigclock_07"


def load_scores(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=3)


def primary(users, labels, scores, mask=None) -> float:
    if mask is not None:
        users, labels, scores = users[mask], labels[mask], scores[mask]
    return evaluate(users, labels, scores)["primary"]


def main() -> None:
    val = np.load(ROOT / "data/real_ws/val.npz")
    users, labels, dates = val["user"], val["y"].astype(int), val["date"]
    champ = load_scores(RUN / "node_006/predictions.csv")
    base = load_scores(RUN / "calib_seed42/predictions.csv")
    assert len(champ) == len(base) == len(users)

    p_champ, p_base = primary(users, labels, champ), primary(users, labels, base)
    delta = p_champ - p_base

    uniq = np.unique(users)
    order = np.argsort(users, kind="stable")
    su = users[order]
    starts = np.searchsorted(su, uniq, side="left")
    ends = np.searchsorted(su, uniq, side="right")
    rng = np.random.RandomState(7)
    deltas = []
    for b in range(400):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[i]:ends[i]] for i in pick])
        # relabel duplicated users so each bootstrap copy stays a distinct "user"
        reps = (ends - starts)[pick]
        boot_users = np.repeat(np.arange(len(pick)), reps)
        d = (evaluate(boot_users, labels[idx], champ[idx])["primary"]
             - evaluate(boot_users, labels[idx], base[idx])["primary"])
        deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    day_rows = []
    for d in np.unique(dates):
        m = dates == d
        day_rows.append((int(d), int(m.sum()),
                         primary(users, labels, champ, m) - primary(users, labels, base, m)))

    counts = {u: 0 for u in uniq}
    for u in users:
        counts[u] += 1
    ucount = np.array([counts[u] for u in users])
    act_rows = []
    for name, m in (("<=3 impressions", ucount <= 3),
                    ("4-8", (ucount > 3) & (ucount <= 8)),
                    (">8", ucount > 8)):
        if m.sum():
            act_rows.append((name, int(m.sum()),
                             primary(users, labels, champ, m) - primary(users, labels, base, m)))

    md = [
        "# Champion robustness audit (analysis only; predeclared: no re-selection)",
        "",
        f"Champion (node_006 ensemble) primary: **{p_champ:.6f}**; baseline (calib seed42): {p_base:.6f}; "
        f"delta **{delta:+.6f}**.",
        "",
        f"## Paired per-user bootstrap (400 resamples, users resampled with replacement)",
        f"Delta 95% CI: **[{lo:+.6f}, {hi:+.6f}]** "
        f"({'excludes zero — the gain is not a user-sampling artifact' if lo > 0 else 'INCLUDES zero'}).",
        "",
        "## Temporal slices (per validation day)",
        "| date | rows | delta |", "|---|---:|---:|",
        *[f"| {d} | {n} | {v:+.5f} |" for d, n, v in day_rows],
        "",
        "## User-activity bins",
        "| bin | rows | delta |", "|---|---:|---:|",
        *[f"| {b} | {n} | {v:+.5f} |" for b, n, v in act_rows],
        "",
        "Interpretation notes: per-day deltas fluctuate (small slices); the audit asks "
        "whether the gain is broad-based rather than concentrated in one day or one "
        "user segment. Bootstrap keeps each user's rows together (cluster bootstrap).",
    ]
    out = ROOT / "evidence/bc07_robustness.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    print(f"delta {delta:+.6f}  CI [{lo:+.6f},{hi:+.6f}]")
    for d, n, v in day_rows:
        print(f"day {d}: {v:+.5f} ({n})")


if __name__ == "__main__":
    main()
