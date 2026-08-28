#!/usr/bin/env python3
"""Post-hoc diagnostics and rank ensembles for one or more experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.official.evaluate import evaluate  # noqa: E402

EPSILON = 0.002
HIGH_CHANGE_TAU = 0.90
HIGH_CHANGE_TOP1_PCT = 10.0


@dataclass
class ValidationData:
    users: np.ndarray
    labels: np.ndarray
    duration_ms: np.ndarray
    tabs: np.ndarray
    history_depth: np.ndarray
    list_length: np.ndarray
    segments: dict[str, np.ndarray]
    segment_notes: list[str] = field(default_factory=list)


@dataclass
class NodeData:
    run_dir: Path
    node_id: str
    parent: str
    accepted: bool
    record: dict
    scores: np.ndarray | None
    metrics_payload: dict | None
    note: str | None = None

    @property
    def key(self) -> str:
        return f"{self.run_dir.name}/{self.node_id}"


@dataclass
class RankingChange:
    node_key: str
    parent_key: str
    tau: float
    top1_changed_pct: float
    metric_delta: float
    classification: str
    high_change: bool


@dataclass
class EnsembleResult:
    name: str
    members: list[str]
    metrics: dict[str, float] | None
    delta: float | None
    note: str | None = None


@dataclass
class RunAnalysis:
    run_dir: Path
    nodes: list[NodeData]
    metrics: dict[str, dict[str, float]]
    segment_rows: list[dict]
    ranking_changes: list[RankingChange]
    ensembles: list[EnsembleResult]
    best_key: str | None
    best_metrics: dict[str, float] | None
    curve_rows: list[dict]
    notes: list[str]


def official_metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    raw = evaluate(users.tolist(), labels.astype(int).tolist(), scores.astype(float).tolist())
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg5": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }


def _npz_vector(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name]).reshape(-1)
    return None


def _load_csv_columns(path: Path, wanted: Sequence[str]) -> dict[str, np.ndarray]:
    values: dict[str, list] = {name: [] for name in wanted}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(wanted) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            for name in wanted:
                values[name].append(int(float(row[name])))
    return {name: np.asarray(column, dtype=np.int64) for name, column in values.items()}


def _history_counts(data_dir: Path) -> Counter:
    npz_path = data_dir / "train.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as train:
            users = _npz_vector(train, "user", "user_id")
            if users is not None:
                unique, counts = np.unique(users, return_counts=True)
                return Counter(dict(zip(unique.tolist(), counts.tolist())))
    csv_path = data_dir / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"neither train.npz nor train.csv found under {data_dir}")
    counts: Counter = Counter()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "user_id" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path}: missing user_id")
        for row in reader:
            counts[int(row["user_id"])] += 1
    return counts


def load_validation_data(data_dir: Path) -> ValidationData:
    """Load validation labels/features and construct all requested segment masks."""
    npz_path = data_dir / "val.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as val:
            users = _npz_vector(val, "user", "user_id")
            labels = _npz_vector(val, "y", "long_view")
            duration = _npz_vector(val, "duration_ms")
            tabs = _npz_vector(val, "tab")
            if tabs is None and "X" in val.files:
                x = np.asarray(val["X"])
                if x.ndim == 2 and x.shape[1] >= 4:
                    tabs = x[:, 3].astype(np.int64)
                    if "field_dims" in val.files and len(val["field_dims"]) >= 3:
                        tabs = tabs - int(np.sum(val["field_dims"][:3]))
        missing = [name for name, value in (("user", users), ("y", labels),
                                             ("duration_ms", duration), ("tab", tabs))
                   if value is None]
        if missing:
            csv_path = data_dir / "val.csv"
            if not csv_path.exists():
                raise ValueError(f"{npz_path}: missing {missing} and val.csv is unavailable")
            cols = _load_csv_columns(csv_path, ["user_id", "long_view", "duration_ms", "tab"])
            users, labels, duration, tabs = (cols["user_id"], cols["long_view"],
                                               cols["duration_ms"], cols["tab"])
    else:
        cols = _load_csv_columns(data_dir / "val.csv",
                                 ["user_id", "long_view", "duration_ms", "tab"])
        users, labels, duration, tabs = (cols["user_id"], cols["long_view"],
                                         cols["duration_ms"], cols["tab"])

    assert users is not None and labels is not None and duration is not None and tabs is not None
    users = np.asarray(users, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    duration = np.asarray(duration, dtype=np.float64)
    tabs = np.asarray(tabs, dtype=np.int64)
    n = len(users)
    if not (len(labels) == len(duration) == len(tabs) == n):
        raise ValueError("validation feature arrays have different lengths")

    train_counts = _history_counts(data_dir)
    history_depth = np.asarray([train_counts.get(int(user), 0) for user in users], dtype=np.int64)
    val_counts = Counter(users.tolist())
    list_length = np.asarray([val_counts[int(user)] for user in users], dtype=np.int64)

    unique_users = np.unique(users)
    user_depths = np.asarray([train_counts.get(int(user), 0) for user in unique_users], dtype=float)
    q1, q2 = np.quantile(user_depths, [1 / 3, 2 / 3]) if len(user_depths) else (0.0, 0.0)
    segments: dict[str, np.ndarray] = {
        "duration<=18000ms": duration <= 18000,
        "duration>18000ms": duration > 18000,
    }
    for tab in sorted(np.unique(tabs).tolist()):
        segments[f"tab={tab}"] = tabs == tab
    segments.update({
        f"history:low<={q1:g}": history_depth <= q1,
        f"history:mid({q1:g},{q2:g}]": (history_depth > q1) & (history_depth <= q2),
        f"history:high>{q2:g}": history_depth > q2,
        "val-list<=5": list_length <= 5,
        "val-list>5": list_length > 5,
    })
    notes = [f"Train-history tercile cut points across validation users: {q1:g}, {q2:g}."]
    return ValidationData(users, labels, duration, tabs, history_depth, list_length,
                          segments, notes)


def _load_predictions(path: Path, expected_rows: int) -> tuple[np.ndarray | None, str | None]:
    if not path.exists():
        return None, f"missing {path.name}"
    scores = np.full(expected_rows, np.nan, dtype=float)
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"row_id", "score"}
            if not required.issubset(reader.fieldnames or []):
                return None, f"predictions missing columns {sorted(required)}"
            for row in reader:
                row_id = int(row["row_id"])
                if 0 <= row_id < expected_rows:
                    scores[row_id] = float(row["score"])
    except (OSError, ValueError, csv.Error) as exc:
        return None, f"cannot read predictions: {exc}"
    missing = int(np.isnan(scores).sum())
    if missing:
        return None, f"predictions omit {missing}/{expected_rows} validation rows"
    if not np.all(np.isfinite(scores)):
        return None, "predictions contain non-finite scores"
    return scores, None


def _read_journal(run_dir: Path) -> tuple[list[dict], list[str]]:
    path = run_dir / "journal.jsonl"
    if not path.exists():
        return [], [f"Missing journal: {path}"]
    records, notes = [], []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            notes.append(f"Skipped malformed journal line {line_no}: {exc}")
    return records, notes


def load_nodes(run_dir: Path, expected_rows: int) -> tuple[list[NodeData], list[str]]:
    records, notes = _read_journal(run_dir)
    nodes: list[NodeData] = []
    for record in records:
        node_id = str(record.get("node_id", ""))
        if not node_id:
            notes.append("Skipped journal record without node_id")
            continue
        if node_id == "node_000":
            candidates = sorted(run_dir.glob("calib_seed*/predictions.csv"))
            seed_42 = run_dir / "calib_seed42" / "predictions.csv"
            pred_path = (seed_42 if seed_42.exists() else candidates[0]) if candidates else (
                run_dir / node_id / "predictions.csv"
            )
            metrics_path = pred_path.with_name("metrics.json")
        else:
            pred_path = run_dir / node_id / "predictions.csv"
            metrics_path = run_dir / node_id / "metrics.json"
        scores, note = _load_predictions(pred_path, expected_rows)
        payload = None
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                note = "; ".join(filter(None, [note, f"bad metrics.json: {exc}"]))
        else:
            note = "; ".join(filter(None, [note, "missing metrics.json"]))
        nodes.append(NodeData(run_dir, node_id, str(record.get("parent", "baseline")),
                              bool(record.get("accepted", False)), record, scores, payload, note))
        if note:
            notes.append(f"{node_id}: {note}")
    return nodes, notes


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float:
    """Kendall's tau-b, with two wholly tied identical rankings treated as unchanged."""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x) - 1):
        dx = np.sign(x[i] - x[i + 1:])
        dy = np.sign(y[i] - y[i + 1:])
        concordant += int(np.sum(dx * dy > 0))
        discordant += int(np.sum(dx * dy < 0))
        ties_x += int(np.sum((dx == 0) & (dy != 0)))
        ties_y += int(np.sum((dy == 0) & (dx != 0)))
    denominator = math.sqrt((concordant + discordant + ties_x) *
                            (concordant + discordant + ties_y))
    if denominator == 0:
        both_wholly_tied = (len(x) < 2 or (np.all(x == x[0]) and np.all(y == y[0])))
        return 1.0 if both_wholly_tied else 0.0
    return (concordant - discordant) / denominator


def ranking_change(users: np.ndarray, child: np.ndarray, parent: np.ndarray) -> tuple[float, float]:
    taus, changed = [], 0
    groups: dict[int, list[int]] = defaultdict(list)
    for index, user in enumerate(users.tolist()):
        groups[int(user)].append(index)
    for indices in groups.values():
        if len(indices) >= 2:
            taus.append(kendall_tau_b(child[indices], parent[indices]))
        # np.argmax is deterministic for ties and uses validation row order.
        changed += int(indices[int(np.argmax(child[indices]))] !=
                       indices[int(np.argmax(parent[indices]))])
    return (float(np.mean(taus)) if taus else float("nan"),
            100.0 * changed / len(groups) if groups else float("nan"))


def _average_ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        result[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return result


def rank_average(users: np.ndarray, model_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Per-user average of within-model ranks, returned as a higher-is-better score."""
    if not model_scores:
        raise ValueError("rank_average needs at least one model")
    result = np.empty(len(users), dtype=float)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, user in enumerate(users.tolist()):
        groups[int(user)].append(index)
    for indices in groups.values():
        ranks = np.vstack([_average_ranks_desc(scores[indices]) for scores in model_scores])
        result[indices] = -np.mean(ranks, axis=0)
    return result


def _curve_row(node: NodeData) -> dict:
    history = None
    record_metrics = node.record.get("metrics")
    if isinstance(record_metrics, dict):
        history = record_metrics.get("history")
    if not history and isinstance(node.metrics_payload, dict):
        history = node.metrics_payload.get("history")
    if not isinstance(history, list) or not history:
        return {"node": node.node_id, "peak_epoch": None, "peak": None, "final": None,
                "slope": None, "note": "history unavailable"}

    points = []
    for index, item in enumerate(history, 1):
        if not isinstance(item, dict):
            continue
        epoch = item.get("epoch", index)
        val = item.get("val_primary")
        if val is None and item.get("val_gauc") is not None and item.get("val_ndcg5") is not None:
            val = (float(item["val_gauc"]) + float(item["val_ndcg5"])) / 2.0
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            points.append((float(epoch), float(val), item.get("train_loss")))
    if not points:
        return {"node": node.node_id, "peak_epoch": None, "peak": None, "final": None,
                "slope": None, "note": "history has no usable validation metric"}
    peak = max(points, key=lambda point: point[1])
    losses = [(epoch, float(loss)) for epoch, _, loss in points[-3:]
              if isinstance(loss, (int, float)) and math.isfinite(float(loss))]
    slope = None
    if len(losses) >= 2:
        slope = float(np.polyfit([p[0] for p in losses], [p[1] for p in losses], 1)[0])
    return {"node": node.node_id, "peak_epoch": peak[0], "peak": peak[1],
            "final": points[-1][1], "slope": slope,
            "note": None if slope is not None else "train-loss slope unavailable"}


def _ensemble_results(nodes: Sequence[NodeData], metrics: dict[str, dict[str, float]],
                      changes: Sequence[RankingChange], validation: ValidationData,
                      name_prefix: str = "") -> tuple[list[EnsembleResult], str | None, dict | None]:
    usable = [node for node in nodes if node.scores is not None and node.key in metrics]
    if not usable:
        return [], None, None
    best = max(usable, key=lambda node: metrics[node.key]["primary"])
    best_metrics = metrics[best.key]
    high_change_keys = {change.node_key for change in changes if change.high_change}
    accepted = [node for node in usable if node.accepted]
    best_plus = [best] + [node for node in usable
                          if not node.accepted and node.key in high_change_keys and node.key != best.key]
    definitions = [
        ("all accepted nodes", accepted),
        ("best + high-change rejected nodes", best_plus),
        ("all nodes", usable),
    ]
    results = []
    for name, members in definitions:
        # Journal duplication should not overweight the exact same node object.
        unique = list(dict.fromkeys(node.key for node in members))
        selected = [next(node for node in members if node.key == key) for key in unique]
        if not selected:
            results.append(EnsembleResult(name_prefix + name, [], None, None,
                                          "no eligible nodes"))
            continue
        scores = rank_average(validation.users, [node.scores for node in selected if node.scores is not None])
        result_metrics = official_metrics(validation.users, validation.labels, scores)
        delta = result_metrics["primary"] - best_metrics["primary"]
        results.append(EnsembleResult(name_prefix + name, unique, result_metrics, delta))
    return results, best.key, best_metrics


def analyze(run_dir: Path, validation: ValidationData) -> RunAnalysis:
    nodes, notes = load_nodes(run_dir, len(validation.users))
    metrics: dict[str, dict[str, float]] = {}
    for node in nodes:
        if node.scores is not None:
            metrics[node.key] = official_metrics(validation.users, validation.labels, node.scores)

    baseline = next((node for node in nodes if node.node_id == "node_000" and node.scores is not None), None)
    segment_rows = []
    if baseline is None:
        notes.append("Segment metrics skipped: usable baseline node_000 predictions are unavailable.")
    else:
        for node in nodes:
            if node.scores is None:
                continue
            for segment, mask in validation.segments.items():
                rows = int(np.sum(mask))
                if rows == 0:
                    segment_rows.append({"node": node.node_id, "segment": segment, "rows": 0,
                                         "metrics": None, "delta": None, "flag": "empty segment"})
                    continue
                node_metrics = official_metrics(validation.users[mask], validation.labels[mask],
                                                node.scores[mask])
                base_metrics = official_metrics(validation.users[mask], validation.labels[mask],
                                                baseline.scores[mask])
                delta = node_metrics["primary"] - base_metrics["primary"]
                flag = "GATING/ENSEMBLE CANDIDATE" if (
                    not node.accepted and node.node_id != "node_000" and delta >= EPSILON
                ) else ""
                segment_rows.append({"node": node.node_id, "segment": segment, "rows": rows,
                                     "metrics": node_metrics, "delta": delta, "flag": flag})

    by_id = {node.node_id: node for node in nodes}
    changes = []
    for node in nodes:
        parent = by_id.get(node.parent)
        if node.node_id == "node_000" or node.scores is None:
            continue
        if parent is None:
            notes.append(f"{node.node_id}: ranking change skipped; parent {node.parent} is absent.")
            continue
        if parent.scores is None:
            notes.append(f"{node.node_id}: ranking change skipped; parent predictions are unavailable.")
            continue
        tau, top1 = ranking_change(validation.users, node.scores, parent.scores)
        delta = metrics[node.key]["primary"] - metrics[parent.key]["primary"]
        high_change = (math.isfinite(tau) and tau < HIGH_CHANGE_TAU) or top1 >= HIGH_CHANGE_TOP1_PCT
        flat = abs(delta) < EPSILON
        if flat and high_change:
            classification = "flat metrics + high change (ensemble candidate)"
        elif flat:
            classification = "flat metrics + low change (idea changed nothing)"
        elif high_change:
            classification = "metric change + high ranking change"
        else:
            classification = "metric change + low ranking change"
        changes.append(RankingChange(node.key, parent.key, tau, top1, delta,
                                     classification, high_change))

    ensembles, best_key, best_metrics = _ensemble_results(nodes, metrics, changes, validation)
    curves = [_curve_row(node) for node in nodes]
    return RunAnalysis(run_dir, nodes, metrics, segment_rows, changes, ensembles,
                       best_key, best_metrics, curves, notes + validation.segment_notes)


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
            for row in rows]
    return "\n".join([head, separator, *body])


def render_ensemble_section(title: str, ensembles: Sequence[EnsembleResult],
                            best_key: str | None, best_metrics: dict | None) -> str:
    lines = [f"## {title}", ""]
    if best_key is None or best_metrics is None:
        return "\n".join(lines + ["No usable node predictions were available."])
    lines.append(f"Best single: `{best_key}` — primary {_fmt(best_metrics['primary'])}.")
    lines.append("")
    rows = []
    prominent = []
    for result in ensembles:
        if result.metrics is None:
            rows.append((result.name, "—", "N/A", "N/A", "N/A", "N/A", result.note or ""))
            continue
        flag = "BEATS BEST >= 0.002" if result.delta is not None and result.delta >= EPSILON else ""
        if flag:
            prominent.append(f"**{result.name} beats the best single by {_fmt(result.delta, True)}.**")
        rows.append((result.name, len(result.members), _fmt(result.metrics["gauc"]),
                     _fmt(result.metrics["ndcg5"]), _fmt(result.metrics["primary"]),
                     _fmt(result.delta, True), flag))
    if prominent:
        lines.extend(prominent + [""])
    lines.append(_table(["ensemble", "members", "GAUC", "nDCG@5", "primary",
                         "delta vs best", "flag"], rows))
    lines.extend(["", "Members:"])
    for result in ensembles:
        lines.append(f"- {result.name}: {', '.join(f'`{key}`' for key in result.members) or 'none'}")
    return "\n".join(lines)


def render_report(analysis: RunAnalysis, cross_section: str | None = None) -> str:
    lines = [f"# Post-hoc analysis: {analysis.run_dir.name}", "",
             "Metrics are recomputed from validation predictions with `data/official/evaluate.py` conventions.",
             "Rank-change uses mean per-user Kendall tau-b; high change means tau < 0.90 or top-1 changes for at least 10% of users.", ""]
    if analysis.notes:
        lines.extend(["## Data availability notes", ""] + [f"- {note}" for note in analysis.notes] + [""])

    lines.extend(["## Segment metrics vs node_000", ""])
    if not analysis.segment_rows:
        lines.extend(["No segment metrics available.", ""])
    else:
        rows = []
        for row in analysis.segment_rows:
            met = row["metrics"]
            rows.append((row["node"], row["segment"], row["rows"],
                         _fmt(met["gauc"]) if met else "N/A",
                         _fmt(met["ndcg5"]) if met else "N/A",
                         _fmt(met["primary"]) if met else "N/A",
                         _fmt(row["delta"], True), row["flag"]))
        lines.extend([_table(["node", "segment", "rows", "GAUC", "nDCG@5", "primary",
                              "primary delta vs node_000", "flag"], rows), ""])

    lines.extend(["## Ranking change vs parent", ""])
    if analysis.ranking_changes:
        lines.extend([_table(["node", "parent", "mean Kendall tau-b", "top-1 changed",
                              "primary delta", "classification"],
                             ((change.node_key.split("/", 1)[1],
                               change.parent_key.split("/", 1)[1], _fmt(change.tau),
                               f"{change.top1_changed_pct:.2f}%", _fmt(change.metric_delta, True),
                               change.classification) for change in analysis.ranking_changes)), ""])
    else:
        lines.extend(["No parent/child prediction pairs were available.", ""])

    lines.extend([render_ensemble_section("Post-hoc ensembles", analysis.ensembles,
                                           analysis.best_key, analysis.best_metrics), ""])
    lines.extend(["## Curve stats", "",
                  "Train-loss slope is an ordinary least-squares slope over the final three usable epochs.", ""])
    lines.append(_table(["node", "val peak epoch", "val peak", "final val", "peak-final",
                         "train-loss slope at stop", "note"],
                        ((row["node"], _fmt(row["peak_epoch"]), _fmt(row["peak"]),
                          _fmt(row["final"]),
                          _fmt(row["peak"] - row["final"], True)
                          if row["peak"] is not None and row["final"] is not None else "N/A",
                          _fmt(row["slope"], True), row["note"] or "")
                         for row in analysis.curve_rows)))
    if cross_section:
        lines.extend(["", cross_section])
    return "\n".join(lines).rstrip() + "\n"


def cross_run_ensembles(analyses: Sequence[RunAnalysis], validation: ValidationData
                        ) -> tuple[list[EnsembleResult], str | None, dict | None]:
    nodes = [node for analysis in analyses for node in analysis.nodes if node.scores is not None]
    metrics = {key: value for analysis in analyses for key, value in analysis.metrics.items()}
    changes = [change for analysis in analyses for change in analysis.ranking_changes]
    return _ensemble_results(nodes, metrics, changes, validation, "cross-run: ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="single run directory (omit when using --runs)")
    parser.add_argument("--runs", nargs="+", type=Path,
                        help="two or more run directories sharing a validation split")
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if bool(args.run_dir) == bool(args.runs):
        parser.error("provide exactly one of RUN_DIR or --runs RUN_DIR [RUN_DIR ...]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = [args.run_dir] if args.run_dir else args.runs
    validation = load_validation_data(args.data_dir)
    analyses = [analyze(run_dir, validation) for run_dir in run_dirs]
    cross_section = None
    if len(analyses) > 1:
        ensembles, best_key, best_metrics = cross_run_ensembles(analyses, validation)
        cross_section = render_ensemble_section("Combined cross-run ensembles", ensembles,
                                                best_key, best_metrics)
    for analysis in analyses:
        report_dir = analysis.run_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        output = report_dir / "analysis.md"
        output.write_text(render_report(analysis, cross_section), encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
