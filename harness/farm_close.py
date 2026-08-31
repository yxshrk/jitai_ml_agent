"""Harness-owned three-phase executor for typed cross-family farm-close plans."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.official.evaluate import evaluate as official_evaluate


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = ("gauc", "ndcg5", "primary")
PREDICTION_COLUMNS = ("row_id", "user_id", "video_id", "score")
DEFAULT_ADMISSION_PRIMARY = 0.6040
FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
CONFIG_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
RESERVED_CONFIG_KEYS = {"data_dir", "out_dir", "seed"}
THREAD_CAP_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


FARM_CLOSE_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["members", "blend"],
    "additionalProperties": False,
    "properties": {
        "probe_epochs": {"type": "integer", "minimum": 1, "maximum": 2},
        "admission_primary": {
            "type": "number",
            "minimum": DEFAULT_ADMISSION_PRIMARY,
            "maximum": 1.0,
        },
        "members": {
            "type": "array",
            "minItems": 4,
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["family", "config", "seed"],
                "oneOf": [
                    {"required": ["script_source"]},
                    {"required": ["code"]},
                ],
                "additionalProperties": False,
                "properties": {
                    "family": {"type": "string", "pattern": FAMILY_PATTERN.pattern},
                    "script_source": {"type": "string", "minLength": 1},
                    "code": {"type": "string", "minLength": 1},
                    "config": {"type": "object"},
                    "seed": {"type": "integer"},
                },
            },
        },
        "blend": {
            "type": "object",
            "required": ["method", "scope"],
            "additionalProperties": False,
            "properties": {
                "method": {"const": "rank_average"},
                "scope": {"enum": ["per_user", "global"]},
            },
        },
    },
}


class FarmClosePlanError(ValueError):
    """A typed farm-close plan does not satisfy the harness schema."""


@dataclass
class PredictionVector:
    row_ids: np.ndarray
    users: np.ndarray
    videos: np.ndarray
    scores: np.ndarray


@dataclass
class MemberResult:
    index: int
    family: str
    seed: int
    config: dict[str, Any]
    out_dir: Path
    metrics: dict[str, Any]
    predictions: PredictionVector


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_plan(value: Any) -> dict[str, Any]:
    """Validate and normalize a farm-close plan against ``FARM_CLOSE_PLAN_SCHEMA``."""
    errors: list[str] = []
    if not isinstance(value, dict):
        raise FarmClosePlanError("farm_close_plan must be a JSON object")
    allowed_plan = {"members", "blend", "probe_epochs", "admission_primary"}
    extra_plan = set(value) - allowed_plan
    if extra_plan:
        errors.append(f"unknown plan fields: {sorted(extra_plan)}")

    members = value.get("members")
    if not isinstance(members, list) or not 4 <= len(members) <= 6:
        errors.append("members must be a list of 4 to 6 member specs")
        members = []
    normalized_members: list[dict[str, Any]] = []
    families: list[str] = []
    seeds: list[int] = []
    allowed_member = {"family", "script_source", "code", "config", "seed"}
    for index, member in enumerate(members):
        prefix = f"members[{index}]"
        if not isinstance(member, dict):
            errors.append(f"{prefix} must be an object")
            continue
        extra_member = set(member) - allowed_member
        if extra_member:
            errors.append(f"{prefix} has unknown fields: {sorted(extra_member)}")
        family = member.get("family")
        if not isinstance(family, str) or not FAMILY_PATTERN.fullmatch(family):
            errors.append(f"{prefix}.family must match {FAMILY_PATTERN.pattern!r}")
        else:
            families.append(family)
        seed = member.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            errors.append(f"{prefix}.seed must be an integer")
        else:
            seeds.append(seed)
        has_source = "script_source" in member
        has_code = "code" in member
        if has_source == has_code:
            errors.append(f"{prefix} must contain exactly one of script_source or code")
        if has_source and (
            not isinstance(member.get("script_source"), str)
            or not member["script_source"].strip()
        ):
            errors.append(f"{prefix}.script_source must be a non-empty string")
        if has_code and (
            not isinstance(member.get("code"), str) or not member["code"].strip()
        ):
            errors.append(f"{prefix}.code must be a non-empty string")
        config = member.get("config")
        if not isinstance(config, dict):
            errors.append(f"{prefix}.config must be an object of CLI dials")
            config = {}
        normalized_config: dict[str, Any] = {}
        for key, dial in config.items():
            if not isinstance(key, str) or not CONFIG_KEY_PATTERN.fullmatch(key):
                errors.append(f"{prefix}.config key {key!r} is not a safe CLI dial")
                continue
            if key in RESERVED_CONFIG_KEYS:
                errors.append(f"{prefix}.config cannot override {key}")
                continue
            if isinstance(dial, bool) or isinstance(dial, str) or _is_number(dial):
                normalized_config[key] = dial
            elif isinstance(dial, list) and dial and all(
                isinstance(item, str) or _is_number(item) for item in dial
            ):
                normalized_config[key] = list(dial)
            else:
                errors.append(
                    f"{prefix}.config.{key} must be a string, finite number, boolean, "
                    "or non-empty list of strings/numbers"
                )
        normalized = {
            "family": family,
            "config": normalized_config,
            "seed": seed,
        }
        if has_source:
            normalized["script_source"] = member.get("script_source", "").strip()
        if has_code:
            normalized["code"] = member.get("code", "")
        normalized_members.append(normalized)

    if len(set(families)) != len(families):
        errors.append("member families must be distinct")
    if len(set(seeds)) != len(seeds):
        errors.append("member seeds must be distinct")

    blend = value.get("blend")
    if not isinstance(blend, dict):
        errors.append("blend must be an object")
        blend = {}
    else:
        extra_blend = set(blend) - {"method", "scope"}
        if extra_blend:
            errors.append(f"blend has unknown fields: {sorted(extra_blend)}")
    if blend.get("method") != "rank_average":
        errors.append("blend.method must be 'rank_average'")
    if blend.get("scope") not in ("per_user", "global"):
        errors.append("blend.scope must be 'per_user' or 'global'")

    probe_epochs = value.get("probe_epochs", 2)
    if isinstance(probe_epochs, bool) or not isinstance(probe_epochs, int) or not 1 <= probe_epochs <= 2:
        errors.append("probe_epochs must be 1 or 2")
    admission = value.get("admission_primary", DEFAULT_ADMISSION_PRIMARY)
    if not _is_number(admission) or not DEFAULT_ADMISSION_PRIMARY <= float(admission) <= 1.0:
        errors.append(
            f"admission_primary must be between {DEFAULT_ADMISSION_PRIMARY:.4f} and 1.0"
        )

    if errors:
        raise FarmClosePlanError("; ".join(errors))
    return {
        "members": normalized_members,
        "blend": {"method": "rank_average", "scope": blend["scope"]},
        "probe_epochs": probe_epochs,
        "admission_primary": float(admission),
    }


def build_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the restricted child environment used by harness model subprocesses."""
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    for key in THREAD_CAP_KEYS:
        if os.environ.get(key):
            env[key] = os.environ[key]
    if extra:
        env.update(extra)
    return env


def _config_args(config: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in sorted(config):
        flag = "--" + key.replace("_", "-")
        value = config[key]
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif isinstance(value, list):
            result.append(flag)
            result.extend(str(item) for item in value)
        else:
            result.extend((flag, str(value)))
    return result


def read_predictions(path: Path) -> PredictionVector:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0]) != PREDICTION_COLUMNS:
        raise ValueError(f"{path} is empty or does not have the node prediction header")
    row_ids = np.fromiter((int(row["row_id"]) for row in rows), dtype=np.int64)
    users = np.fromiter((int(row["user_id"]) for row in rows), dtype=np.int64)
    videos = np.fromiter((int(row["video_id"]) for row in rows), dtype=np.int64)
    scores = np.fromiter((float(row["score"]) for row in rows), dtype=np.float64)
    if not np.array_equal(row_ids, np.arange(len(rows))):
        raise ValueError(f"{path} row_id values are not in validation-file order")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{path} contains non-finite scores")
    return PredictionVector(row_ids, users, videos, scores)


def _assert_aligned(reference: PredictionVector, other: PredictionVector, name: str) -> None:
    if not (
        np.array_equal(reference.row_ids, other.row_ids)
        and np.array_equal(reference.users, other.users)
        and np.array_equal(reference.videos, other.videos)
    ):
        raise ValueError(f"{name} prediction rows do not align with the other members")


def assert_member_distinctness(
    score_vectors: list[np.ndarray],
    names: list[str] | None = None,
    against: np.ndarray | None = None,
) -> None:
    """Assert pairwise score diversity and optional diversity from parent predictions."""
    labels = names or [f"member_{index}" for index in range(len(score_vectors))]
    for left, right in itertools.combinations(range(len(score_vectors)), 2):
        if np.allclose(score_vectors[left], score_vectors[right]):
            raise AssertionError(
                f"member-distinctness failed: {labels[left]} and {labels[right]} are allclose"
            )
    if against is not None:
        for label, scores in zip(labels, score_vectors):
            if np.allclose(scores, against):
                raise AssertionError(
                    f"member-distinctness failed: {label} is allclose to parent predictions"
                )


def _ordinal_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) > 1:
        ranks /= len(scores) - 1
    else:
        ranks.fill(0.5)
    return ranks


def blend_rank_average(
    users: np.ndarray,
    score_vectors: list[np.ndarray],
    scope: str,
) -> np.ndarray:
    """Rank-average aligned score vectors globally or independently within each user."""
    if len(score_vectors) < 2:
        raise ValueError("rank-average requires at least two score vectors")
    if scope not in ("per_user", "global"):
        raise ValueError("scope must be 'per_user' or 'global'")
    length = len(users)
    if any(np.asarray(scores).shape != (length,) for scores in score_vectors):
        raise ValueError("rank-average inputs must be aligned one-dimensional vectors")
    if scope == "global":
        ranks = [_ordinal_ranks(np.asarray(scores, dtype=np.float64)) for scores in score_vectors]
    else:
        groups = [np.flatnonzero(users == user) for user in np.unique(users)]
        ranks = []
        for scores in score_vectors:
            member_ranks = np.empty(length, dtype=np.float64)
            for indices in groups:
                member_ranks[indices] = _ordinal_ranks(np.asarray(scores)[indices])
            ranks.append(member_ranks)
    return np.mean(np.column_stack(ranks), axis=1)


def _validation_labels(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = data_dir / "val.npz"
    if npz_path.exists():
        with np.load(npz_path) as valid:
            return valid["user"].astype(np.int64), valid["y"].astype(np.int64)
    with (data_dir / "val.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.fromiter((int(row["user_id"]) for row in rows), dtype=np.int64),
        np.fromiter((int(row["long_view"]) for row in rows), dtype=np.int64),
    )


def _metrics(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    raw = official_evaluate(users.tolist(), labels.tolist(), scores.tolist())
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg5": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }


def _resolve_member_scripts(plan: dict[str, Any], out_dir: Path) -> list[Path]:
    scripts_dir = out_dir / "member_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, member in enumerate(plan["members"]):
        if "script_source" in member:
            path = Path(member["script_source"])
            path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
            if not path.is_relative_to(ROOT) or path.suffix != ".py" or not path.is_file():
                raise FarmClosePlanError(
                    f"members[{index}].script_source must name an existing Python file under {ROOT}"
                )
        else:
            path = scripts_dir / f"{index:02d}_{member['family']}.py"
            path.write_text(member["code"], encoding="utf-8")
        paths.append(path)
    return paths


def validate_script_sources(plan: dict[str, Any]) -> None:
    """Reject unresolved or out-of-repository member sources before node execution."""
    for index, member in enumerate(plan["members"]):
        if "script_source" not in member:
            continue
        path = Path(member["script_source"])
        path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
        if not path.is_relative_to(ROOT) or path.suffix != ".py" or not path.is_file():
            raise FarmClosePlanError(
                f"members[{index}].script_source must name an existing Python file under {ROOT}"
            )


def _run_member(
    index: int,
    member: dict[str, Any],
    script: Path,
    data_dir: Path,
    phase_dir: Path,
    seed_offset: int,
    timeout_s: float,
    smoke_epochs: int | None,
) -> MemberResult:
    member_dir = phase_dir / f"{index:02d}_{member['family']}"
    member_dir.mkdir(parents=True, exist_ok=True)
    for output_name in ("predictions.csv", "metrics.json"):
        output = member_dir / output_name
        if output.exists():
            output.unlink()
    seed = member["seed"] + seed_offset
    command = [
        sys.executable,
        str(script),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(member_dir),
        "--seed",
        str(seed),
        *_config_args(member["config"]),
    ]
    extra_env = {"SMOKE_EPOCHS": str(smoke_epochs)} if smoke_epochs is not None else None
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=build_subprocess_env(extra_env),
            capture_output=True,
            text=True,
            timeout=max(1.0, timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"member timed out after {timeout_s:.1f}s") from exc
    if completed.returncode != 0:
        detail = "\n".join(
            (completed.stderr or completed.stdout or "no child output").splitlines()[-30:]
        )
        raise RuntimeError(detail)
    metrics_path = member_dir / "metrics.json"
    predictions_path = member_dir / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise RuntimeError("member did not write metrics.json and predictions.csv")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in METRIC_KEYS:
        metric = metrics.get(key)
        if not _is_number(metric) or not 0.0 <= float(metric) <= 1.0:
            raise ValueError(f"invalid member metric {key}={metric!r}")
    return MemberResult(
        index=index,
        family=member["family"],
        seed=seed,
        config=dict(member["config"]),
        out_dir=member_dir,
        metrics=metrics,
        predictions=read_predictions(predictions_path),
    )


def _append_progress(path: Path, payload: dict[str, Any], lock: threading.Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_phase(
    phase: str,
    indexed_members: list[tuple[int, dict[str, Any]]],
    scripts: list[Path],
    data_dir: Path,
    out_dir: Path,
    seed_offset: int,
    deadline: float,
    smoke_epochs: int | None,
    progress_path: Path,
) -> tuple[list[MemberResult], list[dict[str, Any]]]:
    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    survivors: list[MemberResult] = []
    failures: list[dict[str, Any]] = []
    member_timeout = max(1.0, deadline - time.monotonic())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(indexed_members)) as executor:
        future_members = {
            executor.submit(
                _run_member,
                index,
                member,
                scripts[index],
                data_dir,
                phase_dir,
                seed_offset,
                member_timeout,
                smoke_epochs,
            ): (index, member)
            for index, member in indexed_members
        }
        for future in concurrent.futures.as_completed(future_members):
            index, member = future_members[future]
            base = {
                "phase": phase,
                "member": index,
                "family": member["family"],
                "seed": member["seed"] + seed_offset,
                "config": member["config"],
            }
            try:
                result = future.result()
            except Exception as exc:
                failure = {**base, "status": "failed", "error": str(exc)[-2000:]}
                failures.append(failure)
                _append_progress(progress_path, failure, lock)
            else:
                survivors.append(result)
                _append_progress(
                    progress_path,
                    {
                        **base,
                        "status": "completed",
                        "gauc": float(result.metrics["gauc"]),
                        "ndcg5": float(result.metrics["ndcg5"]),
                        "primary": float(result.metrics["primary"]),
                    },
                    lock,
                )
    survivors.sort(key=lambda result: result.index)
    failures.sort(key=lambda failure: failure["member"])
    return survivors, failures


def _combination_results(
    members: list[MemberResult],
    users: np.ndarray,
    labels: np.ndarray,
    scope: str,
) -> list[tuple[tuple[int, ...], dict[str, float], np.ndarray]]:
    results: list[tuple[tuple[int, ...], dict[str, float], np.ndarray]] = []
    for size in range(2, len(members) + 1):
        for positions in itertools.combinations(range(len(members)), size):
            scores = blend_rank_average(
                users,
                [members[position].predictions.scores for position in positions],
                scope,
            )
            results.append((positions, _metrics(users, labels, scores), scores))
    return results


def _best_combo(
    results: list[tuple[tuple[int, ...], dict[str, float], np.ndarray]],
    max_members: int | None = None,
) -> tuple[tuple[int, ...], dict[str, float], np.ndarray]:
    eligible = [row for row in results if max_members is None or len(row[0]) <= max_members]
    if not eligible:
        raise ValueError("no eligible blend combination")
    return max(eligible, key=lambda row: (row[1]["primary"], -len(row[0]), row[0]))


def _combo_manifest(
    results: list[tuple[tuple[int, ...], dict[str, float], np.ndarray]],
    members: list[MemberResult],
) -> list[dict[str, Any]]:
    return [
        {
            "families": [members[position].family for position in positions],
            **metrics,
        }
        for positions, metrics, _scores in results
    ]


def _member_manifest(result: MemberResult, admitted: bool) -> dict[str, Any]:
    return {
        "family": result.family,
        "seed": result.seed,
        "config": result.config,
        "gauc": float(result.metrics["gauc"]),
        "ndcg5": float(result.metrics["ndcg5"]),
        "primary": float(result.metrics["primary"]),
        "admitted_to_blends": admitted,
    }


def _history_entry(
    epoch: int,
    stage: str,
    metrics: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    child_history = metrics.get("history")
    last_child = child_history[-1] if isinstance(child_history, list) and child_history else {}
    return {
        "epoch": epoch,
        "stage": stage,
        "train_loss": last_child.get("train_loss"),
        "val_gauc": float(metrics["gauc"]),
        "val_ndcg5": float(metrics["ndcg5"]),
        "val_primary": float(metrics["primary"]),
        **extra,
    }


def run_plan(
    plan_value: Any,
    data_dir: Path,
    out_dir: Path,
    *,
    timeout_s: float,
    execution_seed: int,
    base_seed: int,
    parent_predictions: Path | None = None,
) -> dict[str, Any]:
    """Execute all three farm-close phases and emit one node-contract artifact."""
    plan = validate_plan(plan_value)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    data_dir = data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    scripts = _resolve_member_scripts(plan, out_dir)
    users, labels = _validation_labels(data_dir)
    seed_offset = execution_seed - base_seed
    started = time.monotonic()
    final_deadline = started + timeout_s
    probe_deadline = started + timeout_s * 0.4
    inherited_smoke = os.environ.get("SMOKE_EPOCHS")
    if inherited_smoke is not None:
        try:
            outer_cap = int(inherited_smoke)
        except ValueError as exc:
            raise ValueError("SMOKE_EPOCHS must be an integer") from exc
        if outer_cap <= 0:
            raise ValueError("SMOKE_EPOCHS must be positive")
    else:
        outer_cap = None
    probe_epochs = min(plan["probe_epochs"], outer_cap) if outer_cap is not None else plan["probe_epochs"]

    indexed = list(enumerate(plan["members"]))
    probe, probe_failures = _run_phase(
        "probe",
        indexed,
        scripts,
        data_dir,
        out_dir,
        seed_offset,
        probe_deadline,
        probe_epochs,
        progress_path,
    )
    if len(probe) < 2:
        raise RuntimeError(
            f"farm-close requires at least two probe survivors; got {len(probe)}"
        )
    reference = probe[0].predictions
    for member in probe[1:]:
        _assert_aligned(reference, member.predictions, member.family)
    if len(users) != len(reference.users) or not np.array_equal(users, reference.users):
        raise ValueError("probe predictions do not align with the validation split")
    if parent_predictions is not None and not parent_predictions.exists():
        raise FileNotFoundError(parent_predictions)
    parent = read_predictions(parent_predictions) if parent_predictions is not None else None
    if parent is not None:
        _assert_aligned(reference, parent, "parent")
    assert_member_distinctness(
        [member.predictions.scores for member in probe],
        [member.family for member in probe],
        parent.scores if parent is not None else None,
    )

    threshold = plan["admission_primary"]
    admitted_probe = [member for member in probe if float(member.metrics["primary"]) >= threshold]
    probe_combos = _combination_results(
        admitted_probe, users, labels, plan["blend"]["scope"]
    ) if len(admitted_probe) >= 2 else []
    if probe_combos:
        selected_positions, selected_probe_metrics, _ = _best_combo(probe_combos, max_members=3)
        selected = [admitted_probe[position] for position in selected_positions]
    else:
        selected = [max(probe, key=lambda member: float(member.metrics["primary"]))]
        selected_probe_metrics = {
            key: float(selected[0].metrics[key]) for key in METRIC_KEYS
        }
    selected_indices = {member.index for member in selected}
    full_specs = [(index, plan["members"][index]) for index in sorted(selected_indices)]
    full, full_failures = _run_phase(
        "full",
        full_specs,
        scripts,
        data_dir,
        out_dir,
        seed_offset,
        final_deadline,
        outer_cap,
        progress_path,
    )
    if not full:
        raise RuntimeError("all selected full-fidelity members failed")
    full_reference = full[0].predictions
    for member in full[1:]:
        _assert_aligned(full_reference, member.predictions, member.family)
    assert_member_distinctness(
        [member.predictions.scores for member in full],
        [member.family for member in full],
        parent.scores if parent is not None else None,
    )
    admitted_full = [member for member in full if float(member.metrics["primary"]) >= threshold]
    full_combos = _combination_results(
        admitted_full, users, labels, plan["blend"]["scope"]
    ) if len(admitted_full) >= 2 else []

    single_candidates = [
        ((position,), {key: float(member.metrics[key]) for key in METRIC_KEYS}, member.predictions.scores)
        for position, member in enumerate(full)
    ]
    blend_candidates = []
    full_position_by_index = {member.index: position for position, member in enumerate(full)}
    for positions, metrics, scores in full_combos:
        full_positions = tuple(
            full_position_by_index[admitted_full[position].index] for position in positions
        )
        blend_candidates.append((full_positions, metrics, scores))
    winning_positions, metrics, final_scores = max(
        [*single_candidates, *blend_candidates],
        key=lambda row: (row[1]["primary"], len(row[0]), row[0]),
    )
    winning = [full[position] for position in winning_positions]
    final_kind = "rank_average" if len(winning) >= 2 else "single_member"
    if parent is not None and np.allclose(final_scores, parent.scores):
        raise AssertionError("farm-close final predictions are allclose to parent predictions")

    history: list[dict[str, Any]] = []
    epoch = 0
    for member in probe:
        epoch += 1
        history.append(_history_entry(
            epoch,
            "probe_member",
            member.metrics,
            family=member.family,
            seed=member.seed,
            config=member.config,
        ))
    for member in full:
        epoch += 1
        history.append(_history_entry(
            epoch,
            "full_member",
            member.metrics,
            family=member.family,
            seed=member.seed,
            config=member.config,
        ))
    epoch += 1
    history.append(_history_entry(
        epoch,
        "full_reverified_blend" if len(winning) >= 2 else "full_best_single",
        metrics,
        families=[member.family for member in winning],
    ))
    result: dict[str, Any] = {
        **metrics,
        "history": history,
        "farm_close": {
            "plan": plan,
            "probe_members": [
                _member_manifest(member, member in admitted_probe) for member in probe
            ],
            "probe_failures": probe_failures,
            "probe_combinations": _combo_manifest(probe_combos, admitted_probe),
            "probe_selected_families": [member.family for member in selected],
            "probe_selected_metrics": selected_probe_metrics,
            "full_members": [
                _member_manifest(member, member in admitted_full) for member in full
            ],
            "full_failures": full_failures,
            "full_combinations": _combo_manifest(full_combos, admitted_full),
            "winning_families": [member.family for member in winning],
            "final_kind": final_kind,
            "degraded_to_single": len(winning) == 1,
            "admission_primary": threshold,
            "duration_s": round(time.monotonic() - started, 3),
        },
    }
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        writer.writerows(
            (int(row), int(user), int(video), f"{score:.12g}")
            for row, user, video, score in zip(
                full_reference.row_ids,
                full_reference.users,
                full_reference.videos,
                final_scores,
            )
        )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _append_progress(
        progress_path,
        {
            "phase": "final",
            "status": "completed",
            "kind": final_kind,
            "families": [member.family for member in winning],
            **metrics,
        },
        threading.Lock(),
    )
    return result


def render_plan_node(
    plan: dict[str, Any],
    *,
    parent_predictions: Path | None,
    base_seed: int,
) -> str:
    """Compile a validated plan into a runnable node-contract Python wrapper."""
    normalized = validate_plan(plan)
    plan_json = json.dumps(normalized, sort_keys=True)
    parent_text = str(parent_predictions) if parent_predictions is not None else None
    return f'''\
"""Harness-owned runnable wrapper for a typed farm-close plan."""
import argparse
import json
import os
from pathlib import Path

from harness.farm_close import run_plan

PLAN = json.loads({plan_json!r})
PARENT_PREDICTIONS = {parent_text!r}
BASE_SEED = {base_seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    timeout_s = float(os.environ.get("NODE_TIMEOUT_S", "7200"))
    run_plan(
        PLAN,
        args.data_dir,
        args.out_dir,
        timeout_s=timeout_s,
        execution_seed=args.seed,
        base_seed=BASE_SEED,
        parent_predictions=Path(PARENT_PREDICTIONS) if PARENT_PREDICTIONS else None,
    )


if __name__ == "__main__":
    main()
'''


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_plan(
        plan,
        args.data_dir,
        args.out_dir,
        timeout_s=args.timeout_s,
        execution_seed=args.seed,
        base_seed=args.seed,
        parent_predictions=args.parent_predictions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
