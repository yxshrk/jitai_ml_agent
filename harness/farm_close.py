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
DEFAULT_FULL_MEMBER_LIMIT = 3
DEFAULT_MIN_PROBE_BLEND_GAIN = 0.0
FAMILY_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
MEMBER_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
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
        "full_member_limit": {"type": "integer", "minimum": 2, "maximum": 3},
        "min_probe_blend_gain": {"type": "number", "minimum": 0.0},
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
                    "member_id": {"type": "string", "pattern": MEMBER_ID_PATTERN.pattern},
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
            "additionalProperties": False,
            "oneOf": [
                {
                    "required": ["method", "scope"],
                    "not": {"required": ["aggregations"]},
                },
                {
                    "required": ["aggregations"],
                    "not": {"anyOf": [
                        {"required": ["method"]},
                        {"required": ["scope"]},
                    ]},
                },
            ],
            "properties": {
                "method": {"const": "rank_average"},
                "scope": {"enum": ["per_user", "global"]},
                "weights": {"const": "equal"},
                "aggregations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "uniqueItems": True,
                    "items": {
                        "type": "object",
                        "required": ["method", "scope"],
                        "additionalProperties": False,
                        "properties": {
                            "method": {"const": "rank_average"},
                            "scope": {"enum": ["per_user", "global"]},
                        },
                    },
                },
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
    member_id: str | None = None


@dataclass
class Candidate:
    kind: str
    member_positions: tuple[int, ...]
    member_ids: tuple[str, ...]
    aggregation: dict[str, str] | None
    aggregation_order: int
    metrics: dict[str, float]
    scores: np.ndarray
    source_phase: str


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
    allowed_plan = {
        "members",
        "blend",
        "probe_epochs",
        "admission_primary",
        "full_member_limit",
        "min_probe_blend_gain",
    }
    extra_plan = set(value) - allowed_plan
    if extra_plan:
        errors.append(f"unknown plan fields: {sorted(extra_plan)}")

    members = value.get("members")
    if not isinstance(members, list) or not 4 <= len(members) <= 6:
        errors.append("members must be a list of 4 to 6 member specs")
        members = []
    normalized_members: list[dict[str, Any]] = []
    families: list[str] = []
    member_ids: list[str] = []
    seeds: list[int] = []
    allowed_member = {"member_id", "family", "script_source", "code", "config", "seed"}
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
        member_id = member.get("member_id", family)
        if not isinstance(member_id, str) or not MEMBER_ID_PATTERN.fullmatch(member_id):
            errors.append(f"{prefix}.member_id must match {MEMBER_ID_PATTERN.pattern!r}")
        else:
            member_ids.append(member_id)
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
            "member_id": member_id,
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
    if len(set(member_ids)) != len(member_ids):
        errors.append("member IDs must be distinct")
    if len(set(seeds)) != len(seeds):
        errors.append("member seeds must be distinct")

    blend = value.get("blend")
    if not isinstance(blend, dict):
        errors.append("blend must be an object")
        blend = {}
    else:
        extra_blend = set(blend) - {"method", "scope", "weights", "aggregations"}
        if extra_blend:
            errors.append(f"blend has unknown fields: {sorted(extra_blend)}")
    if blend.get("weights", "equal") != "equal":
        errors.append("blend.weights must be 'equal'")
    raw_aggregations = blend.get("aggregations")
    if raw_aggregations is None:
        raw_aggregations = [{"method": blend.get("method"), "scope": blend.get("scope")}]
    elif "method" in blend or "scope" in blend:
        errors.append("blend must use either method/scope or aggregations, not both")
    if not isinstance(raw_aggregations, list) or not 1 <= len(raw_aggregations) <= 2:
        errors.append("blend.aggregations must declare one or two rules")
        raw_aggregations = []
    aggregations: list[dict[str, str]] = []
    for index, aggregation in enumerate(raw_aggregations):
        if not isinstance(aggregation, dict) or set(aggregation) != {"method", "scope"}:
            errors.append(
                f"blend.aggregations[{index}] must contain only method and scope"
            )
            continue
        method = aggregation.get("method")
        scope = aggregation.get("scope")
        if method != "rank_average":
            errors.append(f"blend.aggregations[{index}].method must be 'rank_average'")
        if scope not in ("per_user", "global"):
            errors.append(
                f"blend.aggregations[{index}].scope must be 'per_user' or 'global'"
            )
        if method == "rank_average" and scope in ("per_user", "global"):
            aggregations.append({"method": method, "scope": scope})
    aggregation_keys = [(row["method"], row["scope"]) for row in aggregations]
    if len(set(aggregation_keys)) != len(aggregation_keys):
        errors.append("blend aggregation rules must be distinct")

    probe_epochs = value.get("probe_epochs", 2)
    if isinstance(probe_epochs, bool) or not isinstance(probe_epochs, int) or not 1 <= probe_epochs <= 2:
        errors.append("probe_epochs must be 1 or 2")
    admission = value.get("admission_primary", DEFAULT_ADMISSION_PRIMARY)
    if not _is_number(admission) or not DEFAULT_ADMISSION_PRIMARY <= float(admission) <= 1.0:
        errors.append(
            f"admission_primary must be between {DEFAULT_ADMISSION_PRIMARY:.4f} and 1.0"
        )
    full_member_limit = value.get("full_member_limit", DEFAULT_FULL_MEMBER_LIMIT)
    if (isinstance(full_member_limit, bool) or not isinstance(full_member_limit, int)
            or not 2 <= full_member_limit <= 3):
        errors.append("full_member_limit must be 2 or 3")
    min_probe_blend_gain = value.get(
        "min_probe_blend_gain", DEFAULT_MIN_PROBE_BLEND_GAIN
    )
    if not _is_number(min_probe_blend_gain) or float(min_probe_blend_gain) < 0:
        errors.append("min_probe_blend_gain must be a finite non-negative number")

    if errors:
        raise FarmClosePlanError("; ".join(errors))
    return {
        "members": normalized_members,
        "blend": {"weights": "equal", "aggregations": aggregations},
        "probe_epochs": probe_epochs,
        "admission_primary": float(admission),
        "full_member_limit": full_member_limit,
        "min_probe_blend_gain": float(min_probe_blend_gain),
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
        member_id=member["member_id"],
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


def _member_id(member: MemberResult) -> str:
    return member.member_id or member.family


def _declared_candidate_count(member_count: int, aggregation_count: int) -> int:
    """Raw singletons plus every multi-member subset for every declared rule."""
    if member_count <= 0:
        return 0
    return member_count + aggregation_count * (2**member_count - member_count - 1)


def _candidate_results(
    members: list[MemberResult],
    users: np.ndarray,
    labels: np.ndarray,
    aggregations: list[dict[str, str]],
    source_phase: str,
) -> list[Candidate]:
    candidates = [
        Candidate(
            kind="singleton",
            member_positions=(position,),
            member_ids=(_member_id(member),),
            aggregation=None,
            aggregation_order=-1,
            metrics=_metrics(users, labels, member.predictions.scores),
            scores=member.predictions.scores,
            source_phase=source_phase,
        )
        for position, member in enumerate(members)
    ]
    for aggregation_order, aggregation in enumerate(aggregations):
        for size in range(2, len(members) + 1):
            for positions in itertools.combinations(range(len(members)), size):
                scores = blend_rank_average(
                    users,
                    [members[position].predictions.scores for position in positions],
                    aggregation["scope"],
                )
                candidates.append(Candidate(
                    kind="blend",
                    member_positions=positions,
                    member_ids=tuple(sorted(_member_id(members[p]) for p in positions)),
                    aggregation=dict(aggregation),
                    aggregation_order=aggregation_order,
                    metrics=_metrics(users, labels, scores),
                    scores=scores,
                    source_phase=source_phase,
                ))
    return candidates


def _candidate_sort_key(candidate: Candidate, *, incumbent_first: bool) -> tuple:
    return (
        -float(candidate.metrics["primary"]),
        0 if incumbent_first and candidate.kind == "incumbent" else 1,
        len(candidate.member_ids),
        candidate.aggregation_order,
        candidate.member_ids,
    )


def _choose_candidate(
    candidates: list[Candidate], *, incumbent_first: bool = False
) -> Candidate:
    if not candidates:
        raise ValueError("no eligible candidate")
    return min(
        candidates,
        key=lambda candidate: _candidate_sort_key(
            candidate, incumbent_first=incumbent_first
        ),
    )


def select_probe_portfolio(
    candidates: list[Candidate],
    members: list[MemberResult],
    *,
    full_member_limit: int,
    min_probe_blend_gain: float,
) -> tuple[Candidate, Candidate | None, Candidate | None, Candidate, float | None]:
    """Apply the deterministic anchor-constrained probe promotion policy."""
    singletons = [candidate for candidate in candidates if candidate.kind == "singleton"]
    anchor = _choose_candidate(singletons)
    blends = [candidate for candidate in candidates if candidate.kind == "blend"]
    unconstrained = _choose_candidate(blends) if blends else None
    anchor_id = anchor.member_ids[0]
    constrained_candidates = [
        candidate
        for candidate in blends
        if anchor_id in candidate.member_ids
        and 2 <= len(candidate.member_ids) <= full_member_limit
        and len({members[p].family for p in candidate.member_positions})
        == len(candidate.member_positions)
    ]
    constrained = (
        _choose_candidate(constrained_candidates) if constrained_candidates else None
    )
    gain = (
        float(constrained.metrics["primary"]) - float(anchor.metrics["primary"])
        if constrained is not None else None
    )
    selected = (
        constrained
        if constrained is not None
        and gain is not None
        and gain > min_probe_blend_gain
        else anchor
    )
    return anchor, unconstrained, constrained, selected, gain


def select_full_candidate(candidates: list[Candidate]) -> Candidate:
    """Select by official primary, retaining the incumbent on an exact tie."""
    return _choose_candidate(candidates, incumbent_first=True)


def _candidate_manifest(
    candidates: list[Candidate], members: list[MemberResult]
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        selected = [members[position] for position in candidate.member_positions]
        rows.append({
            "kind": candidate.kind,
            "member_ids": list(candidate.member_ids),
            "families": [member.family for member in selected],
            "weights": (
                [1.0 / len(selected)] * len(selected) if selected else []
            ),
            "aggregation": candidate.aggregation or {"method": "identity"},
            **candidate.metrics,
        })
    return rows


def _drop_duplicate_members(
    members: list[MemberResult],
) -> tuple[list[MemberResult], list[dict[str, str]]]:
    retained: list[MemberResult] = []
    dropped: list[dict[str, str]] = []
    for member in sorted(members, key=_member_id):
        duplicate = next(
            (
                prior for prior in retained
                if np.array_equal(member.predictions.scores, prior.predictions.scores)
            ),
            None,
        )
        if duplicate is None:
            retained.append(member)
        else:
            dropped.append({
                "member_id": _member_id(member),
                "family": member.family,
                "reason": "duplicate_prediction_vector",
                "duplicate_of": _member_id(duplicate),
            })
    retained.sort(key=lambda member: member.index)
    return retained, dropped


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _member_manifest(result: MemberResult, admitted: bool) -> dict[str, Any]:
    return {
        "member_id": _member_id(result),
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


def _artifact_path(path: Path, out_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(out_dir.resolve()))
    except ValueError:
        return str(resolved)


def _freeze_recipe(
    candidate: Candidate,
    members: list[MemberResult],
    out_dir: Path,
    parent_predictions: Path | None,
) -> dict[str, Any]:
    if candidate.source_phase not in ("full", "incumbent"):
        raise AssertionError("a probe candidate cannot be frozen as the emitted artifact")
    selected = sorted(
        (members[position] for position in candidate.member_positions),
        key=_member_id,
    )
    recipe = {
        "schema_version": "farm-close.recipe.v1",
        "source_phase": candidate.source_phase,
        "candidate_kind": candidate.kind,
        "members": [
            {
                "member_id": _member_id(member),
                "family": member.family,
                "seed": member.seed,
                "config": member.config,
                "phase": "full",
                "predictions_path": _artifact_path(
                    member.out_dir / "predictions.csv", out_dir
                ),
            }
            for member in selected
        ],
        "weights": ([1.0 / len(selected)] * len(selected) if selected else []),
        "aggregation": candidate.aggregation or {"method": "identity"},
    }
    if candidate.kind == "incumbent":
        if parent_predictions is None:
            raise AssertionError("incumbent recipe requires parent predictions")
        recipe["incumbent_predictions_path"] = str(parent_predictions.resolve())

    # Checkpoint persistence is intentionally out of scope here: member scripts are
    # deterministic for a recorded seed, so later model replay reruns these exact
    # member configurations. This verification replay uses their saved full vectors.
    _write_canonical_json(out_dir / "recipe.json", recipe)
    return recipe


def _recipe_path(path_text: str, out_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else out_dir / path


def _replay_frozen_recipe(
    recipe: dict[str, Any],
    out_dir: Path,
    users: np.ndarray,
    labels: np.ndarray,
) -> tuple[PredictionVector, dict[str, float]]:
    if recipe["source_phase"] == "incumbent":
        prediction = read_predictions(
            _recipe_path(recipe["incumbent_predictions_path"], out_dir)
        )
    else:
        if not recipe["members"] or any(
            member.get("phase") != "full" for member in recipe["members"]
        ):
            raise AssertionError("frozen farm-close members must all be full-fidelity")
        vectors = [
            read_predictions(_recipe_path(member["predictions_path"], out_dir))
            for member in recipe["members"]
        ]
        prediction = vectors[0]
        for member, vector in zip(recipe["members"][1:], vectors[1:]):
            _assert_aligned(prediction, vector, member["member_id"])
        if len(vectors) == 1:
            scores = vectors[0].scores.copy()
        else:
            aggregation = recipe["aggregation"]
            if aggregation.get("method") != "rank_average":
                raise AssertionError("unsupported frozen aggregation")
            scores = blend_rank_average(
                prediction.users,
                [vector.scores for vector in vectors],
                aggregation["scope"],
            )
        prediction = PredictionVector(
            prediction.row_ids.copy(),
            prediction.users.copy(),
            prediction.videos.copy(),
            scores,
        )
    if len(users) != len(prediction.users) or not np.array_equal(users, prediction.users):
        raise ValueError("frozen recipe predictions do not align with validation")
    return prediction, _metrics(users, labels, prediction.scores)


def _write_predictions(path: Path, predictions: PredictionVector) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_COLUMNS)
        writer.writerows(
            (int(row), int(user), int(video), f"{score:.17g}")
            for row, user, video, score in zip(
                predictions.row_ids,
                predictions.users,
                predictions.videos,
                predictions.scores,
            )
        )


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
    aggregation_count = len(plan["blend"]["aggregations"])
    planned_probe_candidate_count = _declared_candidate_count(
        len(plan["members"]), aggregation_count
    )
    resolved_plan = {
        **plan,
        "enumeration": {
            "weights": "equal",
            "aggregation_rule_count": aggregation_count,
            "probe_candidate_count": planned_probe_candidate_count,
            "full_candidate_count_upper_bound": _declared_candidate_count(
                plan["full_member_limit"], aggregation_count
            ),
        },
    }
    _write_canonical_json(out_dir / "plan.requested.json", plan_value)
    _write_canonical_json(out_dir / "plan.resolved.json", resolved_plan)
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
    raw_probe, probe_failures = _run_phase(
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
    if not raw_probe:
        raise RuntimeError("farm-close requires at least one valid probe survivor")
    reference = raw_probe[0].predictions
    for member in raw_probe[1:]:
        _assert_aligned(reference, member.predictions, member.family)
    if len(users) != len(reference.users) or not np.array_equal(users, reference.users):
        raise ValueError("probe predictions do not align with the validation split")
    if parent_predictions is not None and not parent_predictions.exists():
        raise FileNotFoundError(parent_predictions)
    parent = read_predictions(parent_predictions) if parent_predictions is not None else None
    if parent is not None:
        _assert_aligned(reference, parent, "parent")

    probe, probe_dropped = _drop_duplicate_members(raw_probe)
    probe_candidates = _candidate_results(
        probe,
        users,
        labels,
        plan["blend"]["aggregations"],
        "probe",
    )
    if len(probe_candidates) != _declared_candidate_count(
        len(probe), aggregation_count
    ):
        raise AssertionError("probe enumeration was not completed")
    probe_blends = [
        candidate for candidate in probe_candidates if candidate.kind == "blend"
    ]
    (
        anchor,
        unconstrained_probe_winner,
        constrained_probe_winner,
        selected_probe_candidate,
        probe_gain,
    ) = select_probe_portfolio(
        probe_candidates,
        probe,
        full_member_limit=plan["full_member_limit"],
        min_probe_blend_gain=plan["min_probe_blend_gain"],
    )
    selected = [probe[position] for position in selected_probe_candidate.member_positions]
    selected_probe_metrics = dict(selected_probe_candidate.metrics)
    selected_indices = {member.index for member in selected}
    full_specs = [(index, plan["members"][index]) for index in sorted(selected_indices)]
    promotion_dropped = [
        {
            "member_id": _member_id(member),
            "family": member.family,
            "reason": "not_selected_by_anchor_constrained_probe_policy",
        }
        for member in probe
        if member.index not in selected_indices
    ]
    probe_audit = {
        "planned_candidate_count": planned_probe_candidate_count,
        "evaluated_candidate_count": len(probe_candidates),
        "candidates": _candidate_manifest(probe_candidates, probe),
        "anchor": _candidate_manifest([anchor], probe)[0],
        "unconstrained_winner": (
            _candidate_manifest([unconstrained_probe_winner], probe)[0]
            if unconstrained_probe_winner is not None else None
        ),
        "constrained_winner": (
            _candidate_manifest([constrained_probe_winner], probe)[0]
            if constrained_probe_winner is not None else None
        ),
        "selected_for_full": _candidate_manifest([selected_probe_candidate], probe)[0],
        "gain_over_anchor": probe_gain,
        "min_probe_blend_gain": plan["min_probe_blend_gain"],
        "failed_members": probe_failures,
        "dropped_members": [*probe_dropped, *promotion_dropped],
    }
    _write_canonical_json(out_dir / "probe_candidates.json", probe_audit)

    planned_full_candidate_count = _declared_candidate_count(
        len(full_specs), aggregation_count
    ) + (1 if parent is not None else 0)
    raw_full, full_failures = _run_phase(
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
    if not raw_full and parent is None:
        raise RuntimeError("all selected full-fidelity members failed and no incumbent exists")
    if raw_full:
        full_reference = raw_full[0].predictions
        for member in raw_full[1:]:
            _assert_aligned(full_reference, member.predictions, member.family)
        if len(users) != len(full_reference.users) or not np.array_equal(
            users, full_reference.users
        ):
            raise ValueError("full predictions do not align with the validation split")
    full, full_dropped = _drop_duplicate_members(raw_full)
    full_candidates = _candidate_results(
        full,
        users,
        labels,
        plan["blend"]["aggregations"],
        "full",
    )
    if len(full_candidates) != _declared_candidate_count(
        len(full), aggregation_count
    ):
        raise AssertionError("full enumeration was not completed")
    if parent is not None:
        full_candidates.append(Candidate(
            kind="incumbent",
            member_positions=(),
            member_ids=("incumbent",),
            aggregation=None,
            aggregation_order=-1,
            metrics=_metrics(users, labels, parent.scores),
            scores=parent.scores,
            source_phase="incumbent",
        ))
    winner = select_full_candidate(full_candidates)
    selected_table_metrics = dict(winner.metrics)
    recipe = _freeze_recipe(winner, full, out_dir, parent_predictions)
    final_predictions, metrics = _replay_frozen_recipe(recipe, out_dir, users, labels)
    winning = [full[position] for position in winner.member_positions]
    if winner.kind == "incumbent":
        final_kind = "incumbent"
    elif winner.kind == "singleton":
        final_kind = "single_member"
    else:
        final_kind = "rank_average"
    fallback_to_singleton = winner.kind == "singleton"
    fallback_to_incumbent = winner.kind == "incumbent"
    full_audit = {
        "planned_candidate_count": planned_full_candidate_count,
        "evaluated_candidate_count": len(full_candidates),
        "candidates": _candidate_manifest(full_candidates, full),
        "winner": _candidate_manifest([winner], full)[0],
        "failed_members": full_failures,
        "dropped_members": full_dropped,
    }
    _write_canonical_json(out_dir / "full_candidates.json", full_audit)

    history: list[dict[str, Any]] = []
    epoch = 0
    for member in raw_probe:
        epoch += 1
        history.append(_history_entry(
            epoch,
            "probe_member",
            member.metrics,
            family=member.family,
            seed=member.seed,
            config=member.config,
        ))
    for member in raw_full:
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
        (
            "full_fallback_incumbent" if fallback_to_incumbent
            else "full_best_single" if fallback_to_singleton
            else "full_reverified_blend"
        ),
        metrics,
        families=[member.family for member in winning],
    ))
    result: dict[str, Any] = {
        **metrics,
        "fallback_to_singleton": fallback_to_singleton,
        "fallback_to_incumbent": fallback_to_incumbent,
        "history": history,
        "farm_close": {
            "plan": plan,
            "probe_members": [
                _member_manifest(member, _member_id(member) in {
                    _member_id(retained) for retained in probe
                }) for member in raw_probe
            ],
            "probe_failures": probe_failures,
            "probe_dropped": probe_dropped,
            "probe_combinations": _candidate_manifest(probe_blends, probe),
            "probe_candidate_count": len(probe_candidates),
            "probe_anchor": _candidate_manifest([anchor], probe)[0],
            "probe_unconstrained_winner": (
                _candidate_manifest([unconstrained_probe_winner], probe)[0]
                if unconstrained_probe_winner is not None else None
            ),
            "probe_gain_over_anchor": probe_gain,
            "probe_selected_families": [member.family for member in selected],
            "probe_selected_metrics": selected_probe_metrics,
            "full_members": [
                _member_manifest(member, _member_id(member) in {
                    _member_id(retained) for retained in full
                }) for member in raw_full
            ],
            "full_failures": full_failures,
            "full_dropped": full_dropped,
            "full_combinations": _candidate_manifest(
                [candidate for candidate in full_candidates if candidate.kind == "blend"],
                full,
            ),
            "full_candidate_count": len(full_candidates),
            "selected_table_metrics": selected_table_metrics,
            "verified_metrics": metrics,
            "winning_families": [member.family for member in winning],
            "final_kind": final_kind,
            "degraded_to_single": fallback_to_singleton,
            "fallback_to_singleton": fallback_to_singleton,
            "fallback_to_incumbent": fallback_to_incumbent,
            "admission_primary": plan["admission_primary"],
            "member_counts": {
                "probe_requested": len(plan["members"]),
                "probe_succeeded": len(raw_probe),
                "probe_failed": len(probe_failures),
                "probe_duplicate_dropped": len(probe_dropped),
                "probe_promotion_dropped": len(promotion_dropped),
                "full_requested": len(full_specs),
                "full_succeeded": len(raw_full),
                "full_failed": len(full_failures),
                "full_duplicate_dropped": len(full_dropped),
                "full_selection_dropped": len(full) - len(winning),
            },
            "enumeration": {
                "probe_planned": planned_probe_candidate_count,
                "probe_evaluated": len(probe_candidates),
                "full_planned": planned_full_candidate_count,
                "full_evaluated": len(full_candidates),
            },
            "duration_s": round(time.monotonic() - started, 3),
        },
    }
    _write_predictions(out_dir / "predictions.csv", final_predictions)
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
            "fallback_to_singleton": fallback_to_singleton,
            "fallback_to_incumbent": fallback_to_incumbent,
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
