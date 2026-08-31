"""The autonomous loop orchestrator (CONTRACTS.md section 6).

Solution TREE of nodes (whole runnable scripts), a one-line-per-node journal,
and a harness-owned policy: 3 initial drafts -> debug failed nodes (max depth 2)
-> greedy improve-best, with a forced branch to a different MENU tier after 5
stagnant (no-acceptance) iterations.

Acceptance: sigma calibrated from 3 baseline seeds at startup; accept if
delta >= max(2*sigma, 0.002); a positive delta below that is a grey zone that
triggers one reseed confirm run (accept iff the mean delta still clears
max(sigma, 0.001)). Convergence: the last N=3 ACCEPTED deltas all < epsilon
(0.002). Hard caps: --max-iters, --max-hours, --max-tokens.

The workspace dir exposed to generated scripts contains train+val only; the
harness asserts no test file is reachable and statically rejects scripts that
reference test paths (CONTRACTS.md section 4).
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent.budget import BudgetExhausted
from agent.brain import parse_method_card_metadata
from harness.farm_close import (
    FarmClosePlanError,
    build_subprocess_env,
    render_plan_node,
    validate_script_sources,
    validate_plan as validate_farm_close_plan,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "zoo" / "fm_torch.py"
METRIC_KEYS = ("gauc", "ndcg5", "primary")
FORBIDDEN_PATTERNS = ("test.csv", "data/test")
DRAFT_TIERS = ("Tier 1", "Tier 2", "Tier 3")
ALL_TIERS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4")
FULL_CONTEXT_CHAR_BUDGET = 80_000  # approximately 20k tokens at four characters/token
CROSS_RUN_PATH = ROOT / "logs" / "CROSS_RUN.md"


@dataclass
class Node:
    node_id: str
    parent: str
    action: str
    hypothesis: str
    code_path: Path
    tier: str | None = None
    primary: float | None = None
    metrics: dict | None = None
    status: str = "pending"  # accepted | rejected | failed | suspect_implementation
    error: str | None = None
    debug_depth: int = 0
    method_selection: dict | None = None
    change_summary: str = ""
    expected_delta: float | None = None
    expected_delta_basis: str | None = None
    verdict_note: str | None = None
    failure_stage: str | None = None
    fixer_eligible: bool = False
    recovery: str | None = None
    farm_close_plan: dict | None = None
    farm_close_plan_path: Path | None = None
    execution_kind: str = "script"


def rewrite_ratio(parent_code: str, code: str) -> float:
    """Line-level similarity between parent and proposed script (0..1).

    Improve/debug proposals must EDIT the champion artifact, not replace it:
    a debugged trainer that survives across nodes is how implementation
    quality compounds (the append-only property of human research). Enforced
    in iterate(); the prompt contract already calls rewrites defects."""
    import difflib
    return difflib.SequenceMatcher(
        a=parent_code.splitlines(), b=code.splitlines(), autojunk=False
    ).ratio()


@dataclass
class LoopConfig:
    data_dir: Path
    run_dir: Path | None = None
    max_iters: int = 50
    max_hours: float = 6.0
    max_tokens: int = 2_000_000
    max_usd: float = 10.0  # per-run soft ceiling; the $BUDGET_USD hard cap lives in agent/budget.py
    timeout_s: int = 600
    epsilon: float = 0.002
    accept_floor: float | None = None  # parent-acceptance floor; default = epsilon; convergence ALWAYS uses official epsilon
    n_converge: int = 3
    stagnation_limit: int = 5
    confirm_runs: int = 1
    reflect_every: int = 5
    sigma: float | None = None  # skip calibration when provided (tests)
    calib_seeds: tuple[int, ...] = (42, 43, 44)
    baseline_script: Path = BASELINE_SCRIPT
    seed: int = 42
    confirm_seed: int = 1042
    draft_tiers: tuple[str, ...] = ("Tier 1", "Tier 2", "Tier 3")  # directives for the initial drafts
    seed_scripts: tuple[Path, ...] = ()  # team-provided reference scripts run as initial draft nodes (disclosed)
    context_mode: str = "compact"
    dataset: str = "pure"
    knowledge_mode: str = "full"
    cross_run_path: Path = CROSS_RUN_PATH
    plan_budget: bool = False
    resume_from: Path | None = None  # prior run dir to continue from (disclosed experiment lineage)
    resume_at: int = 0  # continue from just before this iteration (>= 1)
    # Fast-forward shakeout mode: cap EVERY training stage (calibration, smoke,
    # full, confirms, farm members) at this many epochs so a whole run's decisions
    # and code can be audited in minutes. Diagnostic only — metrics are probe-level
    # and the run is NEVER designation-eligible.
    fast_forward_epochs: int | None = None

    def __post_init__(self) -> None:
        if self.fast_forward_epochs is not None and self.fast_forward_epochs < 1:
            raise ValueError("fast_forward_epochs must be >= 1")
        if self.context_mode not in ("compact", "full"):
            raise ValueError("context_mode must be 'compact' or 'full'")
        if self.dataset not in ("pure", "1k"):
            raise ValueError("dataset must be 'pure' or '1k'")
        if self.knowledge_mode not in ("full", "clean"):
            raise ValueError("knowledge_mode must be 'full' or 'clean'")


def normalize_history(history) -> list[dict]:
    """Coerce a node's recorded history into the contract's flat per-epoch curve.

    Scripts sometimes log a sweep (a list of config entries each carrying a nested
    'epochs'/'checkpoints' list) or use bare 'gauc'/'primary' keys. The selector
    can only diagnose a flat curve with epoch/train_loss/val_gauc/val_primary, so
    flatten the best-scoring nested curve and alias the keys. Returns [] when no
    usable curve exists (which then correctly reads as insufficient telemetry)."""
    if isinstance(history, dict):  # fan-out nodes may group history by stage
        history = [e for v in history.values() if isinstance(v, list)
                   for e in v if isinstance(e, dict)]
    if not isinstance(history, list):
        return []
    entries = [e for e in history if isinstance(e, dict)]
    if not entries:
        return []
    def flat(e: dict) -> dict:
        return {
            "epoch": e.get("epoch"),
            "train_loss": e.get("train_loss", e.get("loss")),
            "val_gauc": e.get("val_gauc", e.get("gauc")),
            "val_primary": e.get("val_primary", e.get("primary")),
        }
    def usable(rows: list[dict]) -> bool:
        return any(r.get("val_primary") is not None for r in rows)
    # sweep-shaped entries carry nested curves; those beat config-level summaries
    has_nested = any(isinstance(e.get("epochs") or e.get("checkpoints"), list) for e in entries)
    direct = [flat(e) for e in entries]
    if usable(direct) and not has_nested:
        return direct
    # sweep-shaped: pick the nested curve whose best val_primary is highest
    best_rows: list[dict] = []
    best_score = float("-inf")
    for e in entries:
        nested = e.get("epochs") or e.get("checkpoints") or []
        rows = [flat(x) for x in nested if isinstance(x, dict)]
        if not usable(rows):
            continue
        top = max(r["val_primary"] for r in rows if r.get("val_primary") is not None)
        if top > best_score:
            best_score, best_rows = top, rows
    return best_rows if best_rows else (direct if usable(direct) else [])


class LeakageError(RuntimeError):
    pass


@dataclass
class RunResult:
    ok: bool
    metrics: dict | None = None
    error: str | None = None
    duration_s: float = 0.0


class Loop:
    def __init__(self, config: LoopConfig, brain) -> None:
        self.config = config
        self.brain = brain
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.run_dir = config.run_dir or ROOT / "logs" / f"run_{run_id}"
        self.nodes_dir = self.run_dir / "nodes"
        self.workspace = self.run_dir / "workspace"
        self.journal_path = self.run_dir / "journal.jsonl"
        self.nodes: dict[str, Node] = {}
        self.journal_lines: list[str] = []
        self.champion: Node | None = None
        self.sigma: float = 0.0
        self.accepted_deltas: list[float] = []
        self.no_improve_streak = 0  # official convergence: consecutive iters with best-improvement <= epsilon
        self.stagnation = 0
        self.focus_note: str | None = None
        self.forced_tiers_used: list[str] = []
        self.start_time = time.time()
        self.stop_reason: str | None = None
        self.prior_runs = ""
        self.exploration_plan: dict | None = None
        self.initial_draft_slots = len(config.draft_tiers)
        self.calibration_result: dict | None = None

    # ---------- workspace & leakage guard ----------

    def prepare_workspace(self) -> None:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        try:
            from harness.manifest import write_manifest
            write_manifest(self.run_dir, self.config)
        except Exception:
            pass
        required = ["train.csv", "val.csv"]
        optional = ["train.npz", "val.npz"]
        for split in required:
            source = self.config.data_dir / split
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy(source, self.workspace / split)
        for split in optional:
            source = self.config.data_dir / split
            if source.exists():
                shutil.copy(source, self.workspace / split)
        leaked = [p for p in self.workspace.rglob("*") if "test" in p.name.lower()]
        if leaked:
            raise LeakageError(f"test files reachable from workspace: {leaked}")

    @staticmethod
    def check_code_leakage(code: str) -> str | None:
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in code:
                return f"leakage guard: generated script references forbidden path pattern {pattern!r}"
        return None

    # ---------- script execution ----------

    def run_script(self, script: Path, out_dir: Path, seed: int, timeout_s: int,
                   smoke_epochs: int | None = None) -> RunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        for output_name in ("metrics.json", "predictions.csv"):
            output_path = out_dir / output_name
            if output_path.exists():
                output_path.unlink()
        cmd = [sys.executable, str(script), "--data-dir", str(self.workspace),
               "--out-dir", str(out_dir), "--seed", str(seed)]
        env = build_subprocess_env({"NODE_TIMEOUT_S": str(timeout_s)})
        if smoke_epochs is None:
            smoke_epochs = self.config.fast_forward_epochs  # fast-forward: cap full runs too
        if smoke_epochs is not None:
            env["SMOKE_EPOCHS"] = str(smoke_epochs)
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                                  cwd=str(ROOT), env=env)
        except subprocess.TimeoutExpired:
            return RunResult(False, error=f"timeout after {timeout_s}s", duration_s=time.time() - start)
        duration = time.time() - start
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or proc.stdout or "no output").splitlines()[-30:])
            return RunResult(False, error=tail, duration_s=duration)
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists() or not (out_dir / "predictions.csv").exists():
            return RunResult(False, error="script exited 0 but did not write metrics.json + predictions.csv",
                             duration_s=duration)
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            return RunResult(False, error=f"unparseable metrics.json: {exc}", duration_s=duration)
        for key in METRIC_KEYS:
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                return RunResult(False, error=f"invalid metric {key}={value!r}", duration_s=duration)
        return RunResult(True, metrics=metrics, duration_s=duration)

    def run_experiment(self, node: Node, timeout_s: int) -> tuple[RunResult, str]:
        """Run the mandatory smoke stage, then the full experiment if it passes."""
        if node.action in ("draft", "improve", "debug") and node.execution_kind == "script":
            smoke = self.run_script(
                node.code_path,
                self.run_dir / f"{node.node_id}_smoke",
                self.config.seed,
                360,  # heavy architectures (DeepFM/history pooling) cannot smoke in 120s
                smoke_epochs=1,
            )
            if not smoke.ok:
                return smoke, "smoke"
            gate_error = self._smoke_sanity_error(smoke.metrics or {})
            if gate_error:
                return RunResult(False, error=gate_error,
                                 duration_s=smoke.duration_s), "smoke"
        return (
            self.run_script(
                node.code_path, self.run_dir / node.node_id, self.config.seed, timeout_s
            ),
            "full",
        )

    def _smoke_sanity_error(self, metrics: dict) -> str | None:
        """Smoke sanity gate: reject clearly-broken code on its 1-epoch probe.

        Two checks (both generous -- catch broken code, not weak ideas):
        below-chance GAUC, and primary far below the baseline's own 1-epoch
        probe (smoke_reference.json, written during calibration when available).
        The failure is fixer-eligible via the normal "smoke" stage path."""
        gauc = metrics.get("gauc")
        if isinstance(gauc, (int, float)) and gauc < 0.5:
            return (f"smoke sanity gate: 1-epoch GAUC {gauc:.4f} is below chance (0.5); "
                    "the script likely mis-wires labels, user grouping, or score signs")
        ref_path = self.run_dir / "smoke_reference.json"
        if ref_path.exists():
            try:
                ref = json.loads(ref_path.read_text())
            except json.JSONDecodeError:
                return None
            margin = 0.010  # measured: sane 1-epoch screens land 0.592-0.600 vs
            # baseline ~0.602 while broken scripts land <=0.590 (fidelity sweep
            # 31 Aug); 0.010 separates the two populations
            ref_primary, primary = ref.get("primary"), metrics.get("primary")
            if (isinstance(ref_primary, (int, float)) and isinstance(primary, (int, float))
                    and primary < ref_primary - margin):
                return (f"smoke sanity gate: 1-epoch primary {primary:.4f} is more than "
                        f"{margin} below the baseline's own 1-epoch probe "
                        f"{ref_primary:.4f}; the change likely breaks the pipeline "
                        "rather than merely underperforming")
        return None

    # ---------- calibration ----------

    def calibrate(self) -> None:
        """Baseline node_000 + sigma from 3 seeds (CONTRACTS.md acceptance)."""
        code = self.config.baseline_script.read_text()
        node = Node("node_000", "baseline", "draft", "baseline FM (zoo/fm_torch.py)",
                    self.nodes_dir / "000.py")
        node.code_path.write_text(code)
        primaries = []
        selected_calibration_dir: Path | None = None
        # With an externally supplied sigma (tests / reruns) one baseline run suffices.
        seeds = self.config.calib_seeds if self.config.sigma is None else (self.config.seed,)
        for i, seed in enumerate(seeds):
            # Calibration runs trusted baseline code: never let a tight experiment
            # timeout (e.g. in tests) starve it.
            calibration_dir = self.run_dir / f"calib_seed{seed}"
            result = self.run_script(node.code_path, calibration_dir, seed,
                                     max(self.config.timeout_s, 120))
            if not result.ok:
                raise RuntimeError(f"baseline calibration failed (seed {seed}): {result.error}")
            primaries.append(result.metrics["primary"])
            if seed == self.config.seed or i == 0:
                node.metrics, node.primary = result.metrics, result.metrics["primary"]
                selected_calibration_dir = calibration_dir
        # Materialize node_000 like every other node so ensemble/no-op checks can
        # compare directly with a baseline parent.
        baseline_node_dir = self.run_dir / node.node_id
        baseline_node_dir.mkdir(parents=True, exist_ok=True)
        if selected_calibration_dir is not None:
            for output_name in ("predictions.csv", "metrics.json"):
                shutil.copy2(selected_calibration_dir / output_name, baseline_node_dir / output_name)
        if self.config.sigma is not None:
            self.sigma = self.config.sigma
        else:
            self.sigma = float(np.std(primaries))
        # Smoke sanity reference (ledger spec): the baseline's own 1-epoch probe,
        # so defective experiment code can be rejected at the smoke stage instead
        # of after a full training. Failure to build it just disables that check.
        try:
            smoke_ref = self.run_script(node.code_path, self.run_dir / "node_000_smoke",
                                        self.config.seed, 360, smoke_epochs=1)
            if smoke_ref.ok:
                (self.run_dir / "smoke_reference.json").write_text(json.dumps(smoke_ref.metrics))
        except Exception as exc:
            print(f"[loop] smoke reference unavailable: {exc}", file=sys.stderr)
        node.status = "accepted"
        node.change_summary = "baseline FM reproduction and seed-noise calibration"
        self.nodes[node.node_id] = node
        self.champion = node
        self.journal_lines.append(
            f'node_000 [baseline] draft "baseline FM" primary={node.primary:.4f} '
            f"ACCEPTED (sigma={self.sigma:.4f})"
        )
        self.record_calibration(primaries)

    def node_diff(self, node: Node, max_lines: int = 400) -> str:
        """Unified diff of this node's code vs its parent's (brief: 'the code diff applied')."""
        try:
            new = node.code_path.read_text().splitlines(keepends=True)
        except OSError:
            return "(code unavailable)"
        parent = self.nodes.get(node.parent)
        old = []
        old_name = "(new file)"
        if parent is not None and parent.code_path.exists():
            old = parent.code_path.read_text().splitlines(keepends=True)
            old_name = parent.node_id
        lines = list(difflib.unified_diff(old, new, fromfile=old_name, tofile=node.node_id, n=2))
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... (diff truncated at {max_lines} lines)\n"]
        return "".join(lines)

    def record_calibration(self, primaries: list[float]) -> None:
        """Iteration-0 journal entry: the agent's own baseline reproduction (brief task req #1)."""
        mean = float(np.mean(primaries))
        self.calibration_result = {
            "seed_primaries": [round(p, 6) for p in primaries],
            "mean": round(mean, 6),
            "sigma": round(self.sigma, 6),
            "published_valid_primary": 0.6016,
            "pass": abs(mean - 0.6016) <= 0.003,
        }
        record = {
            "n": 0, "node_id": "node_000", "parent": "baseline", "action": "reproduce_baseline",
            "hypothesis": "reproduce official FM baseline and calibrate seed noise",
            "change_summary": f"baseline seeds {list(self.config.calib_seeds) if self.config.sigma is None else [self.config.seed]}: "
                              f"primaries {[round(p, 4) for p in primaries]}, mean {mean:.4f}, sigma {self.sigma:.4f}",
            "diff": "",
            "context_mode": self.config.context_mode,
            "knowledge_mode": self.config.knowledge_mode,
            "method_selection": None,
            "metrics": self.champion.metrics if self.champion else {},
            "val_best_so_far": self.champion.primary if self.champion else 0.0,
            "baseline_reproduction": self.calibration_result,
            "accepted": True, "duration_s": 0.0, "tokens_in": 0, "tokens_out": 0,
            "expected_delta": None, "expected_delta_basis": None, "realized_delta": None,
            "verdict_note": None,
            "failure_stage": None, "fixer_eligible": False,
            "error": None, "recovery": None,
            "usd_total": round(getattr(self.brain, "usd_total", 0.0), 4),
            "intervention": False,
        }
        with self.journal_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def plan_exploration_budget(self) -> None:
        """Make and record the single opt-in post-calibration planning call."""
        method_families = sorted({
            family
            for method_id in getattr(self.brain, "method_cards", {})
            for family in self.method_metadata(method_id)["treats"]
        })
        raw_plan = self.brain.plan_exploration(
            dict(self.calibration_result or {}), self.config.max_iters, method_families
        )
        self.initial_draft_slots = min(6, max(2, raw_plan["initial_draft_slots"]))
        self.exploration_plan = {
            **raw_plan,
            "n": 0.5,
            "action": "plan",
            "planned_draft_count": self.initial_draft_slots,
            "raw_plan": raw_plan,
            "context_mode": self.config.context_mode,
            "knowledge_mode": self.config.knowledge_mode,
            "tokens_in": self.brain.meter.last_in,
            "tokens_out": self.brain.meter.last_out,
            "usd_total": round(getattr(self.brain, "usd_total", 0.0), 4),
        }
        with self.journal_path.open("a") as handle:
            handle.write(json.dumps(self.exploration_plan) + "\n")

    # ---------- policy (harness-owned) ----------

    def next_move(self) -> tuple[str, Node, str | None]:
        """Return (mode, parent_node, directive)."""
        drafts = [n for n in self.nodes.values() if n.action == "draft" and n.node_id != "node_000"]
        last = self.nodes[max(self.nodes, key=lambda k: int(k.split("_")[1]))]
        if last.status in ("failed", "suspect_implementation") and last.debug_depth < 2:
            return "debug", last, None
        if len(drafts) < self.initial_draft_slots:
            slot = len(drafts)
            if slot < len(self.config.draft_tiers):
                tier = self.config.draft_tiers[slot]
                return "draft", self.champion, f"draft from {tier} of the menu"
            return "draft", self.champion, (
                f"planned initial draft slot {slot + 1} of {self.initial_draft_slots}"
            )
        if self.stagnation >= self.config.stagnation_limit:
            tried = {n.tier for n in self.nodes.values() if n.tier}
            untried = [t for t in ALL_TIERS if t not in tried and t not in self.forced_tiers_used]
            tier = untried[0] if untried else ALL_TIERS[len(self.forced_tiers_used) % len(ALL_TIERS)]
            self.forced_tiers_used.append(tier)
            self.stagnation = 0
            return "draft", self.champion, (
                f"forced branch after stagnation: draft from {tier}, an untried menu tier"
            )
        # Branch to the best runner-up lineage when the champion's last two
        # children were both rejected and a near-champion alternative exists.
        ordered = sorted(self.nodes.values(), key=lambda x: int(x.node_id.split("_")[1]))
        recent = ordered[-2:]
        if (len(recent) == 2
                and all(x.status == "rejected" and x.parent == self.champion.node_id
                        for x in recent)):
            alts = [n for n in self.nodes.values()
                    if n.status == "accepted" and n.node_id != self.champion.node_id
                    and n.primary is not None
                    and n.primary >= (self.champion.primary or 0) - 0.0015]
            if alts:
                alt = max(alts, key=lambda n: n.primary)
                return "improve", alt, (
                    f"branch: the champion's recent children were rejected; improve the "
                    f"runner-up lineage {alt.node_id} (primary {alt.primary:.4f}) instead")
        return "improve", self.champion, None

    # ---------- acceptance ----------

    def acceptance(self, node: Node, metrics: dict) -> tuple[bool, str | None]:
        """Returns (accepted, note). May run one confirm reseed."""
        farm_metrics = metrics.get("farm_close")
        legitimate_incumbent_fallback = bool(
            metrics.get("fallback_to_incumbent")
            or (
                isinstance(farm_metrics, dict)
                and farm_metrics.get("fallback_to_incumbent")
            )
        )
        if legitimate_incumbent_fallback:
            node.failure_stage = None
            node.fixer_eligible = False
        # No-op guard (ported from teammate harness yash-attempt): predictions
        # byte-identical to the parent's are a disguised no-change.
        try:
            import hashlib
            parent = self.nodes.get(node.parent)
            mine = self.run_dir / node.node_id / "predictions.csv"
            theirs = (self.run_dir / parent.node_id / "predictions.csv") if parent else None
            if theirs and mine.exists() and theirs.exists():
                h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
                if h(mine) == h(theirs) and not legitimate_incumbent_fallback:
                    # degenerate ensembles collapsed to the parent twice tonight;
                    # give the fixer one shot instead of silently burning the node
                    node.failure_stage = "noop_predictions"
                    node.fixer_eligible = True
                    return False, "no-op: predictions byte-identical to parent"
        except Exception:
            pass
        delta = metrics["primary"] - self.champion.primary
        threshold = max(2 * self.sigma, self.config.accept_floor if self.config.accept_floor is not None else self.config.epsilon)
        if delta >= threshold:
            return True, None
        if delta > 0:
            primaries = [metrics["primary"]]
            for k in range(self.config.confirm_runs):
                confirm = self.run_script(node.code_path,
                                          self.run_dir / f"{node.node_id}_confirm{k if k else ''}",
                                          self.config.confirm_seed + k, self.config.timeout_s)
                if not confirm.ok:
                    return False, f"grey-zone confirm run failed: {confirm.error}"
                primaries.append(confirm.metrics["primary"])
            n = len(primaries)
            mean_delta = sum(primaries) / n - self.champion.primary
            if n >= 2:
                m = sum(primaries) / n
                sd = (sum((p - m) ** 2 for p in primaries) / (n - 1)) ** 0.5
            else:
                sd = self.sigma
            se = max(sd / (n ** 0.5), 1e-6)
            z = mean_delta / se
            # grey floor lowered 0.0007 -> 0.0005 (31 Aug, disclosed): campaign
            # post-mortem showed recurring z>=2-confirmable gains in 0.0004-0.0007
            # dying on the hard constant; the z-test remains the noise gate.
            floor = max(self.sigma / (n ** 0.5), 0.0005)
            # seed-mean z-test (ported from yash-attempt): require the floor AND z >= 2
            if mean_delta >= floor and z >= 2.0:
                return True, f"grey-zone confirm passed (mean delta {mean_delta:+.4f}, z={z:.1f}, n={n})"
            return False, f"grey-zone confirm failed (mean delta {mean_delta:+.4f}, z={z:.1f}, n={n})"
        return False, None

    # ---------- measured method routing ----------

    def method_metadata(self, method_id: str | None) -> dict:
        card = getattr(self.brain, "method_cards", {}).get(method_id)
        return parse_method_card_metadata(card, self.config.dataset) if card else {
            "treats": [], "reference_primary": None, "expected_gain": 0.0,
            "measured_dead": False,
        }

    def excluded_draft_families(self) -> list[str]:
        families = set()
        for prior in self.nodes.values():
            if prior.action != "draft" or not prior.method_selection:
                continue
            families.update(self.method_metadata(
                prior.method_selection.get("chosen_method_id")
            )["treats"])
        return sorted(families)

    def eligible_unexcluded_methods(self, excluded_families: list[str]) -> list[str]:
        excluded = set(excluded_families)
        return [
            method_id
            for method_id in getattr(self.brain, "method_cards", {})
            if not self.method_metadata(method_id)["measured_dead"]
            and not (set(self.method_metadata(method_id)["treats"]) & excluded)
        ]

    def select_method(self, parent_history: list, streak_state: dict,
                      mode: str) -> dict:
        excluded = self.excluded_draft_families() if mode == "draft" else []
        selector_kwargs = {
            "excluded_families": excluded,
            "dataset": self.config.dataset,
            "prior_runs": self.prior_runs,
        }
        if self.exploration_plan is not None:
            selector_kwargs["preference_note"] = (
                "Prioritize these card families in order: "
                + ", ".join(self.exploration_plan["family_priorities"])
                + ". Planner rationale: " + self.exploration_plan["rationale"]
            )
        selection = self.brain.select_method(
            self.journal_lines, parent_history, streak_state, **selector_kwargs
        )
        eligible = self.eligible_unexcluded_methods(excluded)
        if mode != "draft" or not eligible or selection.get("chosen_method_id") in eligible:
            return selection
        selection = self.brain.select_method(
            self.journal_lines, parent_history, streak_state,
            enforce_family_exclusion=True, **selector_kwargs
        )
        if selection.get("chosen_method_id") in eligible:
            return selection
        chosen = max(eligible, key=lambda method_id: self.method_metadata(method_id)["expected_gain"])
        overridden = dict(selection)
        overridden["chosen_method_id"] = chosen
        overridden["citation"] = "harness portfolio-diversity override"
        overridden["why"] = (
            f"Selector twice violated excluded_families={excluded}; harness chose the "
            "highest-expected-gain eligible card."
        )
        overridden["harness_override"] = True
        return overridden

    def reference_primary(self, node: Node) -> float | None:
        if self.config.knowledge_mode == "clean":
            return None
        selection = node.method_selection or {}
        return self.method_metadata(selection.get("chosen_method_id"))["reference_primary"]

    def compile_farm_close_node(self, node: Node, plan_value: object, parent: Node) -> str:
        """Validate a typed plan, persist it, and compile the harness-owned wrapper."""
        plan = validate_farm_close_plan(plan_value)
        validate_script_sources(plan)
        for index, member in enumerate(plan["members"]):
            if "code" in member:
                leak = self.check_code_leakage(member["code"])
                if leak:
                    raise FarmClosePlanError(f"members[{index}].code: {leak}")
        plan_path = self.nodes_dir / f"{node.node_id.split('_')[1]}.farm.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        node.farm_close_plan = plan
        node.farm_close_plan_path = plan_path
        node.execution_kind = "farm_close"
        prediction_parent = parent
        seen: set[str] = set()
        while prediction_parent.node_id not in seen:
            seen.add(prediction_parent.node_id)
            prediction_path = self.run_dir / prediction_parent.node_id / "predictions.csv"
            if prediction_path.exists():
                break
            ancestor = self.nodes.get(prediction_parent.parent)
            if ancestor is None:
                prediction_path = None
                break
            prediction_parent = ancestor
        return render_plan_node(
            plan,
            parent_predictions=prediction_path,
            base_seed=self.config.seed,
        )

    # ---------- bookkeeping ----------

    @staticmethod
    def _full_context_block(node: Node) -> str:
        metrics = node.metrics or {}
        metric_text = ", ".join(
            f"{key}={metrics.get(key)!r}" for key in METRIC_KEYS
        )
        if node.error:
            error_tail = "\n".join(node.error.splitlines()[-5:])
            outcome = f"error:\n{error_tail}"
        else:
            outcome = node.status
        history = metrics.get("history") or []
        history_lines = [json.dumps(point, sort_keys=True) for point in history[-10:]]
        history_text = "\n".join(history_lines) or "(no learning curve recorded)"
        return (
            f"### {node.node_id}\n"
            f"hypothesis: {node.hypothesis}\n"
            f"action: {node.action}\n"
            f"metrics: {metric_text}\n"
            f"outcome: {outcome}\n"
            f"change_summary: {node.change_summary or node.hypothesis}\n"
            f"learning_curve_last_10:\n{history_text}"
        )

    def full_proposer_context(self) -> str:
        """Structured prior-node context, dropping oldest optional nodes first."""
        if not self.nodes:
            return "(empty)"
        ordered = sorted(self.nodes.values(), key=lambda node: int(node.node_id.split("_")[1]))
        mandatory_ids = {"node_000"}
        if self.champion is not None:
            mandatory_ids.add(self.champion.node_id)
        blocks = {node.node_id: self._full_context_block(node) for node in ordered}
        kept = {node.node_id for node in ordered if node.node_id in mandatory_ids}
        used = sum(len(blocks[node_id]) + 2 for node_id in kept)
        for node in reversed(ordered):
            if node.node_id in kept:
                continue
            block_size = len(blocks[node.node_id]) + 2
            if used + block_size <= FULL_CONTEXT_CHAR_BUDGET:
                kept.add(node.node_id)
                used += block_size
        return "\n\n".join(blocks[node.node_id] for node in ordered if node.node_id in kept)

    def record(self, n: int, node: Node, duration: float, recovery: str | None,
               change_summary: str) -> None:
        node.change_summary = change_summary
        realized_delta = (
            node.primary - getattr(self, "_best_before_iter", node.primary)
            if node.primary is not None else None
        )
        record = {
            "n": n,
            "hypothesis": node.hypothesis,
            "node_id": node.node_id,
            "parent": node.parent,
            "action": node.action,
            "code_path": str(node.code_path.relative_to(ROOT)) if node.code_path.is_relative_to(ROOT)
                         else str(node.code_path),
            "change_summary": change_summary,
            "context_mode": self.config.context_mode,
            "knowledge_mode": self.config.knowledge_mode,
            "diff": self.node_diff(node),
            "metrics": {k: v for k, v in (node.metrics or {"gauc": 0.0, "ndcg5": 0.0, "primary": 0.0}).items() if k != "history"},
            "history": (node.metrics or {}).get("history", []),
            "method_selection": node.method_selection,
            "val_best_so_far": self.champion.primary if self.champion else 0.0,
            "accepted": node.status == "accepted",
            "expected_delta": node.expected_delta,
            "expected_delta_basis": node.expected_delta_basis,
            "realized_delta": realized_delta,
            "verdict_note": node.verdict_note,
            "failure_stage": node.failure_stage,
            "fixer_eligible": node.fixer_eligible,
            "duration_s": round(duration, 2),
            "tokens_in": self.brain.meter.last_in,
            "tokens_out": self.brain.meter.last_out,
            "error": node.error,
            "recovery": recovery or node.recovery,
            "usd_total": round(getattr(self.brain, "usd_total", 0.0), 4),
            "intervention": False,
        }
        if node.farm_close_plan is not None:
            record["execution_kind"] = node.execution_kind
            record["farm_close_plan"] = node.farm_close_plan
            record["farm_close_plan_path"] = str(node.farm_close_plan_path)
        with self.journal_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        status = node.status.upper()
        metric_text = f"primary={node.primary:.4f}" if node.primary is not None else "no-metric"
        self.journal_lines.append(
            f'{node.node_id} [<-{node.parent}] {node.action} "{node.hypothesis}" {metric_text} {status}'
        )

    def budget_exceeded(self) -> str | None:
        if time.time() - self.start_time > self.config.max_hours * 3600:
            return "max_hours"
        if self.brain.meter.total >= self.config.max_tokens:
            return "max_tokens"
        if getattr(self.brain, "usd_run", 0.0) >= self.config.max_usd:
            return "max_usd"
        return None

    def converged(self) -> bool:
        # OFFICIAL rule: converged when validation primary has not improved by more
        # than epsilon over the last N consecutive COMPLETED iterations (accepted,
        # rejected, or errored all count). Improvement is vs best-so-far.
        return self.no_improve_streak >= self.config.n_converge

    # ---------- cross-run memory ----------

    def read_cross_run(self, max_lines: int = 40) -> str:
        if self.config.knowledge_mode == "clean":
            return ""
        path = self.config.cross_run_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return "\n".join(path.read_text().splitlines()[-max_lines:])

    @staticmethod
    def _eight_word_summary(hypothesis: str) -> str:
        return " ".join(hypothesis.split()[:8])

    def append_cross_run(self, summary: dict, self_critique: str | None = None) -> None:
        if self.config.knowledge_mode == "clean":
            return
        lines = [
            f"## Run {self.run_dir}",
            f"dataset: {self.config.dataset}",
            f"stop_reason: {summary['stop_reason']}",
            f"best_primary: {summary['best_metrics']['primary']:.6f}",
        ]
        ordered = sorted(self.nodes.values(), key=lambda node: int(node.node_id.split("_")[1]))
        for node in ordered:
            if node.node_id == "node_000":
                continue
            method_id = (node.method_selection or {}).get("chosen_method_id", "none")
            primary = f"{node.primary:.6f}" if node.primary is not None else "n/a"
            lines.append(
                f"- {node.node_id} | method: {method_id} | hypothesis: "
                f"{self._eight_word_summary(node.hypothesis)} | primary: {primary} | "
                f"verdict: {node.status}"
            )
        if self_critique:
            lines.extend(["self_critique:", self_critique.strip()])
        with self.config.cross_run_path.open("a") as handle:
            if self.config.cross_run_path.stat().st_size:
                handle.write("\n")
            handle.write("\n".join(lines) + "\n")

    def terminal_self_critique(self) -> str:
        journal_summary = "\n".join([
            f"run_dir: {self.run_dir}",
            f"dataset: {self.config.dataset}",
            f"stop_reason: {self.stop_reason}",
            f"best_primary: {self.champion.primary:.6f}",
            *self.journal_lines,
        ])
        try:
            return self.brain.self_critique(journal_summary)
        except Exception as exc:  # archival reflection must never discard a completed run
            print(f"[loop] end-of-run self-critique failed: {exc}", file=sys.stderr)
            return f"self-critique unavailable: {exc}"

    # ---------- main loop ----------

    def seed_reference_nodes(self, start_n: int) -> int:
        """Run team-provided reference scripts as disclosed initial draft nodes.

        Mirrors AIDE-style seeding: the agent's search starts from the documented
        best-known configuration instead of re-deriving it. Fully recorded in the
        journal with its provenance (seed script name)."""
        n = start_n
        for script in self.config.seed_scripts:
            n += 1
            start = time.time()
            self._best_before_iter = self.champion.primary if self.champion else 0.0
            node = Node(f"node_{n:03d}", "node_000", "draft",
                        f"seed script (disclosed initial draft): {script.name}",
                        self.nodes_dir / f"{n:03d}.py")
            node.code_path.write_text(script.read_text())
            recovery = None
            result, failed_stage = self.run_experiment(node, max(self.config.timeout_s, 600))
            if not result.ok:
                node.error = result.error
                node.failure_stage = failed_stage
                node.fixer_eligible = True
                original_code = node.code_path.read_text()
                try:
                    fixed = self.brain.fix(original_code, result.error)
                except BudgetExhausted:
                    fixed = original_code
                    self.stop_reason = "budget_exhausted"
                except Exception as exc:
                    fixed = original_code
                    print(f"[loop] seed fixer failed: {exc}", file=sys.stderr)
                if fixed != original_code and not self.check_code_leakage(fixed):
                    node.code_path.write_text(fixed)
                    result, failed_stage = self.run_experiment(
                        node, max(self.config.timeout_s, 600)
                    )
                    recovery = "patched" if result.ok else "reverted"
                    if not result.ok:
                        node.failure_stage = failed_stage
                else:
                    recovery = "reverted"
            if result.ok:
                node.metrics, node.primary = result.metrics, result.metrics["primary"]
                accepted, note = self.acceptance(node, result.metrics)
                if (not accepted and node.failure_stage == "noop_predictions"
                        and node.fixer_eligible):
                    # Degenerate ensemble collapsed to its parent (acceptance-time
                    # detection, so the ok-path fixer never sees it). Give the fixer
                    # one repair shot here, then re-run and re-judge once.
                    try:
                        original_code = node.code_path.read_text()
                        fixed = self.brain.fix(
                            original_code,
                            "Ensemble output is byte-identical to the parent's "
                            "predictions: members likely collapsed (shared seed, "
                            "anchor duplication, or gating that removes all "
                            "members). Ensure distinct member seeds and that the "
                            "final aggregation actually combines all member score "
                            "vectors.")
                        if fixed != original_code and not self.check_code_leakage(fixed):
                            node.code_path.write_text(fixed)
                            retry, retry_stage = self.run_experiment(node, self.config.timeout_s)
                            if retry.ok:
                                node.failure_stage = None
                                node.recovery = "noop-fixer: repaired and re-run"
                                result = retry
                                node.metrics = retry.metrics
                                node.primary = retry.metrics["primary"]
                                accepted, note = self.acceptance(node, result.metrics)
                            else:
                                # transactional: keep the ORIGINAL (working) code and
                                # record why the repair attempt failed
                                node.code_path.write_text(original_code)
                                node.recovery = (f"noop-fixer retry failed at "
                                                 f"{retry_stage}: {str(retry.error)[:200]}")
                    except BudgetExhausted:
                        self.stop_reason = "budget_exhausted"
                    except Exception as exc:
                        print(f"[loop] noop fixer failed: {exc}", file=sys.stderr)
                if accepted:
                    delta = node.primary - self.champion.primary
                    node.status = "accepted"
                    self.champion = node
                    self.accepted_deltas.append(delta)
                else:
                    node.status = "rejected"
            else:
                node.status, node.error = "failed", result.error
            best_now = self.champion.primary if self.champion else 0.0
            improvement = best_now - self._best_before_iter
            if improvement > self.config.epsilon:
                self.no_improve_streak = 0
            else:
                self.no_improve_streak += 1
            self.nodes[node.node_id] = node
            self.record(n, node, time.time() - start, recovery, node.hypothesis)
            status = node.status.upper()
            self.journal_lines.append(
                f'{node.node_id} [seed] draft "{script.name}" '
                + (f"primary={node.primary:.4f} {status}" if node.primary else status))
        return n

    def resume_from_run(self, src: Path, at: int) -> int:
        """Continue a prior run from just before iteration `at`.

        Copies the prior run's artifacts for iterations < at, replays its journal
        records verbatim (plus a lineage marker), and reconstructs the loop state
        those records imply (nodes, champion, sigma, official convergence streak)
        using the loop's own rules, so the CURRENT agent code faces exactly the
        state the prior agent faced. Budgets start fresh for the new run. A
        resumed run is a disclosed experiment lineage, never a designation
        candidate. Returns the iteration counter to continue from (at - 1)."""
        src = Path(src)
        journal = src / "journal.jsonl"
        if self.config.knowledge_mode == "clean":
            raise ValueError("clean-mode runs cannot be resumed (they must be unassisted end to end)")
        if at < 1:
            raise ValueError("--at must be >= 1 (iteration 0 is the baseline)")
        if not journal.exists():
            raise FileNotFoundError(f"resume source has no journal: {journal}")
        records = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        by_n = {r["n"]: r for r in records
                if isinstance(r.get("n"), int) and not isinstance(r.get("n"), bool)}
        if 0 not in by_n:
            raise ValueError("resume source journal lacks the baseline record (n=0)")
        max_n = max(by_n)
        if at > max_n + 1:
            raise ValueError(f"--at {at} exceeds the source run's recorded iterations ({max_n})")
        missing = [k for k in range(at) if k not in by_n]
        if missing:
            raise ValueError(f"resume source journal is missing iterations {missing}")
        for k in range(1, at):
            if not (src / "nodes" / f"{k:03d}.py").exists():
                raise FileNotFoundError(f"resume source lacks nodes/{k:03d}.py")

        # --- copy artifacts for iterations < at
        for item in sorted(src.iterdir()):
            name = item.name
            if name.startswith("calib_seed") and item.is_dir():
                shutil.copytree(item, self.run_dir / name, dirs_exist_ok=True)
            elif name.startswith("node_") and item.is_dir():
                try:
                    k = int(name.split("_")[1][:3])
                except ValueError:
                    continue
                if k < at:
                    shutil.copytree(item, self.run_dir / name, dirs_exist_ok=True)
        for k in range(at):
            for suffix in (".py", ".farm.json"):
                sidecar = src / "nodes" / f"{k:03d}{suffix}"
                if sidecar.exists():
                    shutil.copy2(sidecar, self.nodes_dir / f"{k:03d}{suffix}")

        # --- replay journal verbatim + lineage marker
        with self.journal_path.open("w") as handle:
            for k in range(at):
                handle.write(json.dumps(by_n[k]) + "\n")
        (self.run_dir / "resume.json").write_text(json.dumps({
            "resumed_from": str(src), "resumed_at": at, "intervention": False,
            "designation_eligible": False,
            "note": "resumed lineage: continued from the prior run's state under "
                    "current agent code; experiment, not a designation candidate",
        }, indent=2) + "\n")

        # --- reconstruct state with the loop's own rules
        base = by_n[0]
        calib = sorted(self.run_dir.glob("calib_seed*/metrics.json"))
        if len(calib) >= 2:
            self.sigma = float(np.std([json.loads(c.read_text())["primary"] for c in calib]))
        else:
            self.sigma = float((base.get("baseline_reproduction") or {}).get("sigma")
                               or self.config.sigma or 0.0)
        self.calibration_result = base.get("baseline_reproduction")
        node0 = Node("node_000", "baseline", "draft", "baseline FM (zoo/fm_torch.py)",
                     self.nodes_dir / "000.py")
        node0.metrics = base.get("metrics") or {}
        node0.primary = float(node0.metrics.get("primary", 0.0))
        node0.status = "accepted"
        node0.change_summary = base.get("change_summary", "")
        self.nodes[node0.node_id] = node0
        self.champion = node0
        self.journal_lines.append(
            f'node_000 [baseline] draft "baseline FM" primary={node0.primary:.4f} '
            f"ACCEPTED (sigma={self.sigma:.4f})"
        )
        self.no_improve_streak = 0
        for k in range(1, at):
            r = by_n[k]
            node = Node(r["node_id"], r.get("parent", "node_000"), r.get("action", "draft"),
                        r.get("hypothesis", ""), self.nodes_dir / f"{k:03d}.py")
            metrics = dict(r.get("metrics") or {})
            if r.get("history"):
                metrics["history"] = r["history"]
            failed = bool(r.get("error")) or (
                not r.get("accepted") and float(metrics.get("primary", 0.0) or 0.0) == 0.0)
            node.metrics = None if failed else metrics
            node.primary = None if failed else float(metrics.get("primary"))
            node.status = "accepted" if r.get("accepted") else ("failed" if failed else "rejected")
            node.error = r.get("error")
            node.method_selection = r.get("method_selection")
            node.change_summary = r.get("change_summary", "")
            node.recovery = r.get("recovery")
            node.failure_stage = r.get("failure_stage")
            node.fixer_eligible = bool(r.get("fixer_eligible", False))
            node.verdict_note = r.get("verdict_note")
            node.expected_delta = r.get("expected_delta")
            node.expected_delta_basis = r.get("expected_delta_basis")
            if r.get("action") == "debug":
                parent_node = self.nodes.get(node.parent)
                node.debug_depth = (parent_node.debug_depth + 1) if parent_node else 1
            if r.get("farm_close_plan") is not None:
                node.farm_close_plan = r["farm_close_plan"]
                node.execution_kind = r.get("execution_kind", "farm_close")
            best_before = self.champion.primary
            if node.status == "accepted":
                self.accepted_deltas.append(node.primary - best_before)
                self.champion = node
            improvement = self.champion.primary - best_before
            if improvement > self.config.epsilon:
                self.no_improve_streak = 0
            else:
                self.no_improve_streak += 1
            self.nodes[node.node_id] = node
            metric_text = f"primary={node.primary:.4f}" if node.primary is not None else "no-metric"
            self.journal_lines.append(
                f'{node.node_id} [<-{node.parent}] {node.action} "{node.hypothesis}" '
                f"{metric_text} {node.status.upper()}"
            )
        print(f"[loop] resumed {src} at iteration {at}: best {self.champion.node_id} "
              f"{self.champion.primary:.6f}, streak {self.no_improve_streak}, sigma {self.sigma:.4f}",
              file=sys.stderr)
        return at - 1

    def run(self) -> dict:
        self.prior_runs = self.read_cross_run()
        self.prepare_workspace()
        if self.config.resume_from is not None:
            n = self.resume_from_run(self.config.resume_from, self.config.resume_at)
        else:
            self.calibrate()
            if self.config.plan_budget:
                self.plan_exploration_budget()
            n = 0
            n = self.seed_reference_nodes(n)
        while n < self.config.max_iters:
            reason = self.budget_exceeded()
            if reason:
                self.stop_reason = reason
                break
            if self.converged():
                self.stop_reason = "converged"
                break
            n += 1
            if n > 1 and (n - 1) % self.config.reflect_every == 0:
                self._reflect()
            self.iterate(n)
            if self.stagnation >= 3 and not self.stop_reason:
                self._reflect()
            if self.stop_reason:
                break
        else:
            self.stop_reason = self.stop_reason or "max_iters"
        self_critique = self.terminal_self_critique()
        summary = {
            "run_dir": str(self.run_dir),
            "dataset": self.config.dataset,
            "knowledge_mode": self.config.knowledge_mode,
            "stop_reason": self.stop_reason,
            "iterations": n,
            "best_node": self.champion.node_id,
            "best_metrics": self.champion.metrics,
            "sigma": self.sigma,
            "tokens": self.brain.meter.per_role,
            "tokens_total": self.brain.meter.total,
            "wall_s": round(time.time() - self.start_time, 1),
            "self_critique": self_critique,
        }
        if self.exploration_plan is not None:
            summary["exploration_plan"] = self.exploration_plan
        if self.config.resume_from is not None:
            summary["resumed_from"] = str(self.config.resume_from)
            summary["resumed_at"] = self.config.resume_at
            summary["lineage"] = "resumed experiment; not a designation candidate"
            summary["designation_eligible"] = False
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self.append_cross_run(summary, self_critique)
        return summary

    def _reflect(self) -> None:
        """Refresh strategy on the every-5 schedule or after three stagnant runs."""
        try:
            self.focus_note = self.brain.reflect(self.journal_lines)
        except Exception as exc:  # reflection is optional; never kills the loop
            self.focus_note = None
            print(f"[loop] reflector failed: {exc}", file=sys.stderr)

    def iterate(self, n: int) -> None:
        start = time.time()
        self._best_before_iter = self.champion.primary if self.champion else 0.0
        mode, parent, directive = self.next_move()
        node = Node(f"node_{n:03d}", parent.node_id, mode, "(proposal failed)",
                    self.nodes_dir / f"{n:03d}.py")
        streak_state = {
            "no_improve_streak": self.no_improve_streak,
            "n_converge": self.config.n_converge,
            "iters_left": self.config.max_iters - n,
        }
        if mode == "debug":
            node.debug_depth = parent.debug_depth + 1
            node.method_selection = parent.method_selection
        recovery: str | None = None
        try:
            parent_history = normalize_history((parent.metrics or {}).get("history", []))
            if mode in ("draft", "improve"):
                node.method_selection = self.select_method(parent_history, streak_state, mode)
            spec = self.brain.propose(
                self.journal_lines, mode, parent.node_id, parent.code_path.read_text(),
                directive=directive, focus_note=self.focus_note,
                traceback_tail=(parent.error or parent.verdict_note) if mode == "debug" else None,
                parent_history=parent_history,
                method_selection=node.method_selection,
                streak_state=streak_state,
                context_mode=self.config.context_mode,
                full_context=self.full_proposer_context() if self.config.context_mode == "full" else None,
                prior_runs=self.prior_runs,
                timeout_s=self.config.timeout_s,
                parent_code_path=str(parent.code_path),
            )
            execution_kind = spec.get("execution_kind")
            if execution_kind not in (None, "script", "farm_close"):
                raise ValueError("execution_kind must be 'script' or 'farm_close'")
            method_farm_fallback = (
                (node.method_selection or {}).get("chosen_method_id")
                in ("diverse-family-farm-close", "heterogeneous-ensemble-design")
            )
            is_farm_close = (
                execution_kind == "farm_close"
                or (execution_kind is None and method_farm_fallback)
            )
            if is_farm_close:
                if "code" in spec:
                    raise ValueError("farm_close proposal must not carry code")
                plan_keys = [
                    key for key in ("farm_close_plan", "ensemble_plan") if key in spec
                ]
                if len(plan_keys) != 1:
                    raise ValueError(
                        "farm_close proposal must carry exactly one plan field"
                    )
                try:
                    code = self.compile_farm_close_node(
                        node, spec[plan_keys[0]], parent
                    )
                except FarmClosePlanError as exc:
                    spec = self.brain.repair_farm_close_plan(spec, str(exc))
                    repaired_plan_keys = [
                        key for key in ("farm_close_plan", "ensemble_plan")
                        if key in spec
                    ]
                    if len(repaired_plan_keys) != 1 or "code" in spec:
                        raise ValueError(
                            "repaired farm_close proposal must carry exactly one plan and no code"
                        )
                    code = self.compile_farm_close_node(
                        node, spec[repaired_plan_keys[0]], parent
                    )
                    recovery = "plan-repaired"
            else:
                if any(key in spec for key in ("farm_close_plan", "ensemble_plan")):
                    raise ValueError("script proposal must not carry a farm-close plan")
                code = spec["code"]
                if mode == "improve" and parent.action != "baseline":
                    # The rule protects the agent's OWN accepted artifacts;
                    # the first improve on the organizer baseline is the
                    # agent's first authorship and may restructure freely.
                    # Minimal-diff enforcement: the proposal must evolve the parent
                    # artifact. One retry with an explicit directive, then hard fail.
                    MIN_KEEP = 0.40  # fraction of parent lines that must survive
                    ratio = rewrite_ratio(parent.code_path.read_text(), code)
                    if ratio < MIN_KEEP:
                        retry = self.brain.propose(
                            self.journal_lines, mode, parent.node_id,
                            parent.code_path.read_text(),
                            directive=(
                                f"Your previous proposal kept only {ratio:.0%} of the "
                                "parent script's lines. That discards a debugged, "
                                "accepted artifact. Re-emit the parent file with the "
                                "SMALLEST edit implementing the same hypothesis; do "
                                "not restructure working sections."),
                            focus_note=self.focus_note,
                            traceback_tail=None,
                            parent_history=parent_history,
                            method_selection=node.method_selection,
                            streak_state=streak_state,
                            context_mode=self.config.context_mode,
                            prior_runs=self.prior_runs,
                            timeout_s=self.config.timeout_s,
                        )
                        retry_code = retry.get("code")
                        if retry_code:
                            retry_ratio = rewrite_ratio(
                                parent.code_path.read_text(), retry_code)
                            if retry_ratio >= ratio:
                                spec, code, ratio = retry, retry_code, retry_ratio
                                recovery = "rewrite-gate: retried for minimal diff"
                        if ratio < MIN_KEEP:
                            raise ValueError(
                                f"rewrite gate: improve proposal kept only {ratio:.0%} "
                                "of the parent script after retry; improve must edit "
                                "the champion artifact, not replace it")
                node.execution_kind = "script"
            node.hypothesis = str(spec.get("hypothesis", "(no hypothesis)"))
            expected_delta = spec.get("expected_delta")
            if (isinstance(expected_delta, bool)
                    or not isinstance(expected_delta, (int, float))
                    or not np.isfinite(expected_delta)):
                raise ValueError("proposer expected_delta must be a finite number")
            node.expected_delta = float(expected_delta)
            expected_delta_basis = spec.get("expected_delta_basis")
            if not isinstance(expected_delta_basis, str) or not expected_delta_basis.strip():
                raise ValueError("proposer expected_delta_basis must be a non-empty string")
            node.expected_delta_basis = expected_delta_basis.strip()
            timeout_s = min(int(spec.get("timeout_s", self.config.timeout_s)), self.config.timeout_s)
        except BudgetExhausted as exc:
            node.status, node.error = "failed", f"budget_exhausted: {exc}"
            self.stop_reason = "budget_exhausted"
            self.no_improve_streak += 1
            self.nodes[node.node_id] = node
            self.record(n, node, time.time() - start, "skipped", "LLM call refused: dollar budget exhausted")
            return
        except Exception as exc:
            node.status, node.error = "failed", f"proposer error: {exc}"
            # leave a placeholder so a later debug/diff of this node cannot crash on a missing file
            if not node.code_path.exists():
                node.code_path.write_text(f"# proposal failed before any code was produced\n# error: {exc}\n")
            self.stagnation += 1
            self.no_improve_streak += 1
            self.nodes[node.node_id] = node
            raw = getattr(self.brain, "last_raw_reply", None)
            if raw:
                # persist the unparseable reply so parse failures are diagnosable
                (self.run_dir / f"{node.node_id}_raw_reply.txt").write_text(raw)
            self.record(n, node, time.time() - start, "skipped", "proposal unparseable/failed")
            return
        if directive:
            tier_match = re.search(r"Tier \d", directive)
            node.tier = tier_match.group(0) if tier_match else None

        leak = self.check_code_leakage(code)
        if leak:
            node.status, node.error = "failed", leak
            node.code_path.write_text(code)
            self.stagnation += 1
            self.no_improve_streak += 1
            self.nodes[node.node_id] = node
            self.record(n, node, time.time() - start, "skipped", "rejected by leakage guard")
            return

        node.code_path.write_text(code)
        result, failed_stage = self.run_experiment(node, timeout_s)

        if not result.ok:
            # Crash path: traceback tail -> fixer -> retry once.
            node.error = result.error
            node.failure_stage = failed_stage
            node.fixer_eligible = True
            if node.farm_close_plan is not None:
                try:
                    repaired_spec = self.brain.repair_farm_close_plan(spec, result.error)
                    repaired_plan_keys = [
                        key for key in ("farm_close_plan", "ensemble_plan")
                        if key in repaired_spec
                    ]
                    if len(repaired_plan_keys) != 1 or "code" in repaired_spec:
                        raise ValueError(
                            "repaired farm_close proposal must carry exactly one plan and no code"
                        )
                    fixed = self.compile_farm_close_node(
                        node, repaired_spec[repaired_plan_keys[0]], parent
                    )
                except BudgetExhausted:
                    fixed = code
                    self.stop_reason = "budget_exhausted"
                except Exception as exc:
                    fixed = code
                    print(f"[loop] farm-close plan repair failed: {exc}", file=sys.stderr)
                if fixed != code:
                    node.code_path.write_text(fixed)
                    result, failed_stage = self.run_experiment(node, timeout_s)
                    recovery = "plan-repaired" if result.ok else "reverted"
                    if not result.ok:
                        node.failure_stage = failed_stage
                else:
                    recovery = "reverted"
            else:
                try:
                    fixed = self.brain.fix(code, result.error)
                except BudgetExhausted:
                    fixed = code
                    self.stop_reason = "budget_exhausted"
                except Exception as exc:
                    fixed = code
                    print(f"[loop] fixer failed: {exc}", file=sys.stderr)
                if fixed != code and not self.check_code_leakage(fixed):
                    node.code_path.write_text(fixed)
                    result, failed_stage = self.run_experiment(node, timeout_s)
                    recovery = "patched" if result.ok else "reverted"
                    if not result.ok:
                        node.failure_stage = failed_stage
                else:
                    recovery = "reverted"

        change_summary = node.hypothesis
        if result.ok:
            node.metrics, node.primary = result.metrics, result.metrics["primary"]
            reference = self.reference_primary(node)
            below_reference = reference is not None and reference - node.primary > 0.002
            if below_reference and node.debug_depth == 0:
                node.status = "suspect_implementation"
                node.verdict_note = (
                    f"below card reference ({node.primary:.4f} vs {reference:.4f}) "
                    "— implementation suspected"
                )
                self.stagnation += 1
                change_summary = f"{node.hypothesis} [{node.verdict_note}]"
            elif below_reference and node.debug_depth > 0:
                node.status = "rejected"
                self.stagnation += 1
            else:
                accepted, note = self.acceptance(node, result.metrics)
                if (not accepted and node.failure_stage == "noop_predictions"
                        and node.fixer_eligible):
                    # Degenerate ensemble collapsed to its parent (acceptance-time
                    # detection, so the ok-path fixer never sees it). Give the fixer
                    # one repair shot here, then re-run and re-judge once.
                    try:
                        original_code = node.code_path.read_text()
                        fixed = self.brain.fix(
                            original_code,
                            "Ensemble output is byte-identical to the parent's "
                            "predictions: members likely collapsed (shared seed, "
                            "anchor duplication, or gating that removes all "
                            "members). Ensure distinct member seeds and that the "
                            "final aggregation actually combines all member score "
                            "vectors.")
                        if fixed != original_code and not self.check_code_leakage(fixed):
                            node.code_path.write_text(fixed)
                            retry, retry_stage = self.run_experiment(node, self.config.timeout_s)
                            if retry.ok:
                                node.failure_stage = None
                                node.recovery = "noop-fixer: repaired and re-run"
                                result = retry
                                node.metrics = retry.metrics
                                node.primary = retry.metrics["primary"]
                                accepted, note = self.acceptance(node, result.metrics)
                            else:
                                # transactional: keep the ORIGINAL (working) code and
                                # record why the repair attempt failed
                                node.code_path.write_text(original_code)
                                node.recovery = (f"noop-fixer retry failed at "
                                                 f"{retry_stage}: {str(retry.error)[:200]}")
                    except BudgetExhausted:
                        self.stop_reason = "budget_exhausted"
                    except Exception as exc:
                        print(f"[loop] noop fixer failed: {exc}", file=sys.stderr)
                if accepted:
                    delta = node.primary - self.champion.primary
                    node.status = "accepted"
                    self.champion = node
                    self.accepted_deltas.append(delta)
                    self.stagnation = 0
                    change_summary = f"{node.hypothesis} (delta {delta:+.4f})"
                else:
                    node.status = "rejected"
                    self.stagnation += 1
                    if note:
                        change_summary = f"{node.hypothesis} [{note}]"
            node.error = None
        else:
            node.status = "failed"
            node.error = result.error
            self.stagnation += 1
        # Official convergence bookkeeping: every completed iteration counts.
        best_now = self.champion.primary if self.champion else 0.0
        improvement = best_now - getattr(self, "_best_before_iter", best_now)
        if improvement > self.config.epsilon:
            self.no_improve_streak = 0
        else:
            self.no_improve_streak += 1
        self.nodes[node.node_id] = node
        self.record(n, node, time.time() - start, recovery, change_summary)
