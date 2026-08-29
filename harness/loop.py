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
    n_converge: int = 3
    stagnation_limit: int = 5
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
    cross_run_path: Path = CROSS_RUN_PATH

    def __post_init__(self) -> None:
        if self.context_mode not in ("compact", "full"):
            raise ValueError("context_mode must be 'compact' or 'full'")
        if self.dataset not in ("pure", "1k"):
            raise ValueError("dataset must be 'pure' or '1k'")


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

    # ---------- workspace & leakage guard ----------

    def prepare_workspace(self) -> None:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
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
        env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
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
        if node.action in ("draft", "improve"):
            smoke = self.run_script(
                node.code_path,
                self.run_dir / f"{node.node_id}_smoke",
                self.config.seed,
                120,
                smoke_epochs=1,
            )
            if not smoke.ok:
                return smoke, "smoke"
        return (
            self.run_script(
                node.code_path, self.run_dir / node.node_id, self.config.seed, timeout_s
            ),
            "full",
        )

    # ---------- calibration ----------

    def calibrate(self) -> None:
        """Baseline node_000 + sigma from 3 seeds (CONTRACTS.md acceptance)."""
        code = self.config.baseline_script.read_text()
        node = Node("node_000", "baseline", "draft", "baseline FM (zoo/fm_torch.py)",
                    self.nodes_dir / "000.py")
        node.code_path.write_text(code)
        primaries = []
        # With an externally supplied sigma (tests / reruns) one baseline run suffices.
        seeds = self.config.calib_seeds if self.config.sigma is None else (self.config.seed,)
        for i, seed in enumerate(seeds):
            # Calibration runs trusted baseline code: never let a tight experiment
            # timeout (e.g. in tests) starve it.
            result = self.run_script(node.code_path, self.run_dir / f"calib_seed{seed}", seed,
                                     max(self.config.timeout_s, 120))
            if not result.ok:
                raise RuntimeError(f"baseline calibration failed (seed {seed}): {result.error}")
            primaries.append(result.metrics["primary"])
            if seed == self.config.seed or i == 0:
                node.metrics, node.primary = result.metrics, result.metrics["primary"]
        if self.config.sigma is not None:
            self.sigma = self.config.sigma
        else:
            self.sigma = float(np.std(primaries))
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
        record = {
            "n": 0, "node_id": "node_000", "parent": "baseline", "action": "reproduce_baseline",
            "hypothesis": "reproduce official FM baseline and calibrate seed noise",
            "change_summary": f"baseline seeds {list(self.config.calib_seeds) if self.config.sigma is None else [self.config.seed]}: "
                              f"primaries {[round(p, 4) for p in primaries]}, mean {mean:.4f}, sigma {self.sigma:.4f}",
            "diff": "",
            "context_mode": self.config.context_mode,
            "method_selection": None,
            "metrics": self.champion.metrics if self.champion else {},
            "val_best_so_far": self.champion.primary if self.champion else 0.0,
            "baseline_reproduction": {"seed_primaries": [round(p, 6) for p in primaries],
                                       "mean": round(mean, 6), "sigma": round(self.sigma, 6),
                                       "published_valid_primary": 0.6016,
                                       "pass": abs(mean - 0.6016) <= 0.003},
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

    # ---------- policy (harness-owned) ----------

    def next_move(self) -> tuple[str, Node, str | None]:
        """Return (mode, parent_node, directive)."""
        drafts = [n for n in self.nodes.values() if n.action == "draft" and n.node_id != "node_000"]
        last = self.nodes[max(self.nodes, key=lambda k: int(k.split("_")[1]))]
        if last.status in ("failed", "suspect_implementation") and last.debug_depth < 2:
            return "debug", last, None
        if len(drafts) < len(self.config.draft_tiers):
            tier = self.config.draft_tiers[len(drafts)]
            return "draft", self.champion, f"draft from {tier} of the menu"
        if self.stagnation >= self.config.stagnation_limit:
            tried = {n.tier for n in self.nodes.values() if n.tier}
            untried = [t for t in ALL_TIERS if t not in tried and t not in self.forced_tiers_used]
            tier = untried[0] if untried else ALL_TIERS[len(self.forced_tiers_used) % len(ALL_TIERS)]
            self.forced_tiers_used.append(tier)
            self.stagnation = 0
            return "draft", self.champion, (
                f"forced branch after stagnation: draft from {tier}, an untried menu tier"
            )
        return "improve", self.champion, None

    # ---------- acceptance ----------

    def acceptance(self, node: Node, metrics: dict) -> tuple[bool, str | None]:
        """Returns (accepted, note). May run one confirm reseed."""
        delta = metrics["primary"] - self.champion.primary
        threshold = max(2 * self.sigma, self.config.epsilon)
        if delta >= threshold:
            return True, None
        if delta > 0:
            confirm = self.run_script(node.code_path, self.run_dir / f"{node.node_id}_confirm",
                                      self.config.confirm_seed, self.config.timeout_s)
            if confirm.ok:
                mean_delta = ((metrics["primary"] + confirm.metrics["primary"]) / 2
                              - self.champion.primary)
                if mean_delta >= max(self.sigma, 0.001):
                    return True, f"grey-zone confirm passed (mean delta {mean_delta:+.4f})"
                return False, f"grey-zone confirm failed (mean delta {mean_delta:+.4f})"
            return False, f"grey-zone confirm run failed: {confirm.error}"
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
        selection = self.brain.select_method(
            self.journal_lines, parent_history, streak_state,
            excluded_families=excluded,
            dataset=self.config.dataset,
            prior_runs=self.prior_runs,
        )
        eligible = self.eligible_unexcluded_methods(excluded)
        if mode != "draft" or not eligible or selection.get("chosen_method_id") in eligible:
            return selection
        selection = self.brain.select_method(
            self.journal_lines, parent_history, streak_state,
            excluded_families=excluded,
            enforce_family_exclusion=True,
            dataset=self.config.dataset,
            prior_runs=self.prior_runs,
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
        selection = node.method_selection or {}
        return self.method_metadata(selection.get("chosen_method_id"))["reference_primary"]

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
            "recovery": recovery,
            "usd_total": round(getattr(self.brain, "usd_total", 0.0), 4),
            "intervention": False,
        }
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
        path = self.config.cross_run_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return "\n".join(path.read_text().splitlines()[-max_lines:])

    @staticmethod
    def _eight_word_summary(hypothesis: str) -> str:
        return " ".join(hypothesis.split()[:8])

    def append_cross_run(self, summary: dict, self_critique: str | None = None) -> None:
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
        journal as 'team-provided reference implementation'."""
        n = start_n
        for script in self.config.seed_scripts:
            n += 1
            start = time.time()
            self._best_before_iter = self.champion.primary if self.champion else 0.0
            node = Node(f"node_{n:03d}", "node_000", "draft",
                        f"team-provided reference implementation: {script.name} (from MENU frozen stack)",
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

    def run(self) -> dict:
        self.prior_runs = self.read_cross_run()
        self.prepare_workspace()
        self.calibrate()
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
            parent_history = (parent.metrics or {}).get("history", [])
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
            )
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
            code = spec["code"]
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
