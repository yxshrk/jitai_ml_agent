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

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "zoo" / "fm_torch.py"
METRIC_KEYS = ("gauc", "ndcg5", "primary")
FORBIDDEN_PATTERNS = ("test.csv", "data/test")
DRAFT_TIERS = ("Tier 1", "Tier 2", "Tier 3")
ALL_TIERS = ("Tier 1", "Tier 2", "Tier 3", "Tier 4")


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
    status: str = "pending"  # accepted | rejected | failed
    error: str | None = None
    debug_depth: int = 0


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
        self.stagnation = 0
        self.focus_note: str | None = None
        self.forced_tiers_used: list[str] = []
        self.start_time = time.time()
        self.stop_reason: str | None = None

    # ---------- workspace & leakage guard ----------

    def prepare_workspace(self) -> None:
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        for split in ("train.csv", "val.csv"):
            source = self.config.data_dir / split
            if not source.exists():
                raise FileNotFoundError(source)
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

    def run_script(self, script: Path, out_dir: Path, seed: int, timeout_s: int) -> RunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(script), "--data-dir", str(self.workspace),
               "--out-dir", str(out_dir), "--seed", str(seed)]
        env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
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
            result = self.run_script(node.code_path, self.run_dir / f"calib_seed{seed}", seed,
                                     self.config.timeout_s)
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
        self.nodes[node.node_id] = node
        self.champion = node
        self.journal_lines.append(
            f'node_000 [baseline] draft "baseline FM" primary={node.primary:.4f} '
            f"ACCEPTED (sigma={self.sigma:.4f})"
        )

    # ---------- policy (harness-owned) ----------

    def next_move(self) -> tuple[str, Node, str | None]:
        """Return (mode, parent_node, directive)."""
        drafts = [n for n in self.nodes.values() if n.action == "draft" and n.node_id != "node_000"]
        last = self.nodes[max(self.nodes, key=lambda k: int(k.split("_")[1]))]
        if last.status == "failed" and last.debug_depth < 2:
            return "debug", last, None
        if len(drafts) < 3:
            tier = DRAFT_TIERS[len(drafts)]
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

    # ---------- bookkeeping ----------

    def record(self, n: int, node: Node, duration: float, recovery: str | None,
               change_summary: str) -> None:
        record = {
            "n": n,
            "hypothesis": node.hypothesis,
            "node_id": node.node_id,
            "parent": node.parent,
            "action": node.action,
            "code_path": str(node.code_path.relative_to(ROOT)) if node.code_path.is_relative_to(ROOT)
                         else str(node.code_path),
            "change_summary": change_summary,
            "metrics": node.metrics or {"gauc": 0.0, "ndcg5": 0.0, "primary": 0.0},
            "val_best_so_far": self.champion.primary if self.champion else 0.0,
            "accepted": node.status == "accepted",
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
        n = self.config.n_converge
        return (len(self.accepted_deltas) >= n
                and all(d < self.config.epsilon for d in self.accepted_deltas[-n:]))

    # ---------- main loop ----------

    def run(self) -> dict:
        self.prepare_workspace()
        self.calibrate()
        n = 0
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
                try:
                    self.focus_note = self.brain.reflect(self.journal_lines)
                except Exception as exc:  # reflection is optional; never kills the loop
                    self.focus_note = None
                    print(f"[loop] reflector failed: {exc}", file=sys.stderr)
            self.iterate(n)
            if self.stop_reason:
                break
        else:
            self.stop_reason = self.stop_reason or "max_iters"
        summary = {
            "run_dir": str(self.run_dir),
            "stop_reason": self.stop_reason,
            "iterations": n,
            "best_node": self.champion.node_id,
            "best_metrics": self.champion.metrics,
            "sigma": self.sigma,
            "tokens": self.brain.meter.per_role,
            "tokens_total": self.brain.meter.total,
            "wall_s": round(time.time() - self.start_time, 1),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def iterate(self, n: int) -> None:
        start = time.time()
        mode, parent, directive = self.next_move()
        node = Node(f"node_{n:03d}", parent.node_id, mode, "(proposal failed)",
                    self.nodes_dir / f"{n:03d}.py")
        if mode == "debug":
            node.debug_depth = parent.debug_depth + 1
        recovery: str | None = None
        try:
            spec = self.brain.propose(
                self.journal_lines, mode, parent.node_id, parent.code_path.read_text(),
                directive=directive, focus_note=self.focus_note,
                traceback_tail=parent.error if mode == "debug" else None,
            )
            node.hypothesis = str(spec.get("hypothesis", "(no hypothesis)"))
            code = spec["code"]
            timeout_s = min(int(spec.get("timeout_s", self.config.timeout_s)), self.config.timeout_s)
        except BudgetExhausted as exc:
            node.status, node.error = "failed", f"budget_exhausted: {exc}"
            self.stop_reason = "budget_exhausted"
            self.nodes[node.node_id] = node
            self.record(n, node, time.time() - start, "skipped", "LLM call refused: dollar budget exhausted")
            return
        except Exception as exc:
            node.status, node.error = "failed", f"proposer error: {exc}"
            self.stagnation += 1
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
            self.nodes[node.node_id] = node
            self.record(n, node, time.time() - start, "skipped", "rejected by leakage guard")
            return

        node.code_path.write_text(code)
        result = self.run_script(node.code_path, self.run_dir / node.node_id, self.config.seed, timeout_s)

        if not result.ok:
            # Crash path: traceback tail -> fixer -> retry once.
            node.error = result.error
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
                result = self.run_script(node.code_path, self.run_dir / node.node_id,
                                         self.config.seed, timeout_s)
                recovery = "patched" if result.ok else "reverted"
            else:
                recovery = "reverted"

        change_summary = node.hypothesis
        if result.ok:
            node.metrics, node.primary = result.metrics, result.metrics["primary"]
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
        self.nodes[node.node_id] = node
        self.record(n, node, time.time() - start, recovery, change_summary)
