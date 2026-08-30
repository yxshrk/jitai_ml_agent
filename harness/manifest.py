"""Launch-time provenance manifest for agent runs.

write_manifest(run_dir, config) records, before the first LLM call, everything
needed to reproduce or audit the run. Failures never crash the run.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_manifest(run_dir: Path, config: object) -> None:
    try:
        cfg = {k: str(v) for k, v in vars(config).items()} if config else {}
        try:
            import numpy, torch
            pkg = {"numpy": numpy.__version__, "torch": torch.__version__}
        except Exception:
            pkg = {}
        manifest = {
            "run_id": run_dir.name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": sys.argv,
            "prompt_sha256": _sha(ROOT / "agent/prompts.py"),
            "methods_sha256": _sha(ROOT / "agent/METHODS.md"),
            "contracts_sha256": _sha(ROOT / "CONTRACTS.md"),
            "evaluator_sha256": _sha(ROOT / "data/official/evaluate.py"),
            "config": cfg,
            "python": sys.version.split()[0],
            "packages": pkg,
            "platform": platform.platform(),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    except Exception:
        pass
