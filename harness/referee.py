"""Deterministic referee (ADR-0007). No LLM in this file.

Runs an experiment script in a subprocess with a timeout, validates its predictions against the valid split,
scores them with the official evaluate.py, and applies the acceptance and convergence rules."""
import csv, json, math, os, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from . import config as C

sys.path.insert(0, str(C.KIT))
from evaluate import evaluate  # noqa: E402  -- the official scorer, imported from the untouched kit

HEADER = ['row_id', 'user_id', 'video_id', 'score']
_VALID = None

def valid_index():
    """(user_ids, video_ids, labels) of the valid split in row_id order, cached."""
    global _VALID
    if _VALID is None:
        uid, vid, y = [], [], []
        with open(C.WS_DATA / 'valid.csv', newline='') as fh:
            for row in csv.DictReader(fh):
                uid.append(row['user_id']); vid.append(row['video_id']); y.append(1 if row['long_view'] != '0' else 0)
        _VALID = (uid, vid, y)
    return _VALID

def static_check(code: str):
    """Strings an agent script may not contain (firewall). Returns the offending patterns."""
    return [p for p in C.FORBIDDEN_PATTERNS if p in code]

def read_predictions(path):
    """Parse and validate a predictions.csv against the valid split; return the scores (floats)."""
    uid, vid, y = valid_index()
    scores = []
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        head = next(r, None)
        if head != HEADER:
            raise ValueError(f'header must be {",".join(HEADER)}, got {head}')
        for n, rec in enumerate(r):
            if n >= len(uid):
                raise ValueError(f'more rows than the valid split ({len(uid)})')
            if len(rec) != 4:
                raise ValueError(f'row {n}: {len(rec)} fields, expected 4')
            if int(rec[0]) != n:
                raise ValueError(f'row {n}: row_id={rec[0]}, expected {n}')
            if rec[1] != uid[n] or rec[2] != vid[n]:
                raise ValueError(f'row {n} misaligned: got ({rec[1]},{rec[2]}), valid split has ({uid[n]},{vid[n]})')
            v = float(rec[3])
            if not math.isfinite(v):
                raise ValueError(f'row {n}: score is NaN/Inf')
            scores.append(v)
    if len(scores) != len(uid):
        raise ValueError(f'{len(scores)} rows, valid split has {len(uid)}')
    return scores

_COHORTS = None
def valid_cohorts():
    """Shares of all-negative / all-positive / discriminative users in valid (constants of the split)."""
    global _COHORTS
    if _COHORTS is None:
        uid, _, y = valid_index()
        pos, cnt = {}, {}
        for u, l in zip(uid, y):
            pos[u] = pos.get(u, 0) + l; cnt[u] = cnt.get(u, 0) + 1
        n = len(cnt); an = sum(1 for u in cnt if pos[u] == 0); ap = sum(1 for u in cnt if pos[u] == cnt[u])
        _COHORTS = {'all_neg': an / n, 'all_pos': ap / n, 'disc': (n - an - ap) / n}
    return _COHORTS

def score(scores):
    """Official metrics plus `ndcg5_disc`: nDCG@5 restricted to discriminative users, the sharper diagnostic
    (nDCG = all_pos*1 + disc*ndcg_disc + all_neg*0)."""
    uid, _, y = valid_index()
    m = evaluate(uid, y, scores); c = valid_cohorts()
    disc = (m['nDCG@5'] - c['all_pos']) / c['disc'] if c['disc'] else float('nan')
    return {'gauc': m['GAUC'], 'ndcg5': m['nDCG@5'], 'primary': m['primary'], 'ndcg5_disc': disc}

@dataclass
class RunResult:
    stage: str = 'full'                 # static | smoke | full
    ok: bool = False
    exit_code: Optional[int] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    metrics: Optional[dict] = None
    history: list = field(default_factory=list)
    log_tail: str = ''
    out_dir: str = ''
    def to_dict(self): return asdict(self)

def run_script(code_path, out_dir, seed=C.DEFAULT_SEED, smoke=False, score_extra=None, timeout=None, extra_args=(), threads=None):
    """Execute one experiment script under the contract; validate and score its valid predictions."""
    code_path, out_dir = Path(code_path).resolve(), Path(out_dir).resolve()   # the subprocess runs with cwd=workspace
    res = RunResult(stage='smoke' if smoke else 'full', out_dir=str(out_dir))
    code = code_path.read_text()
    hits = static_check(code)
    if hits:
        res.stage, res.error = 'static', f'forbidden reference(s) in script: {hits}'
        return res
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(code_path), '--data-dir', str(C.WS_DATA), '--out-dir', str(out_dir), '--seed', str(seed), *extra_args]
    if score_extra:
        cmd += ['--score-extra', str(score_extra)]
    # scripts import the official scorer with `from evaluate import evaluate`; Python puts the script's own
    # directory on sys.path, not the cwd, so the workspace must be on PYTHONPATH explicitly
    env = dict(os.environ, PYTHONHASHSEED='0',
               PYTHONPATH=str(C.WORKSPACE) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
    if smoke:
        env['SMOKE_EPOCHS'] = '1'
    if threads:   # parallel branches must not oversubscribe the cores
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[var] = str(threads)
    timeout = timeout or (C.SMOKE_TIMEOUT_S if smoke else C.RUN_TIMEOUT_S)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(C.WORKSPACE), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        res.duration_s = time.time() - t0
        res.error = f'timeout after {timeout}s'
        res.log_tail = ((e.stdout or b'')[-1500:].decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')[-1500:])
        return res
    res.duration_s = time.time() - t0
    res.exit_code = p.returncode
    res.log_tail = (p.stdout[-1500:] + ('\n--- stderr ---\n' + p.stderr[-2500:] if p.stderr else '')).strip()
    if p.returncode != 0:
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        res.error = f'exit code {p.returncode}: ' + (lines[-1] if lines else 'no stderr')
        return res
    try:
        scores = read_predictions(out_dir / 'predictions.csv')
    except Exception as e:  # missing file, bad format, misalignment
        res.error = f'invalid predictions.csv: {e}'
        return res
    res.metrics = score(scores)
    mj = out_dir / 'metrics.json'
    if mj.exists():
        try:
            res.history = json.loads(mj.read_text()).get('history', [])
        except Exception:
            pass
    res.ok = True
    return res

def accept(champion_primary, new_primary):
    """A node replaces the champion only if it beats it by at least EPS (2.5 sigma of seed noise)."""
    delta = new_primary - champion_primary
    return delta >= C.EPS, delta

class Convergence:
    """Official rule: converged when validation primary has not improved by more than EPS over the last
    N consecutive iterations. Errored iterations count as non-improving."""
    def __init__(self, best_primary):
        self.best = best_primary; self.streak = 0
    def update(self, primary):
        if primary is not None and primary > self.best + C.EPS:
            self.best, self.streak = primary, 0
        else:
            self.streak += 1
        return self.streak >= C.N_CONVERGE
    @property
    def iters_left_before_convergence(self):
        return max(0, C.N_CONVERGE - self.streak)
