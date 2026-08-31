"""Deterministic referee (ADR-0007). No LLM in this file.

Runs an experiment script in a subprocess with a timeout, validates its predictions against the valid split,
scores them with the official evaluate.py, and applies the acceptance and convergence rules."""
import csv, json, math, os, subprocess, sys, time
import hashlib
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
        uid, vid, y, tab, dur, date, tms = [], [], [], [], [], [], []
        with open(C.WS_DATA / 'valid.csv', newline='') as fh:
            for row in csv.DictReader(fh):
                uid.append(row['user_id']); vid.append(row['video_id']); y.append(1 if row['long_view'] != '0' else 0)
                tab.append(row.get('tab', '')); dur.append(float(row.get('duration_ms') or 0))
                date.append(row.get('date', '')); tms.append(int(row.get('time_ms') or 0))
        _VALID = (uid, vid, y); _VALID_EXTRA.update({'tab': tab, 'dur': dur, 'date': date, 'time': tms})
    return _VALID

_VALID_EXTRA, _GROUPS = {}, None
MIN_GROUP_ROWS = 2000
def valid_groups():
    """Row-index groups of the valid split for the diagnostic breakdown: each tab, and duration bands that follow the
    label's definition (0 = always negative; < 18 s must be watched in full; 18-60 s; 60-180 s; > 180 s). Groups with
    fewer than MIN_GROUP_ROWS rows are dropped. Cached."""
    global _GROUPS
    if _GROUPS is None:
        valid_index(); tab, dur = _VALID_EXTRA['tab'], _VALID_EXTRA['dur']
        g = {}
        for i, (t, d) in enumerate(zip(tab, dur)):
            g.setdefault(f'tab={t}', []).append(i)
            band = 'dur=0' if d <= 0 else 'dur<18s' if d < 18000 else 'dur18-60s' if d < 60000 else 'dur60-180s' if d < 180000 else 'dur>180s'
            g.setdefault(band, []).append(i)
        _GROUPS = {k: v for k, v in sorted(g.items()) if len(v) >= MIN_GROUP_ROWS}
    return _GROUPS

def group_breakdown(scores):
    """Official metrics per tab and per duration band — the Diagnostician's map of where a script is wrong."""
    uid, _, y = valid_index(); out = {}
    for name, idx in valid_groups().items():
        m = evaluate([uid[i] for i in idx], [y[i] for i in idx], [scores[i] for i in idx])
        out[name] = {'rows': len(idx), 'gauc': round(m['GAUC'], 4), 'ndcg5': round(m['nDCG@5'], 4), 'primary': round(m['primary'], 4)}
    return out

PAIR_TYPES = ('same_tab', 'diff_tab', 'tab1_x_tab1', 'same_date', 'diff_date', 'gap>1d', 'gap<10min', 'pos_shorter', 'pos_longer')
def pair_breakdown(scores):
    """GAUC error attributed to positive-negative pair types (the ceiling study, kb/data/screens/CEILING.md).
    Each discriminative user's pairs are weighted 1/#negatives so the user counts #positives, as GAUC weights it;
    `share` = the type's share of the pair mass, `err` = weighted misordered fraction (ties count 0.5),
    `contrib` = share * err; `total_err` = 1 - GAUC. Types overlap (they are different cuts of the same pairs)."""
    uid, _, y = valid_index(); tab, dur, date, tms = (_VALID_EXTRA[k] for k in ('tab', 'dur', 'date', 'time'))
    by_user = {}
    for i, u in enumerate(uid):
        by_user.setdefault(u, ([], []))[y[i]].append(i)
    mass = {t: 0.0 for t in PAIR_TYPES}; err = {t: 0.0 for t in PAIR_TYPES}; total_mass = total_err = 0.0
    for neg, pos in by_user.values():
        if not pos or not neg:
            continue
        w = 1.0 / len(neg)
        for p in pos:
            sp, tp, dp, datep, timep = scores[p], tab[p], dur[p], date[p], tms[p]
            for n in neg:
                e = 0.0 if sp > scores[n] else 0.5 if sp == scores[n] else 1.0
                gap = abs(timep - tms[n])
                types = ('same_tab' if tp == tab[n] else 'diff_tab', 'same_date' if datep == date[n] else 'diff_date')
                if tp == '1' and tab[n] == '1': types += ('tab1_x_tab1',)
                if gap > 86_400_000: types += ('gap>1d',)
                elif gap < 600_000: types += ('gap<10min',)
                if dur[n] > 0 and dp < dur[n]: types += ('pos_shorter',)
                elif dp > dur[n]: types += ('pos_longer',)
                for t in types:
                    mass[t] += w; err[t] += w * e
                total_mass += w; total_err += w * e
    out = {t: {'share': round(mass[t] / total_mass, 3), 'err': round(err[t] / mass[t], 3) if mass[t] else None,
               'contrib': round(err[t] / total_mass, 4)} for t in PAIR_TYPES}
    out['total_err'] = round(total_err / total_mass, 4) if total_mass else None
    return out

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

def score(scores, breakdown=False):
    """Official metrics plus `ndcg5_disc`: nDCG@5 restricted to discriminative users, the sharper diagnostic
    (nDCG = all_pos*1 + disc*ndcg_disc + all_neg*0); with breakdown=True also the per-tab / per-duration map and
    the pair-type attribution of the GAUC error (`by_pair`)."""
    uid, _, y = valid_index()
    m = evaluate(uid, y, scores); c = valid_cohorts()
    disc = (m['nDCG@5'] - c['all_pos']) / c['disc'] if c['disc'] else float('nan')
    out = {'gauc': m['GAUC'], 'ndcg5': m['nDCG@5'], 'primary': m['primary'], 'ndcg5_disc': disc}
    if breakdown:
        out['by_group'] = group_breakdown(scores); out['by_pair'] = pair_breakdown(scores)
    return out

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
    pred_hash: Optional[str] = None     # md5 of predictions.csv: byte-identical to the parent's = a no-op node
    def to_dict(self): return asdict(self)

def run_script(code_path, out_dir, seed=C.DEFAULT_SEED, smoke=False, score_extra=None, timeout=None, extra_args=(), threads=None,
               env_extra=None, data_dir=None):
    """Execute one experiment script under the contract; validate and score its valid predictions."""
    code_path, out_dir = Path(code_path).resolve(), Path(out_dir).resolve()   # the subprocess runs with cwd=workspace
    res = RunResult(stage='smoke' if smoke else 'full', out_dir=str(out_dir))
    code = code_path.read_text()
    hits = static_check(code)
    if hits:
        res.stage, res.error = 'static', f'forbidden reference(s) in script: {hits}'
        return res
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(code_path), '--data-dir', str(data_dir or C.WS_DATA), '--out-dir', str(out_dir), '--seed', str(seed), *extra_args]
    if score_extra:
        cmd += ['--score-extra', str(score_extra)]
    # scripts import the official scorer with `from evaluate import evaluate`; Python puts the script's own
    # directory on sys.path, not the cwd, so the workspace must be on PYTHONPATH explicitly
    env = dict(os.environ, PYTHONHASHSEED='0',
               PYTHONPATH=str(C.WORKSPACE) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
    if smoke:
        env['SMOKE_EPOCHS'] = '1'
    if env_extra:      # ADR-0019: the refit fixes the epoch count through the contract's SMOKE_EPOCHS cap
        env.update({k: str(v) for k, v in env_extra.items()})
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
    res.metrics = score(scores, breakdown=not smoke and seed == C.DEFAULT_SEED)   # the diagnostic map only for the screening seed
    res.pred_hash = hashlib.md5((out_dir / 'predictions.csv').read_bytes()).hexdigest()
    mj = out_dir / 'metrics.json'
    if mj.exists():
        try:
            res.history = json.loads(mj.read_text()).get('history', [])
        except Exception:
            pass
    res.ok = True
    return res

def accept(champion_primary, new_primary):
    """Single-seed screen, informational only: returns (delta >= EPS, delta). The decision that moves the champion
    is the seed-mean test in loop._confirm_with_seeds (ADR-0010/0011); every positive delta goes there."""
    delta = new_primary - champion_primary
    return delta >= C.EPS, delta

class Convergence:
    """ADR-0012 (revised by Yash): the streak counts consecutive generations in which the champion's fresh-seed mean
    has NOT risen by at least RESET_MIN_GAIN since the last reset — cumulative, like early stopping's min_delta
    against the best seen, so a staircase of small confirmed gains adds up while one false acceptance (observed gain
    ~ +0.0005-0.0007 under the null) cannot buy N_CONVERGE more generations. The seed test (fresh seeds, pooled SD,
    z >= Z_CRIT) is a stronger noise filter than a fixed eps on one seed; the organizers' literal rule is tracked
    alongside by OfficialRule and reported, never used to stop unless --convergence official."""
    def __init__(self, streak=0, ref=None):
        self.streak, self.ref = streak, ref
    def update(self, champion_mean):
        """Returns True iff the champion's mean has risen >= RESET_MIN_GAIN since the reference (which then moves)."""
        if champion_mean is None:
            self.streak += 1; return False
        if self.ref is None:
            self.ref = champion_mean
        if champion_mean - self.ref >= C.RESET_MIN_GAIN:
            self.ref, self.streak = champion_mean, 0
            return True
        self.streak += 1
        return False
    @property
    def converged(self):
        return self.streak >= C.N_CONVERGE
    @property
    def iters_left_before_convergence(self):
        return max(0, C.N_CONVERGE - self.streak)

class OfficialRule:
    """The organizers' rule read literally on single-seed validation primaries (best of any node): the reference
    moves only on a rise of more than EPS; N_CONVERGE generations without one = 'converged'. Kept for reporting so
    the judges can see where the literal rule would have stopped each run (summary.official_rule)."""
    def __init__(self, best, streak=0, converged_at=None):
        self.best, self.streak, self.converged_at = best, streak, converged_at
    def update(self, gen_best_single, generation):
        if gen_best_single is not None and gen_best_single > self.best + C.EPS:
            self.best, self.streak = gen_best_single, 0
        else:
            self.streak += 1
            if self.streak >= C.N_CONVERGE and self.converged_at is None:
                self.converged_at = generation
        return self.streak
    def to_dict(self):
        return {'best_single_seed': self.best, 'streak': self.streak, 'converged_at_generation': self.converged_at}
