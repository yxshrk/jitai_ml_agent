"""Feature screen (ADR-0015): a deterministic gate between a feature hypothesis and a node. No LLM in this file.

A probe is a small script (`probe.py --data-dir D --out-dir O`) that writes O/features.csv: `row_id` plus one or more
numeric columns, one row per valid.csv row in file order. It runs like a node (workspace cwd, static firewall, thread
env, timeout) but on a PROBE DATA DIRECTORY whose valid.csv has every outcome column stripped, so a probe cannot use
the scored row's label whatever its code says. The screen then measures what the metric rewards — within-user
discrimination on valid — against the champion's own valid predictions:
  varies    share of users whose valid rows differ on the column (a column constant within a user cannot reorder anything)
  gauc      standalone within-user GAUC of the column, best sign
  additive  best Δprimary of z(champion) + w·z(column) over a small weight grid
  stack     Δprimary of a lambdarank LightGBM on [champion score + all columns] against [champion score] alone,
            5-fold cross-validation over users (interactions the additive test cannot see)
best_gain = max(additive over columns, stack). The loop drops a candidate whose best_gain is below C.SCREEN_MIN_GAIN.

Why (measured 2026-08-31, kb/data/screens): item target statistics with standalone GAUC 0.63-0.65 add <= +0.0004 on the
FM; session-position features vary within a user's rows for 37 % of users and add nothing; a 63-feature stack adds
+0.0007. The FM already carries what those tables know. The screen lets a run learn that in about a minute of compute
instead of an Implementer, a Critic, a 30-minute run and three confirmation seeds."""
import csv, math, os, subprocess, sys, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np
from . import config as C
from . import referee as R

OUTCOME_COLUMNS = ('long_view', 'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
                   'play_time_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter')
MAX_COLUMNS = 8
BLEND_WEIGHTS = (0.1, 0.25, 0.5, 1.0)
STACK_FOLDS = 5

def probe_data_dir(force=False):
    """workspace/data_probe: every file of workspace/data (linked), but valid.csv WITHOUT the outcome columns.
    Rebuilt when valid.csv changes. This is what makes label-freeness a property of the sandbox, not of the prompt."""
    src, dst = C.WS_DATA, C.WORKSPACE / 'data_probe'
    v = src / 'valid.csv'; sig = f'{v.stat().st_mtime_ns}:{v.stat().st_size}'
    stamp = dst / '.stamp'
    if not force and stamp.exists() and stamp.read_text() == sig:
        return dst
    dst.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.name == 'valid.csv' or p.name.startswith('.'):
            continue
        t = dst / p.name
        if t.exists() or t.is_symlink():
            t.unlink()
        os.symlink(p.resolve(), t)
    with open(v, newline='') as fh, open(dst / 'valid.csv', 'w', newline='') as out:
        r = csv.reader(fh); head = next(r)
        keep = [i for i, c in enumerate(head) if c not in OUTCOME_COLUMNS]
        w = csv.writer(out); w.writerow([head[i] for i in keep])
        for row in r:
            w.writerow([row[i] for i in keep])
    stamp.write_text(sig)
    return dst

def read_features(path, n_rows):
    """features.csv -> (names, float array [n_rows, m]); validates alignment (row_id 0..n-1), finiteness, column count."""
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r, None)
        if not head or head[0] != 'row_id' or len(head) < 2:
            raise ValueError(f'header must be row_id,<name>[,<name>...], got {head}')
        names = head[1:]
        if len(names) > MAX_COLUMNS:
            raise ValueError(f'{len(names)} columns; a probe writes at most {MAX_COLUMNS}')
        if len(set(names)) != len(names):
            raise ValueError('duplicate column names')
        F = np.empty((n_rows, len(names)), dtype=np.float64)
        n = 0
        for rec in r:
            if n >= n_rows:
                raise ValueError(f'more rows than the valid split ({n_rows})')
            if len(rec) != len(head):
                raise ValueError(f'row {n}: {len(rec)} fields, expected {len(head)}')
            if int(rec[0]) != n:
                raise ValueError(f'row {n}: row_id={rec[0]}, expected {n}')
            F[n] = [float(x) for x in rec[1:]]; n += 1
        if n != n_rows:
            raise ValueError(f'{n} rows, valid split has {n_rows}')
    if not np.isfinite(F).all():
        raise ValueError('non-finite feature values')
    return names, F

def _groups(uid):
    u = np.asarray(uid); order = np.argsort(u, kind='stable'); us = u[order]
    starts = np.r_[0, np.flatnonzero(us[1:] != us[:-1]) + 1, len(us)]
    return [order[a:b] for a, b in zip(starts[:-1], starts[1:])]

def _z(f):
    f = np.asarray(f, dtype=np.float64); sd = f.std()
    return (f - f.mean()) / sd if sd > 0 else np.zeros_like(f)

def _metrics(uid, y, s):
    m = R.evaluate(uid, y, np.asarray(s, dtype=np.float64).tolist())
    return m['GAUC'], m['primary']

def screen_columns(names, F, uid, y, champ):
    """Per column: varies, standalone GAUC (best sign, with the sign), additive Δprimary on the champion (best weight)."""
    groups = _groups(uid); zc = _z(champ); _, p0 = _metrics(uid, y, champ); out = {}
    for i, nm in enumerate(names):
        f = F[:, i]
        varies = float(np.mean([len(np.unique(f[g])) > 1 for g in groups]))
        if varies == 0.0:
            out[nm] = {'varies': 0.0, 'gauc': 0.5, 'sign': 0, 'additive': 0.0, 'w': 0.0}; continue
        zf = _z(f)
        gp, _ = _metrics(uid, y, zf); gm, _ = _metrics(uid, y, -zf)
        sign = 1 if gp >= gm else -1
        best_w, best_d = 0.0, -1.0
        for w in BLEND_WEIGHTS:
            _, p = _metrics(uid, y, zc + w * sign * zf)
            if p - p0 > best_d:
                best_w, best_d = w, p - p0
        out[nm] = {'varies': round(varies, 3), 'gauc': round(max(gp, gm), 4), 'sign': sign, 'additive': round(best_d, 5), 'w': best_w}
    return out

def stack_gain(F, uid, y, champ, threads=1, seed=0):
    """Δprimary of lambdarank LightGBM on [champion + columns] vs [champion], 5-fold CV over users; None without lightgbm."""
    try:
        import lightgbm as lgb
    except Exception:   # noqa: BLE001
        return None
    u = np.asarray(uid); yv = np.asarray(y, dtype=np.float64); order = np.argsort(u, kind='stable')
    u, yv, Fs, cs = u[order], yv[order], F[order], np.asarray(champ, dtype=np.float64)[order]
    X = np.column_stack([cs, Fs]); X0 = cs[:, None]
    users = np.unique(u); rng = np.random.RandomState(seed); fold_of = dict(zip(users, rng.randint(0, STACK_FOLDS, len(users))))
    fold = np.array([fold_of[x] for x in u])
    def sizes(mask):
        uu = u[mask]; return np.diff(np.r_[0, np.flatnonzero(uu[1:] != uu[:-1]) + 1, len(uu)])
    params = dict(objective='lambdarank', learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, lambda_l2=10.0, feature_fraction=1.0,
                  num_threads=max(1, int(threads or 1)), seed=seed, deterministic=True, force_row_wise=True, verbose=-1, eval_at=[5])
    def cv(Xm):
        oof = np.zeros(len(yv))
        for f in range(STACK_FOLDS):
            test = fold == f; inner = fold == (f + 1) % STACK_FOLDS; fit = ~test & ~inner
            ds = lgb.Dataset(Xm[fit], yv[fit], group=sizes(fit)); dv = lgb.Dataset(Xm[inner], yv[inner], group=sizes(inner), reference=ds)
            m = lgb.train(params, ds, num_boost_round=400, valid_sets=[dv], callbacks=[lgb.early_stopping(30, verbose=False)])
            oof[test] = m.predict(Xm[test], num_iteration=m.best_iteration)
        return oof
    uid_s, y_s = u.tolist(), yv.astype(int).tolist()
    _, p_with = _metrics(uid_s, y_s, cv(X)); _, p_without = _metrics(uid_s, y_s, cv(X0))
    return round(p_with - p_without, 5)

@dataclass
class ScreenResult:
    ok: bool = False
    stage: str = 'probe'                # static | probe | features | measure
    error: Optional[str] = None
    duration_s: float = 0.0
    columns: dict = field(default_factory=dict)
    stack_gain: Optional[float] = None
    best_gain: Optional[float] = None
    best_column: Optional[str] = None
    log_tail: str = ''
    def summary(self):
        return {'best_gain': self.best_gain, 'best_column': self.best_column, 'stack_gain': self.stack_gain, 'columns': self.columns,
                'duration_s': round(self.duration_s, 1), 'ok': self.ok, 'error': self.error}
    def text(self):
        if not self.ok:
            return f'screen failed at {self.stage}: {self.error}'
        cols = '; '.join(f"{n}: varies {c['varies']}, GAUC {c['gauc']}, additive {c['additive']:+.4f}" for n, c in self.columns.items())
        return f"best_gain {self.best_gain:+.4f} ({self.best_column}); stack {self.stack_gain if self.stack_gain is None else format(self.stack_gain, '+.4f')}; {cols}"

def run_probe(code_path, out_dir, champion_pred_csv, threads=None, timeout=None, seed=0):
    """Execute one probe script under the contract on the label-stripped data dir; measure its columns against the champion."""
    code_path, out_dir = Path(code_path).resolve(), Path(out_dir).resolve()
    res = ScreenResult(); t0 = time.time()
    hits = R.static_check(code_path.read_text())
    if hits:
        res.stage, res.error = 'static', f'forbidden reference(s) in probe: {hits}'; return res
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(code_path), '--data-dir', str(probe_data_dir()), '--out-dir', str(out_dir)]
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(C.WORKSPACE) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''))
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = str(threads or 1)
    timeout = timeout or C.SCREEN_TIMEOUT_S
    try:
        p = subprocess.run(cmd, cwd=str(C.WORKSPACE), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        res.duration_s = time.time() - t0; res.error = f'timeout after {timeout}s'
        res.log_tail = ((e.stdout or b'')[-1000:].decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')[-1000:])
        return res
    res.log_tail = (p.stdout[-1000:] + ('\n--- stderr ---\n' + p.stderr[-1500:] if p.stderr else '')).strip()
    if p.returncode != 0:
        lines = [l for l in p.stderr.strip().splitlines() if l.strip()]
        res.duration_s = time.time() - t0; res.error = f'exit code {p.returncode}: ' + (lines[-1] if lines else 'no stderr'); return res
    uid, _, y = R.valid_index()
    try:
        names, F = read_features(out_dir / 'features.csv', len(uid))
    except Exception as e:   # noqa: BLE001
        res.stage, res.error, res.duration_s = 'features', f'invalid features.csv: {e}', time.time() - t0; return res
    try:
        champ = R.read_predictions(champion_pred_csv)
        res.stage = 'measure'
        res.columns = screen_columns(names, F, uid, y, champ)
        live = [i for i, n in enumerate(names) if res.columns[n]['varies'] > 0]
        res.stack_gain = stack_gain(F[:, live], uid, y, champ, threads=threads, seed=seed) if live else 0.0
        best_col = max(res.columns, key=lambda n: res.columns[n]['additive'])
        add = res.columns[best_col]['additive']
        res.best_gain, res.best_column = (add, best_col) if res.stack_gain is None or add >= res.stack_gain else (res.stack_gain, 'stack')
        res.ok = True
    except Exception as e:   # noqa: BLE001
        res.error = f'{type(e).__name__}: {str(e)[:300]}'
    res.duration_s = time.time() - t0
    return res

def passes(res, min_gain=None):
    """The gate: a failed screen never blocks (the candidate proceeds and the failure is journaled)."""
    if not res.ok or res.best_gain is None:
        return True
    return res.best_gain >= (C.SCREEN_MIN_GAIN if min_gain is None else min_gain)
