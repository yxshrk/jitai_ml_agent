"""Designation-time refit (ADR-0019): train the submitted model on train + valid, epochs fixed at the number
validated on valid. No LLM, no search — a pipeline step applied to the node the run designated.

Why: the label window is what matters, not recency of the model. Measured (facts §11.3, research session): the
champion script trained on train up to 04-14 (78 % of the rows) scores 0.5999 against 0.6031 on the full window,
and a train+half-valid model scored on the other half is flat (0.5821 vs 0.5826) — volume, not recency. The hidden
test week follows the validation week, so a model that has also seen the validation week trains on ~11 % more rows
that are adjacent to the test period. Expected on the hidden test: +0.001 to +0.003; it cannot be measured on valid,
because valid is now inside the training data — which is exactly why the epoch count must come from the validated
run and early stopping must not be allowed to choose it again.

How, without rewriting the node's script: the contract gives two levers every script obeys. `--data-dir` points at
`workspace/data_refit`, whose train.csv is train ∪ valid (the valid rows carry their `long_view`; the outcome columns
valid does not have stay empty, so a script that needs them fails loudly instead of training on fabrications), and
`SMOKE_EPOCHS` caps every training phase — set to the node's validated `best_epoch`, it fixes the epoch count for
every member of an ensemble at the value the run validated. The refit's own valid metrics are in-sample and are
reported as such, never as a validation score.
"""
from __future__ import annotations
import csv, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from . import config as C, referee as R

OUTCOMES_ONLY_IN_TRAIN = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate',
                          'play_time_ms', 'profile_stay_time', 'comment_stay_time', 'is_profile_enter')

def refit_data_dir(force=False):
    """workspace/data_refit: the side tables and valid.csv as they are; train.csv = train ∪ valid."""
    src, dst = C.WS_DATA, C.WORKSPACE / 'data_refit'
    t, v = src / 'train.csv', src / 'valid.csv'
    sig = f'{t.stat().st_mtime_ns}:{t.stat().st_size}:{v.stat().st_mtime_ns}:{v.stat().st_size}'
    stamp = dst / '.stamp'
    if not force and stamp.exists() and stamp.read_text() == sig:
        return dst
    dst.mkdir(exist_ok=True)
    for p in src.iterdir():                      # side tables and valid.csv unchanged (symlinks)
        if p.name in ('train.csv',) or p.name.startswith('.'):
            continue
        link = dst / p.name
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(p.resolve(), link)
    with open(t, newline='') as fh:
        head = next(csv.reader(fh))
    with open(v, newline='') as fh:
        vhead = next(csv.reader(fh))
    take = {c: vhead.index(c) for c in head if c in vhead}        # the columns valid actually has
    missing = [c for c in head if c not in take]
    assert set(missing) <= set(OUTCOMES_ONLY_IN_TRAIN), f'valid.csv lacks a feature column: {missing}'
    n_train = n_valid = 0
    with open(t, newline='') as fin, open(dst / 'train.csv', 'w', newline='') as out:
        w = csv.writer(out); r = csv.reader(fin); w.writerow(next(r))
        for rec in r:
            w.writerow(rec); n_train += 1
        with open(v, newline='') as fv:
            rv = csv.reader(fv); next(rv)
            for rec in rv:
                w.writerow([rec[take[c]] if c in take else '' for c in head]); n_valid += 1
    stamp.write_text(sig)
    return dst

def _spearman(a, b):
    """Rank correlation without scipy — a sanity bound on how far the refit moved the ordering."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra, rb = np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))

def _read_scores(path):
    with open(path, newline='') as fh:
        r = csv.reader(fh); head = next(r)
        assert head == R.HEADER, f'bad header {head}'
        rows = [(int(rec[0]), float(rec[3])) for rec in r]
    return rows

def refit_submission(run_id, n, out_csv, seed=C.DEFAULT_SEED, epochs=None, baseline_csv=None, log=print):
    """Write the submission for node n of run_id from a model refit on train + valid at its validated epoch count.
    Returns the record to journal: epochs used, row count, the rank correlation with the train-only predictions,
    and the organizers' --check output. Never computes a test metric."""
    run_dir = C.RUNS / run_id
    code = run_dir / 'nodes' / f'{n:03d}.py'
    test_feats = C.PRIVATE / 'test_features.csv'
    if not test_feats.exists():
        raise FileNotFoundError('private/test_features.csv missing — run harness.data_access.build() first')
    if epochs is None:                                  # the epoch count validated by the run itself
        m = json.loads((run_dir / 'outputs' / f'{n:03d}' / 'metrics.json').read_text())
        epochs = int(m.get('best_epoch') or 0)
        hist = len(m.get('history') or [])
        if not epochs:
            raise RuntimeError(f'node_{n:03d} has no best_epoch in metrics.json; pass epochs=')
        log(f'  node_{n:03d}: validated best_epoch {epochs} (of {hist} epochs run), valid primary {m.get("primary"):.4f}')
    data_dir = refit_data_dir()
    with open(data_dir / 'train.csv', newline='') as fh:
        n_rows = sum(1 for _ in fh) - 1
    log(f'  refit data: {data_dir.name}, {n_rows} training rows (train + valid)')
    out_dir = run_dir / 'outputs' / f'{n:03d}_refit'
    res = R.run_script(code, out_dir, seed=seed, score_extra=test_feats, threads=os.cpu_count() or 2,
                       env_extra={'SMOKE_EPOCHS': str(epochs)}, data_dir=data_dir)
    if not res.ok:
        raise RuntimeError(f'refit run failed: {res.error}\n{res.log_tail[-1200:]}')
    ran = json.loads((out_dir / 'metrics.json').read_text())
    n_ep = len(ran.get('history') or [])
    if n_ep != epochs:
        log(f'  WARNING: the refit ran {n_ep} epochs, not {epochs} — the script does not honour SMOKE_EPOCHS as a cap')
    src = out_dir / 'predictions_extra.csv'
    rows = list(csv.reader(open(src, newline='')))
    assert rows[0] == R.HEADER, f'bad header in predictions_extra.csv: {rows[0]}'
    scores = [float(r[3]) for r in rows[1:]]
    assert all(np.isfinite(scores)), 'non-finite scores in the refit submission'
    rho = None
    if baseline_csv and Path(baseline_csv).exists():         # sanity only: the refit must not reorder everything
        a, b = _read_scores(baseline_csv), _read_scores(src)
        assert [x[0] for x in a] == [x[0] for x in b], 'row_id order differs between the two submissions'
        rho = round(_spearman([x[1] for x in a], [x[1] for x in b]), 4)
        log(f'  rank correlation with the train-only submission: {rho}')
    with open(out_csv, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    chk = subprocess.run([sys.executable, str(C.KIT / 'submit.py'), '--check', '--split', 'test', '--data_dir', str(C.RAW),
                          str(Path(out_csv).resolve())], cwd=str(C.KIT), capture_output=True, text=True)
    if chk.returncode != 0:
        raise RuntimeError('official --check failed:\n' + chk.stdout + chk.stderr)
    return {'run_id': run_id, 'node': n, 'refit': True, 'epochs': epochs, 'epochs_ran': n_ep, 'train_rows': n_rows,
            'rows': len(rows) - 1, 'seed': seed, 'spearman_vs_train_only': rho, 'submission': str(out_csv),
            'in_sample_valid_metrics': res.metrics, 'check': chk.stdout.strip(),
            'evidence': ('facts §11.3: the same script trained to 04-14 scores 0.5999 vs 0.6031 on the full window; '
                         'the refit adds the validation week to the training data at the validated epoch count')}
