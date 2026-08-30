"""Produce the final submission for one node: the only place the harness touches test features (ADR-0005).
Never computes or prints a test metric."""
from __future__ import annotations
import csv, subprocess, sys
from pathlib import Path
from . import config as C, referee as R

def make_submission(run_id, n, out_csv, seed=C.DEFAULT_SEED):
    run_dir = C.RUNS / run_id
    code = run_dir / 'nodes' / f'{n:03d}.py'
    out_dir = run_dir / 'outputs' / f'{n:03d}_final'
    test_feats = C.PRIVATE / 'test_features.csv'
    if not test_feats.exists():
        raise FileNotFoundError('private/test_features.csv missing — run harness.data_access.build() first')
    res = R.run_script(code, out_dir, seed=seed, score_extra=test_feats)
    if not res.ok:
        raise RuntimeError(f'final run failed: {res.error}\n{res.log_tail}')
    src = out_dir / 'predictions_extra.csv'
    rows = list(csv.reader(open(src, newline='')))
    assert rows[0] == R.HEADER, f'bad header in predictions_extra.csv: {rows[0]}'
    with open(out_csv, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    # validate format + alignment with the organizers' own checker (it reads the raw data dir; harness-side only)
    chk = subprocess.run([sys.executable, str(C.KIT / 'submit.py'), '--check', '--split', 'test', '--data_dir', str(C.RAW), str(Path(out_csv).resolve())],
                         cwd=str(C.KIT), capture_output=True, text=True)
    if chk.returncode != 0:
        raise RuntimeError('official --check failed:\n' + chk.stdout + chk.stderr)
    return {'node': n, 'valid_metrics': res.metrics, 'rows': len(rows) - 1, 'submission': str(out_csv), 'check': chk.stdout.strip()}
