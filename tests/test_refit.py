"""ADR-0019: the designation-time refit — train + valid as the training set, epochs fixed at the validated count."""
import pytest

import csv, json, types
from pathlib import Path

TRAIN_HEAD = ['user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'is_click', 'long_view', 'play_time_ms', 'duration_ms', 'is_rand', 'tab']
VALID_HEAD = ['row_id', 'user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'tab', 'duration_ms', 'is_rand', 'long_view']


def _workspace(tmp_path, monkeypatch):
    from harness import config as C
    ws = tmp_path / 'workspace'; data = ws / 'data'; data.mkdir(parents=True)
    with open(data / 'train.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(TRAIN_HEAD)
        for i in range(4):
            w.writerow([i, 100 + i, '20220409', '1200', 1000 + i, 1, i % 2, 5000, 20000, 0, 1])
    with open(data / 'valid.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(VALID_HEAD)
        for i in range(3):
            w.writerow([i, 200 + i, 300 + i, '20220422', '0900', 2000 + i, 2, 30000, 0, i % 2])
    (data / 'video_features_basic.csv').write_text('video_id,author_id\n100,7\n')
    monkeypatch.setattr(C, 'WORKSPACE', ws); monkeypatch.setattr(C, 'WS_DATA', data)
    return ws, data


def test_refit_data_dir_appends_valid_rows_and_leaves_unknown_outcomes_empty(tmp_path, monkeypatch):
    from harness.refit import refit_data_dir
    ws, data = _workspace(tmp_path, monkeypatch)
    d = refit_data_dir()
    rows = list(csv.reader(open(d / 'train.csv', newline='')))
    assert rows[0] == TRAIN_HEAD and len(rows) == 1 + 4 + 3                       # header + train + valid
    tail = rows[-3:]
    for i, rec in enumerate(tail):
        r = dict(zip(TRAIN_HEAD, rec))
        assert r['user_id'] == str(200 + i) and r['long_view'] == str(i % 2)      # the label came across …
        assert r['tab'] == '2' and r['duration_ms'] == '30000' and r['date'] == '20220422'
        assert r['is_click'] == '' and r['play_time_ms'] == ''                    # … outcomes valid lacks stay EMPTY
    assert (d / 'valid.csv').is_symlink() and (d / 'video_features_basic.csv').is_symlink()
    assert len(list(csv.reader(open(d / 'valid.csv', newline='')))) == 4          # valid untouched
    before = (d / 'train.csv').stat().st_mtime_ns
    refit_data_dir()                                                              # cached on the source stamp
    assert (d / 'train.csv').stat().st_mtime_ns == before


def test_spearman_bounds():
    from harness.refit import _spearman
    assert _spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert _spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert 0.5 < _spearman([1, 2, 3, 4, 5], [1, 3, 2, 5, 4]) < 1.0


def test_refit_submission_fixes_epochs_and_checks(tmp_path, monkeypatch):
    from harness import config as C, referee as R
    import harness.refit as RF
    ws, data = _workspace(tmp_path, monkeypatch)
    runs = tmp_path / 'runs'; node_dir = runs / 'r' / 'nodes'; node_dir.mkdir(parents=True)
    (node_dir / '006.py').write_text('# script\n')
    out006 = runs / 'r' / 'outputs' / '006'; out006.mkdir(parents=True)
    (out006 / 'metrics.json').write_text(json.dumps({'primary': 0.6043, 'best_epoch': 8, 'history': [{'epoch': i} for i in range(12)]}))
    priv = tmp_path / 'private'; priv.mkdir()
    with open(priv / 'test_features.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['row_id', 'user_id', 'video_id']); [w.writerow([i, 500 + i, 600 + i]) for i in range(3)]
    monkeypatch.setattr(C, 'RUNS', runs); monkeypatch.setattr(C, 'PRIVATE', priv)
    seen = {}
    def fake_run(code_path, out_dir, **kw):
        seen.update(kw); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'metrics.json').write_text(json.dumps({'primary': 0.62, 'history': [{'epoch': i} for i in range(8)]}))
        with open(out_dir / 'predictions_extra.csv', 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(R.HEADER); [w.writerow([i, 500 + i, 600 + i, 0.5 - 0.1 * i]) for i in range(3)]
        return types.SimpleNamespace(ok=True, error=None, log_tail='', metrics={'primary': 0.62})
    monkeypatch.setattr(RF.R, 'run_script', fake_run)
    monkeypatch.setattr(RF.subprocess, 'run', lambda *a, **k: types.SimpleNamespace(returncode=0, stdout='OK 3 rows', stderr=''))
    base = tmp_path / 'train_only.csv'
    with open(base, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(R.HEADER); [w.writerow([i, 500 + i, 600 + i, 0.4 - 0.1 * i]) for i in range(3)]
    rec = RF.refit_submission('r', 6, str(tmp_path / 'sub.csv'), baseline_csv=str(base), log=lambda *a: None)
    assert seen['env_extra'] == {'SMOKE_EPOCHS': '8'}                             # the VALIDATED epoch count, not early stopping
    assert Path(seen['data_dir']).name == 'data_refit' and seen['score_extra'] == priv / 'test_features.csv'
    assert rec['epochs'] == 8 and rec['epochs_ran'] == 8 and rec['rows'] == 3 and rec['train_rows'] == 7
    assert rec['spearman_vs_train_only'] == 1.0 and rec['refit'] is True and 'facts §11.3' in rec['evidence']
    assert 'in_sample_valid_metrics' in rec and 'check' in rec                    # in-sample, never reported as validation
    assert list(csv.reader(open(tmp_path / 'sub.csv', newline='')))[0] == R.HEADER
