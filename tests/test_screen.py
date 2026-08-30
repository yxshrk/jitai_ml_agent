"""ADR-0015: the feature screen. A probe runs on the label-stripped data dir, its columns are measured within-user against
the champion's predictions, and the loop drops candidates below SCREEN_MIN_GAIN before any node is built."""
import csv, json
import numpy as np
import pytest
from harness import config as C
from harness import screen as S
from harness import referee as R

pytestmark = pytest.mark.skipif(not (C.WS_DATA / 'valid.csv').exists(), reason='workspace not built')

PROBE_HEAD = '''import argparse, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('--data-dir'); ap.add_argument('--out-dir'); a = ap.parse_args()
tr = pd.read_csv(os.path.join(a.data_dir, 'train.csv')); va = pd.read_csv(os.path.join(a.data_dir, 'valid.csv'))
'''
PROBE_TAIL = '''os.makedirs(a.out_dir, exist_ok=True); out.insert(0, 'row_id', va.row_id.values); out.to_csv(os.path.join(a.out_dir, 'features.csv'), index=False)
'''
# a real within-user signal (the video's train long-view rate), a column constant within users, and a constant column
PROBE_GOOD = PROBE_HEAD + '''mu = tr.long_view.mean(); g = tr.groupby('video_id').long_view.agg(['sum', 'count']); r = (g['sum'] + 20 * mu) / (g['count'] + 20)
gu = tr.groupby('user_id').long_view.agg(['sum', 'count']); ru = (gu['sum'] + 20 * mu) / (gu['count'] + 20)
out = pd.DataFrame({'video_rate': va.video_id.map(r).fillna(mu).values, 'user_rate': va.user_id.map(ru).fillna(mu).values, 'const': np.ones(len(va))})
''' + PROBE_TAIL
PROBE_NULL = PROBE_HEAD + '''out = pd.DataFrame({'const': np.ones(len(va)), 'noise': np.random.RandomState(0).rand(len(va))})
''' + PROBE_TAIL
PROBE_LEAK = PROBE_HEAD + '''out = pd.DataFrame({'label': va.long_view.values})
''' + PROBE_TAIL
PROBE_SHORT = PROBE_HEAD + '''va = va.iloc[:-1]; out = pd.DataFrame({'x': va.tab.values})
''' + PROBE_TAIL

def _prior_champion(path):
    """Synthetic champion: the train long-view rate by (tab, duration decile), written as a valid predictions.csv."""
    import pandas as pd
    tr = pd.read_csv(C.WS_DATA / 'train.csv', usecols=['tab', 'duration_ms', 'long_view']); va = pd.read_csv(C.WS_DATA / 'valid.csv', usecols=['row_id', 'user_id', 'video_id', 'tab', 'duration_ms'])
    edges = tr.duration_ms.quantile(np.linspace(0, 1, 11)[1:-1]).values
    tr['b'] = np.searchsorted(edges, tr.duration_ms.values); va['b'] = np.searchsorted(edges, va.duration_ms.values)
    rate = tr.groupby(['tab', 'b']).long_view.mean()
    va['score'] = [rate.get((t, b), tr.long_view.mean()) for t, b in zip(va.tab, va.b)]
    path.parent.mkdir(parents=True, exist_ok=True)
    va[['row_id', 'user_id', 'video_id', 'score']].to_csv(path, index=False)
    return path

def test_probe_dir_strips_outcomes():
    d = S.probe_data_dir()
    head = next(csv.reader(open(d / 'valid.csv', newline='')))
    assert 'row_id' in head and 'time_ms' in head and not set(head) & set(S.OUTCOME_COLUMNS)
    assert (d / 'train.csv').is_symlink() and (d / 'video_features_basic.csv').exists()
    assert S.probe_data_dir() == d                                       # cached by valid.csv signature

def test_run_probe_measures_a_real_signal(tmp_path):
    champ = _prior_champion(tmp_path / 'champ' / 'predictions.csv')
    (tmp_path / 'good.py').write_text(PROBE_GOOD)
    res = S.run_probe(tmp_path / 'good.py', tmp_path / 'o_good', champ, threads=2)
    assert res.ok, (res.stage, res.error, res.log_tail[-500:])
    c = res.columns
    assert c['video_rate']['varies'] > 0.8 and c['video_rate']['gauc'] > 0.6 and c['video_rate']['additive'] > 0.005
    assert c['user_rate']['varies'] == 0.0 and c['user_rate']['additive'] == 0.0     # constant within users: cannot reorder
    assert c['const']['varies'] == 0.0
    assert res.stack_gain is not None and res.stack_gain > 0.005
    assert res.best_gain >= c['video_rate']['additive'] and S.passes(res)
    assert json.dumps(res.summary())                                                   # journal-serialisable
    (tmp_path / 'null.py').write_text(PROBE_NULL)
    res2 = S.run_probe(tmp_path / 'null.py', tmp_path / 'o_null', champ, threads=2)
    assert res2.ok and res2.best_gain < C.SCREEN_MIN_GAIN and not S.passes(res2)
    assert res2.columns['noise']['varies'] > 0.8   # 17.5 % of users have one valid row: 0.825 is the ceiling and abs(res2.columns['noise']['gauc'] - 0.5) < 0.01

def test_probe_cannot_read_labels_and_bad_outputs_fail(tmp_path):
    champ = _prior_champion(tmp_path / 'champ' / 'predictions.csv')
    (tmp_path / 'leak.py').write_text(PROBE_LEAK)
    res = S.run_probe(tmp_path / 'leak.py', tmp_path / 'o_leak', champ, threads=2)
    assert not res.ok and res.stage == 'probe' and 'long_view' in (res.error or '') + res.log_tail and S.passes(res)  # a failure never blocks
    (tmp_path / 'fw.py').write_text(PROBE_HEAD + "# private/ path\n" + PROBE_TAIL)
    res = S.run_probe(tmp_path / 'fw.py', tmp_path / 'o_fw', champ)
    assert not res.ok and res.stage == 'static'
    (tmp_path / 'short.py').write_text(PROBE_SHORT)
    res = S.run_probe(tmp_path / 'short.py', tmp_path / 'o_short', champ, threads=2)
    assert not res.ok and res.stage == 'features' and 'rows' in res.error

def test_loop_gate_drops_null_feature_candidates(tmp_path, monkeypatch):
    from harness import prompts as P
    from harness.loop import Loop
    from harness.brain import FakeBrain
    class Probing(FakeBrain):
        def probe(self, ctx, selection):
            return {'strong': PROBE_GOOD, 'null': PROBE_NULL, 'broken': PROBE_LEAK, 'declined': None}[selection['hypothesis']]
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    lp = Loop('r', Probing([[]]), k=4)
    assert lp.screen
    lp.state['nodes'] = {'0': {'n': 0, 'metrics': {'primary': 0.6}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True}}
    lp.state['champion'] = 0; lp.state['generation'] = 0
    (tmp_path / 'r' / 'nodes').mkdir(parents=True, exist_ok=True); (tmp_path / 'r' / 'nodes' / '000.py').write_text('import numpy as np\n')
    _prior_champion(lp.j.out_dir(0) / 'predictions.csv')
    card = 'encoding-ordered-item-context-target-statistics'
    sels = [{'type': 'improve', 'card': card, 'target_component': 'encoding', 'hypothesis': 'strong'},
            {'type': 'explore', 'card': 'a-new-idea', 'target_component': 'features', 'hypothesis': 'null', 'wildcard': True, 'new_signal': 'noise from nowhere'},
            {'type': 'improve', 'card': 'history-x', 'target_component': 'history', 'hypothesis': 'broken'},
            {'type': 'improve', 'card': 'loss-bpr-pairwise-within-user', 'target_component': 'loss', 'hypothesis': 'not screened'},
            {'type': 'retest', 'card': 'features-exposure-session', 'target_component': 'features', 'hypothesis': 'retests are not screened'},
            {'type': 'explore', 'card': 'din-attention', 'target_component': 'model', 'hypothesis': 'model wildcards are not screened', 'wildcard': True, 'new_signal': 'attention over history'},
            {'type': 'improve', 'card': 'history-y', 'target_component': 'history', 'hypothesis': 'declined'}]
    kept = lp._screen(sels, 1)
    assert [s['hypothesis'] for s in kept] == ['strong', 'broken', 'not screened', 'retests are not screened', 'model wildcards are not screened', 'declined']
    assert kept[0]['screen']['best_gain'] > 0.005 and 'screen' not in kept[2]
    recs = lp.j.records(); sc = [r for r in recs if r.get('action') == 'screen']
    assert [(r['card'], r['kept']) for r in sc] == [(card, True), ('a-new-idea', False)]
    assert sc[0]['family'] == P._front_fields((C.KB / 'methods' / f'{card}.md').read_text()).get('family') and sc[1]['family'] == 'features'
    assert sc[1]['new_signal'] == 'noise from nowhere' and sc[1]['columns']['noise']['varies'] > 0.8
    assert any(r.get('action') == 'event' and 'history-x' in r['note'] for r in recs)      # the failed probe is journaled, not fatal
    assert any(r.get('action') == 'event' and 'history-y' in r['note'] and 'declined' in r['note'] for r in recs)   # so is a declining Probe
    assert [(e['card'], e['kept']) for e in lp.state['screened']] == [(card, True), ('a-new-idea', False)]
    assert 'DROPPED' in P._screened_state(lp.ctx()) and 'a-new-idea' in P._screened_state(lp.ctx())
    assert 'screen' in lp.j.digest() and 'a-new-idea' in lp.j.digest()
    assert lp.j.compact_lines() == []                                                       # screen records are not node lines
    assert (tmp_path / 'r' / 'screens' / f'g01_{card}' / 'features.csv').exists()

def test_screen_off_without_a_probe_role(tmp_path, monkeypatch):
    from harness.loop import Loop
    from harness.brain import FakeBrain
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    assert not Loop('r', FakeBrain([[]]), k=2).screen
    class Probing(FakeBrain):
        def probe(self, ctx, selection): return PROBE_NULL
    assert not Loop('r2', Probing([[]]), k=2, screen=False).screen
