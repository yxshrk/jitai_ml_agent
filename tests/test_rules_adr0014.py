"""ADR-0014: slot rules in code (closed mechanisms, hard groups, information-adding wildcards, the free slot), the
Critic-directed rebase, deepen variants filed on their base card, and the library contract (LightGBM + torch under
the runner, deterministic)."""
import json
from pathlib import Path


def _loop(tmp_path, monkeypatch, brain=None, k=3):
    from harness import config as C
    from harness.loop import Loop
    from harness.brain import FakeBrain
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    lp = Loop('r', brain or FakeBrain([[]]), k=k)
    lp.state['nodes'] = {'0': {'n': 0, 'metrics': {'primary': 0.6}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True},
                         '2': {'n': 2, 'metrics': {'primary': 0.603}, 'parent': 0, 'action': 'improve', 'accepted': True,
                               'method': 'loss-bpr-pairwise-within-user'},
                         '5': {'n': 5, 'metrics': {'primary': 0.6028}, 'parent': 2, 'action': 'deepen', 'accepted': False,
                               'mechanism': 'Long-duration matched pairs', 'target_group': 'dur>180s'},
                         '6': {'n': 6, 'metrics': {'primary': 0.6027}, 'parent': 2, 'action': 'deepen', 'accepted': False,
                               'mechanism': 'tab-4 specialist', 'target_group': 'dur>180s'}}
    lp.state['champion'] = 2
    (tmp_path / 'r' / 'nodes').mkdir(parents=True, exist_ok=True)
    for n, code in ((0, 'import numpy as np\n# user_id video_id tab duration_ms long_view\n'), (2, 'import numpy as np\n# bpr user_id video_id tab duration_ms long_view\n'),
                    (5, '# node five\n')):
        (tmp_path / 'r' / 'nodes' / f'{n:03d}.py').write_text(code)
    return lp


def test_closed_mechanisms_hard_groups_and_rules(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch)
    assert lp._closed_mechanisms() == {'long-duration-matched-pairs': [5], 'tab-4-specialist': [6]}
    assert lp._hard_groups() == {'dur>180s': [5, 6]}                     # two rejected deepens on one group
    lp.state['nodes']['6']['target_group'] = 'dur > 180 s'                 # same group, written differently
    assert lp._hard_groups() == {'dur>180s': [5, 6]}
    sels = [{'type': 'deepen', 'mechanism': 'long duration MATCHED pairs', 'target_group': 'all', 'hypothesis': 'dose 2.5%'},
            {'type': 'deepen', 'mechanism': 'fresh idea', 'target_group': 'dur > 180s', 'hypothesis': 'hard group'},
            {'type': 'deepen', 'mechanism': 'fresh idea', 'target_group': 'tab=4', 'hypothesis': 'ok deepen'},
            {'type': 'explore', 'wildcard': True, 'new_signal': 'none', 'hypothesis': 'capacity only'},
            {'type': 'explore', 'wildcard': True, 'new_signal': 'same-author run length from earlier exposures', 'hypothesis': 'info'},
            {'type': 'improve', 'card': 'history-same-author-run-features', 'hypothesis': 'free slot'}]
    kept = [s['hypothesis'] for s in lp._apply_rules(sels)]
    assert kept == ['ok deepen', 'info', 'free slot']
    assert 'user_id' in lp.inputs_of(2) and 'author_id' not in lp.inputs_of(2)
    (lp.j.node_path(2)).write_text('import numpy as np\n# user_id onehot_feat3 register_days_range\n')
    assert {'user_id', 'onehot_feat', 'register_days'} <= set(lp.inputs_of(2))


def test_free_slot_survives_wildcard_collision(tmp_path, monkeypatch):
    from harness import prompts as P
    lp = _loop(tmp_path, monkeypatch, k=3)
    monkeypatch.setattr(P, 'untried_cards', lambda: ['features-exposure-session']); monkeypatch.setattr(P, 'proven_cards', lambda: [])
    sels = [{'type': 'improve', 'card': 'features-exposure-session', 'target_component': 'features', 'hypothesis': 'free'},
            {'type': 'deepen', 'card': 'x', 'target_component': 'loss', 'hypothesis': 'deepen'}]
    assert lp._free_slot_ok(sels) and sels[0].get('free_slot')
    wild = {'type': 'explore', 'wildcard': True, 'card': 'w', 'target_component': 'features', 'hypothesis': 'wild', 'new_signal': 'session position'}
    kept = lp._diversify([wild] + sels)                                   # the loop prepends the wildcard
    assert [s['hypothesis'] for s in kept] == ['free', 'deepen']          # the free slot wins the collision, not the wildcard
    kept = lp._diversify([wild, {'type': 'improve', 'card': 'y', 'target_component': 'features', 'hypothesis': 'plain'}])
    assert [s['hypothesis'] for s in kept] == ['wild']                    # an ordinary candidate still loses to it


def test_free_slot_rule(tmp_path, monkeypatch):
    from harness import prompts as P
    lp = _loop(tmp_path, monkeypatch)
    monkeypatch.setattr(P, 'untried_cards', lambda: ['history-same-author-run-features'])
    monkeypatch.setattr(P, 'proven_cards', lambda: ['loss-bpr-pairwise-within-user', 'model-dcn-cross-head'])
    assert lp._proven_not_on_stack() == ['model-dcn-cross-head']       # bpr is in the champion's stack
    assert not lp._free_slot_ok([{'type': 'deepen', 'card': 'loss-bpr-pairwise-within-user — variant'}])
    assert lp._free_slot_ok([{'type': 'deepen', 'card': 'x'}, {'type': 'improve', 'card': 'model-dcn-cross-head'}])
    monkeypatch.setattr(P, 'untried_cards', lambda: []); monkeypatch.setattr(P, 'proven_cards', lambda: ['loss-bpr-pairwise-within-user'])
    assert lp._free_slot_ok([{'type': 'deepen', 'card': 'x'}])          # nothing eligible: the rule is void


def test_critic_rebase(tmp_path, monkeypatch):
    from harness.brain import FakeBrain
    class Rebasing(FakeBrain):
        def __init__(self):
            super().__init__([[]]); self.calls = 0
        def implement(self, ctx, selection, parent_code, extra_parent_code=None):
            return {'code': parent_code + '# edit\n', 'change_summary': 'edit'}
        def critique(self, ctx, code, selection, diff_text=''):
            self.calls += 1
            if self.calls == 1:
                return {'verdict': 'revise', 'reasons': ['built on the champion, not on node_005'], 'instructions': 'x', 'rebase_to': 'node_005'}
            return {'verdict': 'ok', 'reasons': [], 'instructions': '', 'rebase_to': None}
    lp = _loop(tmp_path, monkeypatch, brain=Rebasing())
    sel = {'type': 'deepen', 'parent': 'champion', 'parent_n': 2, 'hypothesis': 'gate node_005', 'target_component': 'ensembling'}
    code, log, err = lp._implement_with_critic(sel, lp.code_of(2), None, [])
    assert err is None and code == '# node five\n# edit\n'                 # second round edited node_005's script
    assert sel['parent_n'] == 5 and sel['parent'] == 5 and sel['rebased_from'] == 2
    assert [c['verdict'] for c in log] == ['revise', 'ok']


def test_distill_files_variants_on_base_card(tmp_path, monkeypatch):
    from harness import config as C
    from harness.distill import distill, _archivable, _card_for
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    methods = tmp_path / 'methods'; methods.mkdir()
    (methods / 'loss-bpr-pairwise-within-user.md').write_text('---\nid: loss-bpr-pairwise-within-user\nstatus: proven — accepted on [official FM]\nevidence: []\n---\n## Claim\nx\n')
    run = tmp_path / 'r'; run.mkdir()
    recs = [{'n': 0, 'action': 'reproduce_baseline', 'metrics': {'primary': 0.6}, 'parent': None},
            {'n': 3, 'action': 'deepen', 'method': 'loss-bpr-pairwise-within-user — same-tab long-duration stream', 'parent': 0,
             'metrics': {'primary': 0.6021, 'gauc': 0.5, 'ndcg5': 0.7}, 'realized_delta': 0.0021, 'accepted': False, 'diff_lines': 12,
             'seed_confirmation': {'delta_mean': -0.0002, 'z': -0.4}},
            {'n': 4, 'action': 'explore', 'method': 'brand new idea', 'wildcard': True, 'parent': 0, 'metrics': {'primary': 0.59}, 'realized_delta': -0.01, 'accepted': False}]
    (run / 'journal.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    assert _card_for('loss-bpr-pairwise-within-user — same-tab long-duration stream', methods)[1] == 'same-tab long-duration stream'
    assert not _archivable(recs[1], methods) and _archivable(recs[2], methods)
    distill('r', methods_dir=methods, log=lambda *a: None)
    text = (methods / 'loss-bpr-pairwise-within-user.md').read_text()
    assert 'r:node_003' in text and '(variant: same-tab long-duration stream)' in text and 'seed-mean Δ -0.0002 (z -0.4)' in text


def test_libraries_run_under_the_contract(tmp_path):
    """LightGBM and torch import and train inside the runner (thread env, seed), deterministically, within the smoke budget."""
    from harness import referee as R
    script = tmp_path / 'libs.py'
    script.write_text('''import argparse, csv, json, os, time
import numpy as np, lightgbm as lgb, torch
from evaluate import evaluate
ap = argparse.ArgumentParser(); ap.add_argument('--data-dir'); ap.add_argument('--out-dir'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--score-extra', default=None); a = ap.parse_args()
n_threads = int(os.environ.get('OMP_NUM_THREADS', '1')); torch.set_num_threads(n_threads); torch.manual_seed(a.seed)
rounds = min(30, int(os.environ.get('SMOKE_EPOCHS', '30')) * 30)
def load(name, cols):
    with open(os.path.join(a.data_dir, name), newline='') as fh:
        r = csv.DictReader(fh); rows = [[float(x[c]) for c in cols] for x in r]
    return np.array(rows)
tr = load('train.csv', ['user_id', 'video_id', 'tab', 'duration_ms', 'hourmin', 'long_view'])[:200000]
va = load('valid.csv', ['row_id', 'user_id', 'video_id', 'tab', 'duration_ms', 'hourmin', 'long_view'])
X, y = tr[:, :5], tr[:, 5]
m = lgb.LGBMClassifier(n_estimators=rounds, num_threads=n_threads, seed=a.seed, deterministic=True, force_row_wise=True, verbose=-1).fit(X, y)
s = m.predict_proba(va[:, 1:6])[:, 1]
w = torch.nn.Linear(1, 1); s = (w(torch.tensor(s, dtype=torch.float32)[:, None]).detach().numpy()[:, 0]) + s   # torch is exercised
metrics = evaluate(va[:, 1].astype(int).tolist(), va[:, 6].astype(int).tolist(), s.tolist())
os.makedirs(a.out_dir, exist_ok=True)
with open(os.path.join(a.out_dir, 'predictions.csv'), 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['row_id', 'user_id', 'video_id', 'score'])
    for i in range(len(va)): wr.writerow([int(va[i, 0]), int(va[i, 1]), int(va[i, 2]), float(s[i])])
json.dump({'gauc': metrics['GAUC'], 'ndcg5': metrics['nDCG@5'], 'primary': metrics['primary'], 'best_epoch': rounds, 'seed': a.seed, 'duration_s': 0,
           'history': [{'epoch': 1, 'train_loss': 0.0, 'val_gauc': metrics['GAUC'], 'val_ndcg5': metrics['nDCG@5'], 'val_primary': metrics['primary']}]},
          open(os.path.join(a.out_dir, 'metrics.json'), 'w'))
print('epoch 1 primary %.4f' % metrics['primary'])
''')
    r1 = R.run_script(script, tmp_path / 'o1', seed=0, smoke=True, threads=2)
    assert r1.ok, (r1.error, r1.log_tail[-800:])
    r2 = R.run_script(script, tmp_path / 'o2', seed=0, smoke=True, threads=2)
    assert r2.ok and r1.pred_hash == r2.pred_hash                        # deterministic given the seed
    assert r1.duration_s < 120 and 0.5 < r1.metrics['primary'] < 0.7
