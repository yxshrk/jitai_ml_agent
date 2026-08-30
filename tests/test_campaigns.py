"""ADR-0016: family campaigns — one card family per generation from generation 2, chosen in code; mechanism-level
diversity inside the family; a family closes after CAMPAIGN_FLAT_GENERATIONS flat generations; ensembling last."""
import json

CARD = "---\nid: {id}\nfamily: {family}\ntarget_component: {tc}\nsource: x\napplies_when:\n  - a\nexpected_delta: [0.0, {hi}]\nexpected_delta_basis: b\ncost: c\ncomposes_with: []\nconflicts_with: []\nstatus: {status}\nevidence: []\n---\n## Claim\nx\n"


def _kb(tmp_path, monkeypatch):
    from harness import config as C
    kb = tmp_path / 'kb'; (kb / 'methods').mkdir(parents=True)
    cards = [('history-a', 'history', 'history', 0.004, 'untried'), ('history-b', 'history', 'history', 0.002, 'untried'),
             ('model-x', 'model', 'model', 0.008, 'dead_under [official FM + loss-bpr ×1 (best -0.0022)]'),
             ('ensembling-z', 'ensembling', 'ensembling', 0.010, 'untried'),
             ('features-s', 'features', 'features', 0.003, 'proven — accepted on [official FM]'),
             ('loss-bpr', 'ranking-loss', 'loss', 0.010, 'proven — accepted on [official FM]')]
    for cid, fam, tc, hi, st in cards:
        (kb / 'methods' / f'{cid}.md').write_text(CARD.format(id=cid, family=fam, tc=tc, hi=hi, status=st))
    monkeypatch.setattr(C, 'KB', kb); monkeypatch.setattr(C, 'RUNS', tmp_path / 'runs')
    return kb


def _loop(tmp_path, monkeypatch, **kw):
    from harness.loop import Loop
    from harness.brain import FakeBrain
    lp = Loop('r', FakeBrain([[]]), k=3, **kw)
    lp.state['nodes'] = {'0': {'n': 0, 'metrics': {'primary': 0.6}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True},
                         '3': {'n': 3, 'metrics': {'primary': 0.603}, 'parent': 0, 'action': 'improve', 'accepted': True, 'method': 'loss-bpr'}}
    lp.state['champion'] = 3
    (tmp_path / 'runs' / 'r' / 'nodes').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'runs' / 'r' / 'nodes' / '003.py').write_text('# bpr\n')
    return lp


def test_family_choice_order(tmp_path, monkeypatch):
    _kb(tmp_path, monkeypatch); lp = _loop(tmp_path, monkeypatch)
    assert lp._campaign_family(1) is None                                   # generation 1 stays the breadth generation
    assert lp._campaign_family(2) == 'history'                              # highest expected_delta among measurable families …
    fams = lp.state['families']
    assert fams['model']['status'] == 'exhausted'                            # … model-x is dead on this very stack
    assert fams['ranking-loss']['status'] == 'exhausted'                     # loss-bpr is already in the champion stack
    assert 'ensembling' in fams and fams['ensembling']['status'] == 'open'   # last, not skipped
    lp.state['campaign'] = None; lp.state['families']['history']['status'] = 'closed'
    assert lp._campaign_family(3) == 'features'                             # proven card not on the stack still counts
    lp.state['campaign'] = None; lp.state['families']['features']['status'] = 'closed'
    assert lp._campaign_family(4) == 'ensembling'                           # composition comes last
    lp.state['campaign'] = None; lp.state['families']['ensembling']['status'] = 'closed'
    assert lp._campaign_family(5) is None                                   # every family closed: the old behaviour


def test_screen_gain_orders_families_first(tmp_path, monkeypatch):
    _kb(tmp_path, monkeypatch); lp = _loop(tmp_path, monkeypatch)
    lp.state['screened'] = [{'card': 'features-s', 'family': 'features', 'best_gain': 0.0006}]
    assert lp._campaign_family(2) == 'features'                             # a measured screen gain beats a card's promise


def test_no_campaigns_flag_and_state_in_ctx(tmp_path, monkeypatch):
    _kb(tmp_path, monkeypatch)
    lp = _loop(tmp_path, monkeypatch, campaigns=False)
    assert lp._campaign_family(2) is None
    lp = _loop(tmp_path, monkeypatch); lp.state['campaign'] = lp._campaign_family(2)
    ctx = lp.ctx()
    assert ctx['campaign'] == 'history' and ctx['campaign_cards'] == ['history-a [untried]', 'history-b [untried]']
    from harness import prompts as P
    s = P._rules_state(ctx)
    assert 'CAMPAIGN this generation: history' in s and 'ensembling (open' in s


def test_diversify_by_mechanism_inside_a_campaign(tmp_path, monkeypatch):
    _kb(tmp_path, monkeypatch); lp = _loop(tmp_path, monkeypatch); lp.state['campaign'] = 'history'; lp.k = 5
    sels = [{'type': 'explore', 'wildcard': True, 'target_component': 'history', 'hypothesis': 'wild', 'new_signal': 'last positive overlap'},
            {'type': 'improve', 'card': 'history-a', 'target_component': 'history', 'mechanism': 'author-run', 'hypothesis': 'a'},
            {'type': 'deepen', 'card': 'history-b', 'target_component': 'history', 'mechanism': 'Author Run', 'hypothesis': 'same mechanism again'},
            {'type': 'deepen', 'card': 'history-b', 'target_component': 'history', 'mechanism': 'tag-recurrence', 'hypothesis': 'b'},
            {'type': 'merge', 'merge_parents': [3, 4], 'target_component': 'ensembling', 'hypothesis': 'm'}]
    kept = [s['hypothesis'] for s in lp._diversify(sels)]
    assert kept == ['wild', 'a', 'b', 'm']                                 # same component allowed; same mechanism dropped
    lp.state['campaign'] = None
    kept = [s['hypothesis'] for s in lp._diversify(sels)]
    assert kept == ['wild', 'm']                                            # breadth generation: one per component


def test_rules_drop_outside_family_and_free_slot_stays_inside(tmp_path, monkeypatch):
    _kb(tmp_path, monkeypatch); lp = _loop(tmp_path, monkeypatch); lp.state['campaign'] = 'history'
    sels = [{'type': 'improve', 'card': 'features-s', 'target_component': 'features', 'hypothesis': 'outside'},
            {'type': 'retest', 'card': 'features-s', 'target_component': 'features', 'hypothesis': 'consolidator retest'},
            {'type': 'improve', 'card': 'history-a', 'target_component': 'history', 'mechanism': 'x', 'hypothesis': 'inside'},
            {'type': 'explore', 'wildcard': True, 'target_component': 'model', 'new_signal': 'session position', 'hypothesis': 'wild'}]
    assert [s['hypothesis'] for s in lp._apply_rules(sels)] == ['consolidator retest', 'inside', 'wild']
    assert not lp._free_slot_ok([{'type': 'improve', 'card': 'features-s'}])       # eligible outside the family does not count
    assert lp._free_slot_ok([{'type': 'improve', 'card': 'history-a'}])


def test_family_closes_after_flat_generations(tmp_path, monkeypatch):
    from harness import config as C
    _kb(tmp_path, monkeypatch); lp = _loop(tmp_path, monkeypatch)
    lp.state['campaign'] = lp._campaign_family(2)
    def res(n, method, acc, dm):
        return {'n': n, 'metrics': {'primary': 0.603}, 'method': method, 'action': 'improve', 'accepted': acc,
                'seed_confirmation': {'delta_mean': dm}, 'realized_delta': dm}
    lp._campaign_update(2, [res(4, 'history-a', False, 0.0002), res(5, 'ensembling-z', True, 0.002)])   # the accepted node is another family's
    f = lp.state['families']['history']
    assert f['flat_streak'] == 1 and f['best_gain'] == 0.0002 and f['nodes'] == [4] and f['status'] == 'open'
    lp._campaign_update(3, [res(6, 'history-b — variant', False, -0.0001)])
    assert f['status'] == 'closed' and f['flat_streak'] == C.CAMPAIGN_FLAT_GENERATIONS and 'closed at generation 3' in f['evidence']
    lp.state['campaign'] = lp._campaign_family(4)
    assert lp.state['campaign'] == 'features'                               # the next family opens
    lp._campaign_update(4, [res(7, 'features-s', True, 0.0009)])
    assert lp.state['families']['features']['flat_streak'] == 0 and lp.state['families']['features']['best_gain'] == 0.0009
