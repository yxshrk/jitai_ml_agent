"""ADR-0012 amendment (after live_07): strict designation never submits a rejected node; adaptive designation gives an
unaccepted leader MAX_CONFIRM_SEEDS fresh seeds and requires >= MIN_EFFECT at z >= Z_BORDER over the champion."""


def _loop(tmp_path, monkeypatch, designation):
    from harness import config as C
    from harness.loop import Loop
    from harness.brain import FakeBrain
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    lp = Loop('r', FakeBrain([[]]), k=3, designation=designation)
    lp.state['nodes'] = {'0': {'n': 0, 'metrics': {'primary': 0.6015}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True},
                         '9': {'n': 9, 'metrics': {'primary': 0.6041}, 'parent': 0, 'action': 'merge', 'accepted': True},
                         '19': {'n': 19, 'metrics': {'primary': 0.6046}, 'parent': 9, 'action': 'deepen', 'accepted': False},
                         '13': {'n': 13, 'metrics': {'primary': 0.6046}, 'parent': 9, 'action': 'retest', 'accepted': False}}
    lp.state['champion'] = 9; lp.state['generation'] = 5
    lp.state['seed_cache'] = {'9:1': 0.60408, '9:2': 0.60457, '9:3': 0.60445,          # live_07's numbers
                              '19:1': 0.60459, '19:2': 0.60486, '19:3': 0.60512,
                              '13:1': 0.60457, '13:2': 0.60469, '13:3': 0.60458}
    (tmp_path / 'r').mkdir(exist_ok=True)
    return lp


def test_strict_never_submits_a_rejected_node(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch, 'strict')
    ranking = lp.designate_final()
    assert lp.state['designated'] == 9                                        # the accepted champion, not the 0.6049 leader
    assert ranking[0]['n'] == 9 and ranking[0]['accepted']
    assert lp.state['best_unaccepted'] == {'n': 19, 'mean': 0.60486, 'valid_primary': 0.6046}
    assert all(r.get('excluded') for r in ranking if not r['accepted'])
    assert lp.state['designation_events'] and 'node_019 leads on fresh-seed mean' in lp.state['designation_events'][0]


def test_adaptive_designates_only_after_more_seeds_and_z(tmp_path, monkeypatch):
    from harness import config as C
    lp = _loop(tmp_path, monkeypatch, 'adaptive')
    ran = []
    def fake_seeds(m, seeds, extra={19: {4: 0.60500, 5: 0.60480}, 13: {4: 0.60440, 5: 0.60450}}):
        for sd in seeds:
            if f'{m}:{sd}' not in lp.state['seed_cache']:
                lp.state['seed_cache'][f'{m}:{sd}'] = extra[m][sd]; ran.append((m, sd))
    monkeypatch.setattr(lp, '_ensure_seeds', fake_seeds)
    ranking = lp.designate_final()
    assert (19, 4) in ran and (19, 5) in ran                                 # the leader got its two extra seeds …
    r19 = next(r for r in ranking if r['n'] == 19)
    assert len(r19['fresh_seeds']) == C.MAX_CONFIRM_SEEDS and r19['adaptive']['z'] >= C.Z_BORDER and 'excluded' not in r19
    assert lp.state['designated'] == 19                                       # … and passed: designated with the reason journaled
    r13 = next(r for r in ranking if r['n'] == 13)
    assert 'excluded' in r13 and r13['adaptive']['z'] < C.Z_BORDER           # the other unaccepted node failed the test
    assert any('node_019' in e and 'eligible' in e for e in lp.state['designation_events'])


def test_adaptive_falls_back_to_the_champion(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch, 'adaptive')
    monkeypatch.setattr(lp, '_ensure_seeds', lambda m, seeds: [lp.state['seed_cache'].setdefault(f'{m}:{sd}', 0.6040) for sd in seeds])
    lp.designate_final()
    assert lp.state['designated'] == 9                                        # two poor extra seeds: the leader loses its lead
