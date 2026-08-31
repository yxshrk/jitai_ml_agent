"""ADR-0021: a frontier of progressing nodes (not one champion) and a persistent proposal queue over it."""
import json


def _loop(tmp_path, monkeypatch, **kw):
    from harness import config as C
    from harness.loop import Loop
    from harness.brain import FakeBrain
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    lp = Loop('r', FakeBrain([[]]), k=3, **kw)
    lp.state['nodes'] = {
        '0': {'n': 0, 'metrics': {'primary': 0.6015}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True},
        '1': {'n': 1, 'metrics': {'primary': 0.6031}, 'parent': 0, 'action': 'improve', 'accepted': True, 'method': 'loss-bpr-pairwise-within-user'},
        '14': {'n': 14, 'metrics': {'primary': 0.6043}, 'parent': 1, 'action': 'improve', 'accepted': True, 'method': 'ensembling-multiseed-heterogeneous-rank-blend'},
        '17': {'n': 17, 'metrics': {'primary': 0.6046}, 'parent': 14, 'action': 'deepen', 'accepted': False, 'method': 'features-exposure-session'},
        '20': {'n': 20, 'metrics': {'primary': 0.6021}, 'parent': 1, 'action': 'improve', 'accepted': False, 'method': 'model-dcn-cross-head'}}
    lp.state['champion'] = 14; lp.state['generation'] = 4
    lp.state['seed_cache'] = {'0:1': 0.60176, '0:2': 0.60109, '0:3': 0.60150,
                              '1:1': 0.60312, '1:2': 0.60272, '1:3': 0.60354,
                              '14:1': 0.60467, '14:2': 0.60436, '14:3': 0.60470,     # champion mean 0.60458
                              '17:1': 0.60487, '17:2': 0.60450, '17:3': 0.60471,     # unconfirmed but HIGHER: 0.60469
                              '20:1': 0.60210, '20:2': 0.60190, '20:3': 0.60220}     # far below: not on the frontier
    (tmp_path / 'r' / 'nodes').mkdir(parents=True, exist_ok=True)
    return lp


def test_frontier_holds_the_unconfirmed_near_miss_and_drops_the_laggard(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch)
    fr = lp.frontier_update(4)
    ns = {e['n'] for e in fr.values()}
    assert 14 in ns and 17 in ns                       # the champion and the higher-mean unconfirmed node
    assert 20 not in ns                                # 0.0025 below the champion: not worth building on
    view = lp.frontier_view()
    assert view[0]['n'] == 17 and view[0]['accepted'] is False and view[0]['champion'] is False
    assert [e for e in view if e['champion']][0]['n'] == 14


def test_queue_persists_ideas_and_pops_by_score(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    sels = [{'type': 'deepen', 'card': 'ensembling-seed-average', 'mechanism': 'more-members', 'target_component': 'ensembling',
             'hypothesis': 'a', 'expected_delta': 0.001, 'parent': 17},
            {'type': 'improve', 'card': 'regularization-embedding-dropout-l2', 'mechanism': 'l2', 'target_component': 'regularization',
             'hypothesis': 'b', 'expected_delta': 0.001, 'parent': 'champion'},
            {'type': 'deepen', 'card': 'ensembling-seed-average', 'mechanism': 'MORE members', 'target_component': 'ensembling',
             'hypothesis': 'duplicate of the first', 'expected_delta': 0.001, 'parent': 17}]
    assert lp.queue_add(sels, 4) == 2                  # the duplicate mechanism on the same parent is not queued twice
    assert len(lp.state['queue']) == 2
    popped = lp.queue_pop(1, 4)
    assert len(popped) == 1 and popped[0]['parent'] == 17          # the higher-mean parent's proposal outranks the champion's
    assert len(lp.state['queue']) == 1 and lp.state['queue'][0]['hypothesis'] == 'b'   # the rest WAITS, it is not thrown away
    later = lp.queue_pop(3, 5)
    assert [x['hypothesis'] for x in later] == ['b'] and lp.state['queue'] == []       # and runs when a slot frees


def test_queue_drops_stale_retired_and_closed_but_keeps_out_of_campaign(tmp_path, monkeypatch):
    from harness import config as C
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    lp.state['queue'] = [
        {'type': 'deepen', 'card': 'a', 'mechanism': 'old', 'parent': 17, 'added': 1, 'score': 0.9},           # stale
        {'type': 'deepen', 'card': 'b', 'mechanism': 'gone', 'parent': 20, 'added': 4, 'score': 0.8},          # parent off-frontier
        {'type': 'deepen', 'card': 'c', 'mechanism': 'dead-mech', 'parent': 14, 'added': 4, 'score': 0.7},     # closed mechanism
        {'type': 'improve', 'card': 'features-exposure-session', 'mechanism': 'feat', 'parent': 14, 'added': 4, 'score': 0.6}]
    lp.state['nodes']['21'] = {'n': 21, 'metrics': {'primary': 0.60}, 'parent': 14, 'action': 'deepen', 'accepted': False,
                               'mechanism': 'dead-mech', 'target_group': 'all'}
    lp.state['campaign'] = 'ensembling'
    popped = lp.queue_pop(4, 4)
    assert popped == []                                                    # nothing eligible …
    assert [x['card'] for x in lp.state['queue']] == ['features-exposure-session']   # … only the out-of-campaign one waits
    lp.state['campaign'] = None
    assert [x['card'] for x in lp.queue_pop(4, 4)] == ['features-exposure-session']  # its own campaign comes round


def test_barren_frontier_node_retires_and_its_queue_goes_with_it(tmp_path, monkeypatch):
    from harness import config as C
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    lp.state['queue'] = [{'type': 'deepen', 'card': 'x', 'mechanism': 'm', 'parent': 17, 'added': 4, 'score': 0.5}]
    for g in (5, 6):
        lp._frontier_book(g, [{'n': 30 + g, 'parent': 17, 'metrics': {'primary': 0.60}, 'accepted': False}])
        lp.frontier_update(g)
    assert str(17) not in lp.state['frontier']                             # two generations without an accepted child
    assert lp.state['queue'] == []                                         # its pending proposals left with it
    assert str(lp.state['champion']) in lp.state['frontier']               # the champion never retires


def test_frontier_off_restores_champion_only(tmp_path, monkeypatch):
    lp = _loop(tmp_path, monkeypatch, frontier=False)
    (tmp_path / 'r' / 'nodes' / '014.py').write_text('# champion\n')      # ctx() reads the champion's script
    assert lp.frontier_update(4) == {} and lp.frontier_view() == [] and lp.ctx()['frontier'] == []
