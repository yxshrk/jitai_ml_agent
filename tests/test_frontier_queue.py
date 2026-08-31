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


def test_slate_returns_the_unrun_to_the_queue_but_not_the_screened_out(tmp_path, monkeypatch):
    """BLOCKING 1 (review): the k cap and diversity collisions must not swallow proposals; a measured screen must."""
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4); lp.k = 2
    popped = [{'type': 'improve', 'card': 'a', 'target_component': 'loss', 'mechanism': 'm1', 'parent': 14, 'popped': 4, 'hypothesis': 'a'},
              {'type': 'improve', 'card': 'b', 'target_component': 'features', 'mechanism': 'm2', 'parent': 17, 'popped': 4, 'hypothesis': 'b'},
              {'type': 'improve', 'card': 'c', 'target_component': 'loss', 'mechanism': 'm3', 'parent': 14, 'popped': 4, 'hypothesis': 'c'},
              {'type': 'improve', 'card': 'd', 'target_component': 'model', 'mechanism': 'm4', 'parent': 14, 'popped': 4, 'hypothesis': 'd'},
              {'type': 'improve', 'card': 'e', 'target_component': 'encoding', 'mechanism': 'm5', 'parent': 17, 'popped': 4, 'hypothesis': 'e'}]
    monkeypatch.setattr(lp, '_screen', lambda sels, g: [s for s in sels if s['card'] != 'd'])     # 'd' measured and dropped
    running = lp._slate(popped, 4)
    assert [s['card'] for s in running] == ['a', 'b']                       # k = 2 …
    waiting = [s['card'] for s in lp.state['queue']]
    assert 'c' in waiting                                                    # … 'c' lost the component collision: back in the queue
    assert 'e' in waiting                                                    # … 'e' survived diversity AND the screen, cut only by the k cap
    assert 'd' not in waiting                                                # … 'd' was measured by the screen: answered, not requeued
    assert all('popped' not in s for s in lp.state['queue'])


def test_childless_frontier_node_ages_out(tmp_path, monkeypatch):
    """BLOCKING 2 (review): a node nobody proposes for must age too, or the frontier fills with abandoned nodes."""
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    assert str(17) in lp.state['frontier']
    for g in (5, 6):
        lp._frontier_book(g, [{'n': 40 + g, 'parent': 14, 'metrics': {'primary': 0.60}, 'accepted': False}])   # nothing on node_017
        lp.frontier_update(g)
    assert str(17) not in lp.state['frontier'] and str(lp.state['champion']) in lp.state['frontier']


def test_near_miss_needs_its_own_fresh_seeds_and_the_margin_needs_this_run(tmp_path, monkeypatch):
    """BLOCKING 3 + ruling (a): no seed-0 admission, and no cross-run SD in an in-run decision."""
    from harness import config as C
    lp = _loop(tmp_path, monkeypatch)
    lp.state['nodes']['25'] = {'n': 25, 'metrics': {'primary': 0.6047}, 'parent': 14, 'action': 'deepen', 'accepted': False}
    fr = lp.frontier_update(4)
    assert '25' not in fr                                                    # a lucky seed-0 primary is not a frontier claim
    assert lp._frontier_margin() > 0                                         # this run has pooled seeds …
    lp.state['seed_cache'] = {}                                              # … and without them the margin is 0, not C.SEED_SD
    assert lp._frontier_margin() == 0.0


def test_queue_scores_are_refreshed_at_pop_time(tmp_path, monkeypatch):
    """Review (b): a proposal that waits must be re-scored — its parent gains seeds and moves."""
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    lp.queue_add([{'type': 'deepen', 'card': 'ensembling-seed-average', 'mechanism': 'more', 'target_component': 'ensembling',
                   'hypothesis': 'h', 'expected_delta': 0.001, 'parent': 17}], 4)
    stale = lp.state['queue'][0]['score']
    lp.state['queue'][0]['score'] = 99.0                                     # a stale score must not survive the pop
    lp.state['seed_cache'].update({'17:1': 0.60600, '17:2': 0.60600, '17:3': 0.60600})   # the parent moved up
    popped = lp.queue_pop(1, 5)
    assert popped[0]['score'] != 99.0 and popped[0]['score'] > stale


def test_duplicate_key_falls_back_to_the_hypothesis(tmp_path, monkeypatch):
    """Review (c): two card-less, mechanism-less wildcards on one parent are different proposals."""
    lp = _loop(tmp_path, monkeypatch); lp.frontier_update(4)
    a = {'type': 'explore', 'wildcard': True, 'parent': 14, 'hypothesis': 'attention over the user history', 'expected_delta': 0.001}
    b = {'type': 'explore', 'wildcard': True, 'parent': 14, 'hypothesis': 'a duration-gated blend', 'expected_delta': 0.001}
    assert lp.queue_add([a, b], 4) == 2 and lp.queue_add([dict(a)], 4) == 0
