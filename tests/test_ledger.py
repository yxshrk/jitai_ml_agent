import json, pathlib
from kb.methods import ledger as L

CARD = """---
id: {cid}
family: {fam}
target_component: features
source: {source}
applies_when:
  - always
expected_delta: [{lo}, {hi}]
expected_delta_basis: {basis}
cost: none
composes_with: []
conflicts_with: []
status: {status}
evidence: []
---
## Claim
x
## Mechanism
x
## How to implement
x
## Risks
x
## Measured
{measured}
"""

def _kb(tmp_path, bounds, cards):
    d = tmp_path / 'methods'; d.mkdir(parents=True)
    (d / 'family_bounds.json').write_text(json.dumps(bounds))
    for c in cards:
        (d / f"{c['cid']}.md").write_text(CARD.format(**c))
    return d

def test_ledger_parses_measured_lines_bounds_and_statuses(tmp_path):
    bounds = {'signal_families': {'item-side': {'bound': 0.0003, 'source': 'facts §11 row'}, 'session': {'bound': 0.001, 'source': 'facts §11.1'}},
              'cards': {'a-card': 'item-side', 'c-card': 'session'}}
    cards = [dict(cid='a-card', fam='features', source='kb/data/facts.md', lo=0.001, hi=0.004, basis='facts', status='dead_under [x]',
                  measured='- live_04:node_005 on [official FM]: primary 0.6010, single-seed Δ -0.0005 — rejected; 10 changed lines\n'
                           '- live_06:node_011 on [official FM + BPR]: primary 0.6030, single-seed Δ +0.0004, seed-mean Δ +0.0002 (z 0.9) — rejected; 12 changed lines\n'
                           '- live_07:screen-g3 on [official FM + BPR]: SCREENED DROPPED best_gain +0.0001 (col); note'),
             dict(cid='b-card', fam='model', source='arXiv 1234.5678 (Zhou et al.)', lo=0.0, hi=0.004, basis='the paper reports +2 % AUC', status='untried', measured=''),
             dict(cid='c-card', fam='features', source='kb/data/facts.md §10.5', lo=0.0005, hi=0.003, basis='measured effect', status='untried', measured='')]
    led = L.build(_kb(tmp_path, bounds, cards))
    a, b, c = led['cards']['a-card'], led['cards']['b-card'], led['cards']['c-card']
    assert a['bound'] == 0.0003 and a['measured_max'] == 0.0002 and a['basis_class'] == 'measured' and not a['accepted']
    assert b['bound'] is None and b['basis_class'] == 'paper' and b['measured_max'] is None
    assert c['bound'] == 0.001 and c['basis_class'] == 'oracle'
    feats = led['families']['features']
    assert feats['bound'] == 0.001 and feats['status'] == 'open'          # the session card's bound clears the threshold
    assert feats['best_measured'] == 0.0002 and len(feats['measured']) == 2 and len(feats['screen_gains']) == 1
    assert feats['screen_gains'][0] == {'run': 'live_07', 'generation': 3, 'best_gain': 0.0001, 'kept': False, 'stack': 'official FM + BPR', 'card': 'a-card'}
    assert led['families']['model']['status'] == 'open' and led['families']['model']['bound'] is None
    # a family whose every card is bounded at or below the acceptance threshold is closed
    bounds['cards'].pop('c-card'); cards[2]['fam'] = 'history'; bounds['cards']['a-card'] = 'item-side'
    led = L.build(_kb(tmp_path / 'two', bounds, cards[:1]))
    assert led['families']['features']['status'] == 'bounded'
    out = L.write(tmp_path / 'two' / 'methods'); assert json.loads(out.read_text())['families']['features']['bound'] == 0.0003

def test_ledger_on_the_real_knowledge_base():
    led = L.build()
    bounds = json.loads((L.METHODS / 'family_bounds.json').read_text())
    assert set(bounds['cards'].values()) <= set(bounds['signal_families'])       # every mapping names a known signal family
    assert len(led['cards']) >= 40 and 'ranking-loss' in led['families']
    bpr = led['cards']['loss-bpr-pairwise-within-user']
    assert bpr['accepted'] and bpr['measured_max'] >= 0.001 and bpr['basis_class'] == 'measured'
    for fam, f in led['families'].items():                                          # a closed family clears nothing
        if f['status'] == 'bounded':
            assert f['bound'] <= L.CLOSED_BOUND and not any(led['cards'][c]['accepted'] for c in f['cards'])
