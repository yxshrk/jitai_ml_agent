import csv, math
import pytest
from harness import referee as R, config as C

def test_static_check_flags_forbidden_paths():
    assert R.static_check("open('KuaiRand-Pure/data/x.csv')") == ['KuaiRand-Pure']
    assert 'private/' in R.static_check("p = 'private/test_features.csv'")
    assert R.static_check("import numpy as np\nprint('hello')") == []

def test_pair_breakdown():
    uid, _, y = R.valid_index()
    perfect = R.pair_breakdown([float(v) for v in y]); flat = R.pair_breakdown([0.0] * len(y))
    assert perfect['total_err'] == 0.0 and flat['total_err'] == 0.5                       # labels as scores / all ties
    assert abs(perfect['same_tab']['share'] + perfect['diff_tab']['share'] - 1.0) < 1e-6   # the tab cut partitions the mass
    assert abs(perfect['same_date']['share'] + perfect['diff_date']['share'] - 1.0) < 1e-6
    assert all(flat[t]['err'] == 0.5 for t in R.PAIR_TYPES if flat[t]['err'] is not None)
    assert flat['tab1_x_tab1']['share'] > 0.5                                               # facts §11: the feed pairs dominate
    m = R.score([0.0] * len(y), breakdown=True); assert 'by_pair' in m and m['by_pair']['total_err'] == 0.5

def test_accept_and_convergence():
    assert R.accept(0.6015, 0.6040) == (True, pytest.approx(0.0025))
    assert R.accept(0.6015, 0.6030)[0] is False
    c = R.Convergence(0, 0.60144)                                # ADR-0012: cumulative rise of the champion's fresh-seed mean
    assert c.update(0.60214) is False and c.streak == 1          # +0.0007 since the reference: not yet
    assert c.update(0.60284) is True and c.streak == 0 and c.ref == 0.60284   # staircase adds up to +0.0014: reset
    assert c.update(0.60340) is False and c.streak == 1          # +0.00056: one false acceptance cannot buy time
    assert all(c.update(0.60340) is False for _ in range(2)) and c.converged
    c = R.Convergence(0, 0.60143)                                # the loop seeds ref with the BASELINE (start()), so generation 1's gain counts
    assert c.update(0.60246) is True and c.streak == 0           # live_04 gen 1: node_001 fresh mean +0.0010 over the baseline -> reset
    o = R.OfficialRule(0.6015)                                   # the literal single-seed rule, tracked for reporting
    o.update(0.6030, 1); o.update(0.6032, 2); o.update(0.6033, 3)
    assert o.converged_at == 3 and o.best == 0.6015              # would have stopped at generation 3
    o.update(0.6045, 4); assert o.streak == 0 and o.best == 0.6045

def test_pick_champion_and_confirm_stats():
    from harness.loop import pick_champion, confirm_stats
    res = [{'n': 5, 'metrics': {'primary': 0.6036}, 'accepted': False, 'seed_confirmation': {'delta_mean': 0.0006}},   # lucky seed, rejected
           {'n': 6, 'metrics': {'primary': 0.6030}, 'accepted': True, 'seed_confirmation': {'delta_mean': 0.0012}},
           {'n': 7, 'metrics': {'primary': 0.6031}, 'accepted': True, 'seed_confirmation': {'delta_mean': 0.0009}},
           {'n': 8, 'metrics': None, 'accepted': False}]
    assert pick_champion(res) == 6                               # accepted, best seed-mean gain — not the best single seed
    assert pick_champion([res[0], res[3]]) is None
    from harness.loop import pooled_sigma
    sigma, df = pooled_sigma([[0.60147, 0.60176, 0.60109], [0.60304, 0.60232, 0.60261]])
    assert df == 4 and 0.00025 < sigma < 0.00035
    m1, m2, diff, se, z, ok = confirm_stats([0.60447, 0.6045, 0.60399], [0.60304, 0.60232, 0.60261], 0.0003)   # live_04 node_015
    assert diff == pytest.approx(0.00166, abs=1e-5) and z > 6 and ok
    assert confirm_stats([0.6035, 0.6036, 0.6034], [0.6030, 0.6031, 0.6029], 0.0003)[5] is False   # +0.0005 at z ~ 2: borderline, not accepted
    assert confirm_stats([0.6040, 0.6041, 0.6039], [0.6030, 0.6031, 0.6029], 0.0003)[5] is True    # +0.0010 at z ~ 4

@pytest.mark.skipif(not (C.WS_DATA / 'valid.csv').exists(), reason='workspace not built')
def test_read_predictions_validates_alignment(tmp_path):
    uid, vid, y = R.valid_index()
    good = tmp_path / 'good.csv'
    with open(good, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(R.HEADER)
        for i in range(len(uid)):
            w.writerow([i, uid[i], vid[i], float(y[i])])
    scores = R.read_predictions(good)
    m = R.score(scores)
    assert m['gauc'] == pytest.approx(1.0) and m['ndcg5'] == pytest.approx(0.6968, abs=1e-4)   # oracle on valid
    bad = tmp_path / 'bad.csv'
    rows = list(csv.reader(open(good))); rows[3][2] = '-1'
    csv.writer(open(bad, 'w', newline='')).writerows(rows)
    with pytest.raises(ValueError, match='misaligned'):
        R.read_predictions(bad)
    short = tmp_path / 'short.csv'
    csv.writer(open(short, 'w', newline='')).writerows(rows[:100])
    with pytest.raises(ValueError):
        R.read_predictions(short)
