import csv, math
import pytest
from harness import referee as R, config as C

def test_static_check_flags_forbidden_paths():
    assert R.static_check("open('KuaiRand-Pure/data/x.csv')") == ['KuaiRand-Pure']
    assert 'private/' in R.static_check("p = 'private/test_features.csv'")
    assert R.static_check("import numpy as np\nprint('hello')") == []

def test_accept_and_convergence():
    assert R.accept(0.6015, 0.6040) == (True, pytest.approx(0.0025))
    assert R.accept(0.6015, 0.6030)[0] is False
    c = R.Convergence()                                          # ADR-0012: confirmed champion changes of >= RESET_MIN_GAIN reset the streak
    assert c.update(True, 0.0012) is True and c.streak == 0
    assert c.update(False) is False and c.update(False) is False and c.streak == 2 and not c.converged
    assert c.update(True, 0.0006) is False and c.streak == 3 and c.converged   # a small confirmed change moves the champion but buys no time
    c = R.Convergence(2); assert c.update(True, 0.0017) is True and c.streak == 0
    assert all(c.update(False) is False for _ in range(3)) and c.converged
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
    m1, m2, diff, se, t, ok = confirm_stats([0.60304, 0.60232, 0.60261], [0.60147, 0.60176, 0.60109])
    assert diff == pytest.approx(0.00122, abs=1e-5) and ok and t > C.T_CRIT
    assert confirm_stats([0.6020, 0.6021, 0.6019], [0.6015, 0.6018, 0.6011])[5] is False   # +0.0005 at t ~ 2: not enough

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
