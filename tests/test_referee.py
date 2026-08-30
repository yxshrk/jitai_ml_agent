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
    c = R.Convergence(0.6015)
    assert c.update(0.6020) is False and c.streak == 1          # sub-epsilon: non-improving
    assert c.update(0.6040) is False and c.streak == 0          # > epsilon vs best: reset, best moves
    assert c.best == 0.6040
    assert c.update(0.6041) is False and c.update(None) is False
    assert c.update(0.6039) is True                              # third consecutive non-improving

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
