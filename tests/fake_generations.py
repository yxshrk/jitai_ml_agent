"""Scripted generations for FakeBrain: three variants of node_000 (two hyper-parameter changes and one broken
script that exercises the fixer/recovery path). Codes are derived from the seed script by string substitution so
they always run under the contract."""
from pathlib import Path
from harness import config as C

def _seed():
    return (C.SEEDS / 'node_000_fm.py').read_text()

def fake_generations():
    base = _seed()
    k8 = base.replace("ap.add_argument('--k', type=int, default=16)", "ap.add_argument('--k', type=int, default=8)")
    lr2 = base.replace("ap.add_argument('--lr', type=float, default=1e-3)", "ap.add_argument('--lr', type=float, default=2e-3)")
    broken = base.replace("m.V, m.W, m.b = best_state", "m.V, m.W, m.b = best_state_typo   # deliberate NameError")
    assert k8 != base and lr2 != base and broken != base
    sel = lambda tc, hyp, card: {'type': 'improve', 'card': card, 'target_component': tc, 'hypothesis': hyp,
                                 'expected_delta': 0.003, 'expected_delta_basis': 'fake basis', 'parent': 'champion',
                                 'rejected_alternative': {'card': 'none', 'reason': 'fake'}}
    return [[
        (sel('model', 'FAKE: embedding size k=8 instead of 16', 'embedding-size'), k8),
        (sel('training-schedule', 'FAKE: learning rate 2e-3 instead of 1e-3', 'lr-sweep'), lr2),
        (sel('features', 'FAKE: a broken script (NameError) to exercise the fixer', 'broken'), broken),
    ]]
