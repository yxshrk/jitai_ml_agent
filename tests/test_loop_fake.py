"""End-to-end: one generation of three fake branches on the real workspace (~1 min). Exercises parallel runs,
the referee, the journal, and the fixer/recovery path (one branch is deliberately broken)."""
import json, shutil
import pytest
from harness import config as C
from harness.loop import Loop
from harness.brain import FakeBrain
from tests.fake_generations import fake_generations

@pytest.mark.skipif(not (C.WS_DATA / 'valid.csv').exists(), reason='workspace not built')
def test_one_fake_generation():
    run_id = '_test_fake'
    shutil.rmtree(C.RUNS / run_id, ignore_errors=True)
    loop = Loop(run_id, FakeBrain(fake_generations()), k=3, max_generations=1, confirm_seeds=False, final_reseed=False, wildcard=False)
    summary = loop.run()
    recs = loop.j.records()
    nodes = [r for r in recs if r.get('action') in ('reproduce_baseline', 'improve')]
    assert len(nodes) == 4, [r.get('action') for r in recs]
    base = nodes[0]; assert base['metrics']['primary'] == pytest.approx(0.6015, abs=2e-4)
    broken = [r for r in nodes if 'broken' in (r.get('hypothesis') or '')][0]
    assert broken['recovery'] and broken['recovery'].startswith('patched by fixer')
    assert broken['metrics'] and broken['metrics']['primary'] == pytest.approx(base['metrics']['primary'], abs=1e-6)  # fixer reverted to parent
    assert broken['realized_delta'] == pytest.approx(0.0, abs=1e-6) and broken['accepted'] is False
    assert summary['generations'] == 1 and summary['nodes'] == 4
    assert (C.RUNS / run_id / 'summary.json').exists() and (C.RUNS / run_id / 'journal.md').exists()
    gen = [r for r in recs if r.get('action') == 'generation'][0]
    assert gen['streak'] in (0, 1)
    # ADR-0021 wiring through a real generation: every proposal was queued and run, so nothing waits; the champion is
    # on the frontier; the generation record and the summary carry both. Blocker 1 was invisible without an assertion.
    assert gen['queue_pending'] == 0 and loop.state['queue'] == []
    assert [e['n'] for e in gen['frontier'] if e['champion']] == [loop.state['champion']]
    assert str(loop.state['champion']) in loop.state['frontier']
    assert summary['frontier_on'] is True and summary['queue_pending'] == 0
