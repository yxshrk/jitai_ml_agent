import json
import pytest
from harness.brain import parse_header, parse_code, ParseError, FakeBrain
from harness.journal import Journal

REPLY = '''Here is my proposal.
```json
{"change_summary": "swap loss", "selections": [{"card": "bpr"}]}
```
and the script:
```python
import numpy as np
print("hi")
```
'''

def test_parse_header_and_code():
    assert parse_header(REPLY)['change_summary'] == 'swap loss'
    assert parse_code(REPLY).startswith('import numpy')
    with pytest.raises(ParseError):
        parse_header('no fences here')
    with pytest.raises(ParseError):
        parse_code('```json\n{}\n```')

def test_fake_brain_roundtrip():
    sel = {'type': 'improve', 'card': 'x', 'target_component': 'loss', 'hypothesis': 'H', 'expected_delta': 0.003,
           'expected_delta_basis': 'b'}
    b = FakeBrain([[(sel, 'print(1)\n')]])
    ctx = {'generation': 1, 'parent_code': 'parent'}
    assert b.select(ctx, 3) == [sel]
    assert b.implement(ctx, sel, 'parent')['code'] == 'print(1)\n'
    assert b.fix(ctx, 'bad', 'err', '')['code'] == 'parent'
    assert b.critique(ctx, 'c', sel)['verdict'] == 'ok'

def test_journal_roundtrip(tmp_path):
    j = Journal(tmp_path / 'run')
    j.append({'n': 0, 'action': 'reproduce_baseline', 'parent': None, 'hypothesis': 'baseline',
              'metrics': {'gauc': 0.6671, 'ndcg5': 0.5358, 'primary': 0.6015}})
    j.append({'n': 1, 'action': 'improve', 'parent': 0, 'hypothesis': 'bpr', 'method': 'bpr',
              'metrics': {'gauc': 0.67, 'ndcg5': 0.54, 'primary': 0.605}, 'realized_delta': 0.0035, 'accepted': True})
    j.append({'n': 2, 'action': 'improve', 'parent': 0, 'hypothesis': 'broken', 'error': 'NameError', 'failure_stage': 'smoke', 'recovery': 'abandoned'})
    lines = j.compact_lines()
    assert len(lines) == 3 and 'ACCEPTED' in lines[1] and 'ERROR at smoke' in lines[2]
    path, changed = j.write_diff(1, 0, 'a\nb\n', 'a\nc\n')
    assert changed == 2 and (tmp_path / 'run' / path).exists()
    md = j.render_md({'stop_reason': 'test'})
    assert 'n=1' in md and 'ERROR' in md


def test_distill_summarize_aggregates_stacks():
    from harness.distill import summarize
    card = ("---\nid: x-y\nstatus: untried\nevidence: []\n---\n## Claim\nc\n## Measured\n(none yet)\n"
            "- r1:node_002 on [official FM]: primary 0.6019, single-seed Δ +0.0005 — rejected; 10 changed lines\n"
            "- r2:node_004 on [official FM + BPR]: primary 0.6031, single-seed Δ +0.0002, seed-mean Δ +0.0001 (t 0.4) — rejected; 5 changed lines\n")
    out = summarize(card)
    assert 'status: dead_under [official FM x1 (best Δ +0.0005); official FM + BPR x1 (best Δ +0.0001)]' in out
    assert '_Verdict:_ never accepted in 2 measurements on 2 stack(s)' in out and '(none yet)' not in out
    acc = card.replace('seed-mean Δ +0.0001 (t 0.4) — rejected', 'seed-mean Δ +0.0016 (t 8.2) — ACCEPTED')
    assert 'status: proven — accepted on [official FM + BPR]' in summarize(acc)

def test_seed_cache_migration(tmp_path, monkeypatch):
    import json, pytest
    from harness import config as C
    from harness.loop import Loop
    from harness.brain import FakeBrain
    monkeypatch.setattr(C, 'RUNS', tmp_path)
    d = tmp_path / 'r'; d.mkdir(); (d / 'nodes').mkdir()
    (d / 'state.json').write_text(json.dumps({'run_id': 'r', 'n_next': 1, 'generation': 0, 'champion': 0, 'best': 0.6, 'streak': 0,
        'nodes': {'0': {'n': 0, 'metrics': {'primary': 0.6}, 'parent': None, 'action': 'reproduce_baseline', 'accepted': True}},
        'plan': None, 'parked': [], 'start': 1.0, 'last_save': 101.0, 'interventions': 0, 'stop_reason': None,
        'usage': {}, 'champion_seeds': {'0:1': 0.61, '0:2': None}, 'final_seeds': {'3:1': 0.62}}))
    lp = Loop('r', FakeBrain([[]]), k=2)
    assert lp.state['seed_cache'] == {'0:1': 0.61, '3:1': 0.62} and 'champion_seeds' not in lp.state
    assert 99 <= lp.elapsed() < 110                             # resumed: 100 s of earlier running time, not the calendar gap
    assert lp.champion_mean() == pytest.approx(0.61)              # fresh seeds only (seed 0 is the screen)


def test_librarian_batch_validation(tmp_path, monkeypatch):
    from harness import prompts as P
    from harness.librarian import run_librarian
    from harness.brain import Brain
    good = "---\nid: %s\nfamily: model\ntarget_component: model\nsource: x\napplies_when:\n  - a\nexpected_delta: [0.0, 0.001]\nexpected_delta_basis: b\ncost: c\ncomposes_with: [%s]\nconflicts_with: []\nstatus: untried\nevidence: []\n---\n## Claim\nx\n## Mechanism\nx\n## How to implement on node_000\nx\n## Risks\nx\n## Measured\n(none yet)\n"
    class Stub(Brain):
        def librarian(self, ctx, example):
            return [{'id': 'model-alpha', 'card': good % ('model-alpha', 'model-beta')},
                    {'id': 'model-beta', 'card': (good % ('model-beta', 'model-alpha')).replace('target_component: model', 'target_component: optimizer')}]
    (tmp_path / 'loss-bpr-pairwise-within-user.md').write_text(good % ('loss-bpr-pairwise-within-user', ''))
    monkeypatch.setattr(P, 'refresh_menu', lambda: None); monkeypatch.setattr(P, 'untried_cards', lambda: [])
    written = run_librarian(Stub(), n=2, methods_dir=tmp_path, log=lambda *a: None)
    assert written == ['model-alpha']                                           # beta fails the validator (bad component)
    assert 'composes_with: []' in (tmp_path / 'model-alpha.md').read_text()     # the dangling reference to beta was scrubbed
