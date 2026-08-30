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
