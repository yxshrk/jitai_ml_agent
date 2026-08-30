"""ADR-0018: the record replaces the promise; oracle bounds cap unmeasured cards; variant lines parse; Selector names validated."""
import json
from pathlib import Path

CARD = "---\nid: {id}\nfamily: {family}\ntarget_component: features\nsource: x\napplies_when:\n  - a\nexpected_delta: [{lo}, {hi}]\nexpected_delta_basis: paper promise\ncost: c\ncomposes_with: []\nconflicts_with: []\nstatus: untried\nevidence: []\n---\n## Claim\nx\n{measured}"


def _methods(tmp_path):
    m = tmp_path / 'methods'; m.mkdir()
    bounds = {'stack': 'official FM + loss-bpr', 'bounds': {'item-side': {'bound': 0.0003, 'source': 'facts §11 row'}},
              'cards': {'features-rates': 'item-side', 'features-measured': 'item-side'}}
    (m / 'family_bounds.json').write_text(json.dumps(bounds))
    (m / 'features-rates.md').write_text(CARD.format(id='features-rates', family='features', lo=0.002, hi=0.008, measured=''))
    (m / 'features-measured.md').write_text(CARD.format(id='features-measured', family='features', lo=0.001, hi=0.006, measured=(
        '\n## Measured\n- r:node_003 on [official FM]: primary 0.6020, single-seed Δ +0.0012, seed-mean Δ +0.0007 (z 3.1) — ACCEPTED; 10 changed lines\n'
        '- r:node_009 on [official FM + loss-bpr] (variant: features-measured — dose 5 %): primary 0.6030, single-seed Δ +0.0001, seed-mean Δ -0.0002 (z -0.4) — rejected; 3 changed lines\n')))
    (m / 'features-never.md').write_text(CARD.format(id='features-never', family='features', lo=0.001, hi=0.004, measured=(
        '\n## Measured\n- r:node_004 on [official FM]: primary 0.6010, single-seed Δ -0.0005 — rejected; 10 changed lines\n')))
    return m


def test_calibrate_caps_promises_at_the_record_and_the_bound(tmp_path):
    from harness.distill import calibrate, _measured_gains, MEASURED_RE
    m = _methods(tmp_path)
    changed = calibrate(m, log=lambda *a: None)
    assert set(changed) == {'features-rates.md', 'features-measured.md', 'features-never.md'}
    rates = (m / 'features-rates.md').read_text()
    assert 'expected_delta: [0.0, 0.0003]' in rates and "bounded (ADR-0018) at +0.0003" in rates and 'ceiling:oracle on [official FM + loss-bpr]: BOUNDED <= +0.0003' in rates
    meas = (m / 'features-measured.md').read_text()
    assert 'expected_delta: [0.0, 0.0007]' in meas and 'measured (ADR-0018): best seed-mean gain +0.0007 over 2 measurement(s)' in meas
    assert _measured_gains(meas) == [0.0007, -0.0002]                                 # the variant line parses (MEASURED_RE)
    assert MEASURED_RE.match('- r:node_009 on [official FM + loss-bpr] (variant: x — y): rest').group('variant')
    never = (m / 'features-never.md').read_text()
    assert 'expected_delta: [0.0, 0.0000]' in never                                     # never positive: no promise left
    assert calibrate(m, log=lambda *a: None) == {}                                      # idempotent
    assert (m / 'features-measured.md').read_text() == meas


def test_selector_parser_validates_card_ids(monkeypatch):
    from harness import prompts as P
    from harness.brain import LLMBrain, Brain
    monkeypatch.setattr(P, 'card_index', lambda: {'loss-bpr-pairwise-within-user': {}, 'features-exposure-session': {}})
    monkeypatch.setattr(P, 'user_select', lambda ctx: 'select')
    answers = iter(['```json\n{"selections": [{"type": "improve", "card": "session feature idea", "target_component": "features", "hypothesis": "h", "expected_delta": 0.001, "expected_delta_basis": "b"}]}\n```',
                    '```json\n{"selections": [{"type": "improve", "card": "session feature idea", "target_component": "features", "hypothesis": "h", "expected_delta": 0.001, "expected_delta_basis": "b"}, {"type": "deepen", "card": "loss-bpr-pairwise-within-user — hard negatives", "target_component": "loss", "hypothesis": "h2", "expected_delta": 0.001, "expected_delta_basis": "b"}]}\n```'])
    class Stub(LLMBrain):
        def __init__(self): Brain.__init__(self); self.calls = []      # LLMBrain.__init__ needs a provider; the stub does not
        def _call(self, role, text, **kw): self.calls.append(text); return next(answers)
    b = Stub(); sels = b.select({'k': 3}, 3)
    assert len(b.calls) == 2 and 'not a card id' in b.calls[1]                        # one format reminder …
    assert sels[0].get('card_unknown') is True and 'card_unknown' not in sels[1]        # … then accepted and flagged; a real deepen name passes
