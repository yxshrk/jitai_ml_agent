"""Validate every card's front matter: required fields, allowed target_component, and that every composes_with /
conflicts_with id exists. Run: .venv/bin/python kb/methods/validate.py"""
import pathlib, re, sys
ROOT = pathlib.Path(__file__).parent
TARGETS = {'features', 'encoding', 'model', 'loss', 'training-schedule', 'regularization', 'aux-targets', 'history', 'data-weighting', 'ensembling'}
REQ = ['id', 'family', 'target_component', 'source', 'applies_when', 'expected_delta', 'expected_delta_basis', 'cost',
       'composes_with', 'conflicts_with', 'status', 'evidence']
SECTIONS = ['## Claim', '## Mechanism', '## How to implement', '## Risks', '## Measured']

def front_matter(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    if not m: return None
    fm = {}
    for line in m.group(1).splitlines():
        if line and not line.startswith(' ') and ':' in line:
            k, v = line.split(':', 1); fm[k.strip()] = v.strip()
    return fm

cards = {p.stem: p.read_text() for p in ROOT.glob('*.md') if p.name != 'README.md'}
errors = []
for cid, text in cards.items():
    fm = front_matter(text)
    if fm is None: errors.append(f'{cid}: no front matter'); continue
    for f in REQ:
        if f not in fm: errors.append(f'{cid}: missing field {f}')
    if fm.get('id') != cid: errors.append(f'{cid}: id field {fm.get("id")!r} != file name')
    if fm.get('target_component') not in TARGETS: errors.append(f'{cid}: bad target_component {fm.get("target_component")!r}')
    for f in ('composes_with', 'conflicts_with'):
        for ref in re.findall(r'[a-z0-9-]+', fm.get(f, '').strip('[]')):
            if ref not in cards: errors.append(f'{cid}: {f} references unknown card {ref!r}')
    for s in SECTIONS:
        if s not in text: errors.append(f'{cid}: missing section {s!r}')
    lo_hi = re.findall(r'[-\d.]+', fm.get('expected_delta', ''))
    if len(lo_hi) != 2 or not (0 <= float(lo_hi[0]) <= float(lo_hi[1]) <= 0.02):
        errors.append(f'{cid}: expected_delta must be [lo, hi] within [0, 0.02], got {fm.get("expected_delta")!r}')
    if len(text.splitlines()) > 70: errors.append(f'{cid}: {len(text.splitlines())} lines (> 70 — trim, it lives in the prompt prefix)')
print(f'{len(cards)} cards checked; {len(errors)} problem(s)')
for e in errors: print('  -', e)
sys.exit(1 if errors else 0)
