"""Validate method cards: front matter fields, allowed target_component, composes_with / conflicts_with references,
required body sections, expected_delta range, length. Run: .venv/bin/python kb/methods/validate.py
Also importable: validate_text(card_id, text, known_ids) -> [errors]; validate_all(dir) -> [errors]."""
import pathlib, re, sys
ROOT = pathlib.Path(__file__).parent
TARGETS = {'features', 'encoding', 'model', 'loss', 'training-schedule', 'regularization', 'aux-targets', 'history', 'data-weighting', 'ensembling'}
REQ = ['id', 'family', 'target_component', 'source', 'applies_when', 'expected_delta', 'expected_delta_basis', 'cost',
       'composes_with', 'conflicts_with', 'status', 'evidence']
SECTIONS = ['## Claim', '## Mechanism', '## How to implement', '## Risks', '## Measured']
MAX_LINES = 70

def front_matter(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    if not m: return None
    fm = {}
    for line in m.group(1).splitlines():
        if line and not line.startswith(' ') and ':' in line:
            k, v = line.split(':', 1); fm[k.strip()] = v.strip()
    return fm

def validate_text(cid, text, known_ids):
    errors = []
    fm = front_matter(text)
    if fm is None: return [f'{cid}: no front matter']
    for f in REQ:
        if f not in fm: errors.append(f'{cid}: missing field {f}')
    if fm.get('id') != cid: errors.append(f'{cid}: id field {fm.get("id")!r} != file name')
    if not re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)+', cid or ''): errors.append(f'{cid}: id must be a lowercase-hyphen slug')
    if fm.get('target_component') not in TARGETS: errors.append(f'{cid}: bad target_component {fm.get("target_component")!r}')
    for f in ('composes_with', 'conflicts_with'):
        for ref in re.findall(r'[a-z0-9-]+', fm.get(f, '').strip('[]')):
            if ref not in known_ids: errors.append(f'{cid}: {f} references unknown card {ref!r}')
    for s in SECTIONS:
        if s not in text: errors.append(f'{cid}: missing section {s!r}')
    lo_hi = re.findall(r'[-\d.]+', fm.get('expected_delta', ''))
    if len(lo_hi) != 2 or not (0 <= float(lo_hi[0]) <= float(lo_hi[1]) <= 0.02):
        errors.append(f'{cid}: expected_delta must be [lo, hi] within [0, 0.02], got {fm.get("expected_delta")!r}')
    if len(text.splitlines()) > MAX_LINES: errors.append(f'{cid}: {len(text.splitlines())} lines (> {MAX_LINES} — trim, it lives in the prompt prefix)')
    return errors

def validate_all(methods_dir=ROOT):
    cards = {p.stem: p.read_text() for p in pathlib.Path(methods_dir).glob('*.md') if p.name != 'README.md'}
    errors = []
    for cid, text in cards.items():
        errors += validate_text(cid, text, set(cards))
    return len(cards), errors

if __name__ == '__main__':
    n, errors = validate_all()
    print(f'{n} cards checked; {len(errors)} problem(s)')
    for e in errors: print('  -', e)
    sys.exit(1 if errors else 0)
