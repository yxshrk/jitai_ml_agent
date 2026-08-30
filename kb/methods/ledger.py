"""The family ledger (ADR-0018): one machine-readable table of what the knowledge base has MEASURED, generated from
the cards' front matter and `## Measured` lines plus the oracle bounds in `family_bounds.json`.

    .venv/bin/python kb/methods/ledger.py            # print the table
    .venv/bin/python kb/methods/ledger.py --write    # also write kb/methods/families.json

Importable: build(methods_dir) -> dict (the JSON), write(methods_dir) -> path. The code ranks families from this
table (ADR-0016 `_family_score`); the cards stay the narrative. A card's `expected_delta` is read as it is — the
calibration pass (distill.calibrate) is what rewrites it; the ledger only reports `measured_max` and `bound` next to it."""
import argparse, datetime, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.distill import MEASURED_RE, SCREEN_RE   # one parser for the Measured lines: the writer's
from kb.methods.validate import front_matter

METHODS = ROOT / 'kb' / 'methods'
CLOSED_BOUND = 0.0005   # = MIN_EFFECT: a family whose oracle bound is at or below the acceptance threshold is closed
PAPER_WORDS = re.compile(r'arxiv|\.pdf|et al|paper|kdd|sigir|cikm|recsys|www\b|neurips|icml|iclr', re.I)

def _measured(text):
    """(measured node lines, screen lines) of a card's ## Measured section, parsed."""
    if '## Measured' not in text:
        return [], []
    tail = text.split('## Measured', 1)[1]
    nodes, screens = [], []
    for l in tail.splitlines():
        if not l.startswith('- '):
            continue
        s = SCREEN_RE.match(l)
        if s:
            run, gen = s.group('ref').split(':'); screens.append({'run': run, 'generation': int(gen.rsplit('g', 1)[1]), 'best_gain': float(s.group('gain')), 'kept': s.group('kept') == 'kept', 'stack': s.group('stack')})
            continue
        m = MEASURED_RE.match(l)
        if not m:
            continue
        run, node = m.group('ref').split(':'); rest = m.group('rest')
        sm = re.search(r'seed-mean Δ ([+-]?\d+\.\d+)', rest); ss = re.search(r'single-seed Δ ([+-]?\d+\.\d+)', rest)
        delta = None if rest.startswith('FAILED') else float(sm.group(1)) if sm else (float(ss.group(1)) if ss else None)   # prefer the seed-mean, as summarize() does
        nodes.append({'run': run, 'node': node, 'stack': m.group('stack'), 'delta': delta, 'accepted': 'ACCEPTED' in rest, 'failed': rest.startswith('FAILED')})
    return nodes, screens

def _expected(fm):
    lo_hi = re.findall(r'[-\d.]+', fm.get('expected_delta', ''))
    return [float(lo_hi[0]), float(lo_hi[1])] if len(lo_hi) == 2 else [None, None]

def _basis_class(fm, measured, bounded):
    if measured:
        return 'measured'
    if bounded:
        return 'oracle'
    basis = fm.get('expected_delta_basis', '') + ' ' + fm.get('source', '')
    if re.search(r'facts\.md|facts §|measured|eda', basis, re.I):
        return 'measured-fact'
    return 'paper' if PAPER_WORDS.search(basis) else 'analogy'

def build(methods_dir=METHODS):
    methods_dir = pathlib.Path(methods_dir)
    bounds = json.loads((methods_dir / 'family_bounds.json').read_text())
    sig_of = bounds['cards']; sig = bounds['signal_families']
    cards, families = {}, {}
    for p in sorted(methods_dir.glob('*.md')):
        if p.name == 'README.md':
            continue
        text = p.read_text(); fm = front_matter(text) or {}
        cid = fm.get('id', p.stem); fam = fm.get('family', 'unknown')
        nodes, screens = _measured(text)
        deltas = [n['delta'] for n in nodes if n['delta'] is not None]
        sf = sig_of.get(cid); bound = sig[sf]['bound'] if sf else None
        cards[cid] = {'family': fam, 'signal_family': sf, 'status': fm.get('status', ''), 'expected': _expected(fm),
                      'basis_class': _basis_class(fm, nodes, sf is not None),
                      'measured_max': max(deltas) if deltas else None, 'accepted': any(n['accepted'] for n in nodes),
                      'bound': bound, 'bound_source': sig[sf]['source'] if sf else None}
        f = families.setdefault(fam, {'bound': None, 'bound_source': None, 'status': 'open', 'cards': [], 'screen_gains': [], 'measured': [], 'best_measured': None})
        f['cards'].append(cid)
        f['screen_gains'] += [dict(s, card=cid) for s in screens]
        f['measured'] += [dict(n, card=cid) for n in nodes]
    for fam, f in families.items():
        cs = [cards[c] for c in f['cards']]
        deltas = [m['delta'] for m in f['measured'] if m['delta'] is not None]
        f['best_measured'] = max(deltas) if deltas else None
        if all(c['bound'] is not None for c in cs):        # every card in the family draws on a bounded signal
            f['bound'] = max(c['bound'] for c in cs)
            f['bound_source'] = '; '.join(sorted({c['bound_source'] for c in cs}))
        untried = [c for c in cs if c['status'].startswith('untried')]
        proven_open = [c for c in cs if c['accepted']]
        if f['bound'] is not None and f['bound'] <= CLOSED_BOUND and not proven_open:
            f['status'] = 'bounded'        # nothing in the family can clear the acceptance threshold
        elif not untried and (f['best_measured'] is None or f['best_measured'] <= 0.0005) and f['measured']:
            f['status'] = 'exhausted'
        else:
            f['status'] = 'open'
    return {'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
            'signal_families': sig, 'families': dict(sorted(families.items())), 'cards': dict(sorted(cards.items()))}

def write(methods_dir=METHODS):
    out = pathlib.Path(methods_dir) / 'families.json'
    out.write_text(json.dumps(build(methods_dir), indent=1, ensure_ascii=False) + '\n')
    return out

def table(ledger):
    def fmt(x): return '' if x is None else f'{x:+.4f}'
    lines = [f"{'family':18s} {'status':10s} {'bound':>7s} {'best Δ':>8s} {'cards':>5s} {'meas':>4s} {'scr':>3s}  untried"]
    for fam, f in ledger['families'].items():
        cs = ledger['cards']; untried = [c for c in f['cards'] if cs[c]['status'].startswith('untried')]
        lines.append(f"{fam:18s} {f['status']:10s} {fmt(f['bound']):>7s} {fmt(f['best_measured']):>8s} "
                     f"{len(f['cards']):5d} {len(f['measured']):4d} {len(f['screen_gains']):3d}  {', '.join(untried)}")
    return '\n'.join(lines)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--write', action='store_true'); ap.add_argument('--methods-dir', default=str(METHODS))
    a = ap.parse_args()
    led = build(a.methods_dir); print(table(led))
    unb = [c for c, v in led['cards'].items() if v['bound'] is None and not v['accepted'] and not v['status'].startswith('untried')]
    print(f"\n{len(led['cards'])} cards, {len(led['families'])} families; cards judged by record only (no oracle): {len(unb)}")
    if a.write:
        print('written', write(a.methods_dir))
