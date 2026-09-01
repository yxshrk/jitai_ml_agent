"""Build site/rundata.json from ANY run directory — the site renders whatever run
this points at. Usage: uv run python tools/build_site.py logs/run_bigclock_07
Re-run after a designation change; the site updates automatically."""
import json, sys, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def entries_of(metrics_path):
    try: m = json.loads(Path(metrics_path).read_text())
    except Exception: return [], None, None
    h = m.get('history') or []
    ents = h if isinstance(h, list) else [e for v in h.values() if isinstance(v, list) for e in v]
    return ents, m.get('selected_ensemble'), m

def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else 'logs/run_bigclock_07')
    steps, nodes = [], {}
    for line in open(run / 'journal.jsonl'):
        r = json.loads(line)
        nid = r.get('node_id')
        met = r.get('metrics') or {}
        node = {'id': nid, 'accepted': bool(r.get('accepted')),
                'primary': round(met.get('primary'), 6) if met.get('primary') else None,
                'summary': (r.get('change_summary') or r.get('error') or '')[:420],
                'error': bool(r.get('error')),
                'probes': [], 'members': [], 'selected_ensemble': None,
                'selection': (r.get('method_selection') or None)}
        ents, sel, _ = entries_of(run / nid / 'metrics.json')
        for e in ents:
            c = e.get('config') or {}
            p = e.get('best_primary') or e.get('val_primary') or e.get('primary')
            if e.get('stage') == 'ensemble_member':
                node['members'].append({'seed': e.get('seed'), 'primary': round(float(p or 0), 6)})
            elif p is not None and c.get('dropout') is not None:
                node['probes'].append({'stage': e.get('stage', '?'), 'd': c.get('dropout'),
                                       'g': c.get('gamma') or c.get('lr_gamma'),
                                       'wd': c.get('weight_decay'), 'p': round(float(p), 6)})
        node['selected_ensemble'] = sel
        steps.append(node)
    corpus = []
    for pth in glob.glob(str(ROOT / 'logs/run_*/node_*/progress.log')):
        for line in open(pth):
            try: e = json.loads(line)
            except Exception: continue
            c = e.get('config') or {}
            s = e.get('val_primary') or e.get('primary') or e.get('best_primary') or e.get('score')
            if not isinstance(c, dict):
                continue
            g = c.get('gamma') or c.get('lr_gamma') or c.get('step_decay_factor')
            if isinstance(c, dict) and isinstance(s, (int, float)) and 0.55 < s < 0.62 \
               and c.get('dropout') is not None and g:
                corpus.append({'d': round(float(c['dropout']), 4), 'g': round(float(g), 4),
                               'p': round(float(s), 5)})
    try:
        summ = json.loads((run / 'summary.json').read_text())
        meta = {'run': run.name, 'stop': summ.get('stop_reason'), 'iterations': summ.get('iterations'),
                'best': round(summ['best_metrics']['primary'], 6),
                'tokens': summ.get('tokens_total'), 'wall_s': int(summ.get('wall_s', 0))}
    except Exception:
        meta = {'run': run.name}
    import re as _re
    cards = _re.findall(r'^### ([a-z0-9-]+):', (ROOT/'agent/METHODS.md').read_text(), _re.M)
    payload = {'meta': meta, 'nodes': steps, 'corpus': corpus[:1200], 'cards': cards}
    json.dump(payload, open(ROOT / 'site/rundata.json', 'w'))
    # file://-safe copy for the site (fetch() is blocked off-server)
    (ROOT / 'site/rundata.js').write_text('window.RUNDATA=' + json.dumps(payload) + ';')
    print(f"site data built from {run.name}: {len(steps)} nodes, {len(corpus)} corpus probes")

if __name__ == '__main__':
    main()
