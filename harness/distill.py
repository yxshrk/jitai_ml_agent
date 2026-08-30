"""Cross-run memory (ADR-0004): fold a run's journal back into the method cards.

For every node that used a card: append a line to the card's `## Measured` section and update `status` / `evidence`
in the front matter. Idempotent (a run:node reference is added once). Status rules:
  accepted (seed-confirmed)                  -> alive
  measured but not accepted, on some stack   -> dead_under {run, parent stack, seed-mean delta}
  errored / never produced a script          -> unchanged status, a note in Measured
`dead_under` is contextual: the stack it was measured on is recorded so a Selector can argue a retest when the
champion stack changes."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from . import config as C

def _front(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    return (m.group(1), text[m.end():]) if m else (None, text)

def _stack(nodes, n):
    """Accepted method chain from node 0 to node n, e.g. 'official FM + loss-bpr-pairwise-within-user'."""
    chain, cur = [], nodes.get(str(n))
    while cur is not None:
        if cur.get('action') == 'reproduce_baseline':
            chain.append('official FM'); break
        if cur.get('accepted') and cur.get('method'):
            chain.append(cur['method'])
        p = cur.get('parent'); cur = nodes.get(str(p)) if p is not None else None
    return ' + '.join(reversed(chain)) or 'official FM'

MEASURED_RE = re.compile(r'^- (?P<ref>[\w-]+:node_\d+) on \[(?P<stack>[^\]]+)\]: (?P<rest>.*)$')
DELTA_RE = re.compile(r'seed-mean Δ ([+-]?\d+\.\d+)|single-seed Δ ([+-]?\d+\.\d+)')

def summarize(text):
    """Recompute `status` and the one-line verdict at the top of ## Measured from all Measured lines."""
    fm, body = _front(text)
    if fm is None or '## Measured' not in body:
        return text
    head, tail = body.split('## Measured', 1)
    lines = [l for l in tail.splitlines() if l.startswith('- ')]
    per_stack, accepted, failed = {}, [], []
    for l in lines:
        m = MEASURED_RE.match(l)
        if not m:
            continue
        ref, stack, rest = m.group('ref'), m.group('stack'), m.group('rest')
        if rest.startswith('FAILED'):
            failed.append(ref); continue
        sm = re.search(r'seed-mean Δ ([+-]?\d+\.\d+)', rest); ss = re.search(r'single-seed Δ ([+-]?\d+\.\d+)', rest)
        delta = float(sm.group(1)) if sm else (float(ss.group(1)) if ss else None)   # prefer the seed-mean
        per_stack.setdefault(stack, []).append(delta)
        if 'ACCEPTED' in rest:
            accepted.append((ref, stack, delta))
    if accepted:
        status = 'alive — accepted on [' + '], ['.join(sorted({a[1] for a in accepted})) + ']'
        verdict = (f"ACCEPTED {len(accepted)}x (" + '; '.join(f"{r} on [{st}] Δ {d:+.4f}" for r, st, d in accepted if d is not None) + ')')
    elif per_stack:
        parts = [f"{st} x{len(ds)} (best Δ {max(x for x in ds if x is not None):+.4f})" for st, ds in per_stack.items() if any(x is not None for x in ds)]
        status = 'dead_under [' + '; '.join(parts) + ']'
        verdict = f"never accepted in {sum(len(v) for v in per_stack.values())} measurements on {len(per_stack)} stack(s); " + '; '.join(parts)
    else:
        status = 'untried' + (f" (implementation failed: {', '.join(failed)})" if failed else '')
        verdict = 'no measurement yet' + (f"; implementation failed in {', '.join(failed)}" if failed else '')
    if failed and per_stack:
        verdict += f"; implementation failed in {', '.join(failed)}"
    fm = re.sub(r'^status:.*$', f'status: {status}', fm, flags=re.M)
    tail_lines = [l for l in tail.splitlines() if not l.startswith('_Verdict:_') and l.strip() != '(none yet)']
    body_tail = '\n_Verdict:_ ' + verdict + '\n' + '\n'.join(l for l in tail_lines if l.strip())
    return f'---\n{fm}\n---\n{head}## Measured{body_tail}\n'

def rebuild(methods_dir=None, log=print):
    methods_dir = Path(methods_dir or C.KB / 'methods')
    for card in sorted(methods_dir.glob('*.md')):
        if card.name == 'README.md':
            continue
        new = summarize(card.read_text())
        if new != card.read_text():
            card.write_text(new); log(f'  {card.name}: status -> {re.search(r"^status: (.*)$", _front(new)[0], re.M).group(1)[:110]}')

def distill(run_id, methods_dir=None, log=print):
    methods_dir = Path(methods_dir or C.KB / 'methods')
    run_dir = C.RUNS / run_id
    recs = [json.loads(l) for l in (run_dir / 'journal.jsonl').read_text().splitlines() if l.strip()]
    nodes = {str(r['n']): r for r in recs if r.get('n') is not None}
    touched = {}
    for r in recs:
        if r.get('action') not in ('improve', 'merge', 'retest', 'explore') or not r.get('method'):
            continue
        card = methods_dir / f"{r['method']}.md"
        if not card.exists():
            log(f"  no card for method {r['method']!r} (node_{r['n']:03d}); skipped"); continue
        ref = f"{run_id}:node_{r['n']:03d}"
        text = card.read_text()
        if ref in text:
            continue
        fm, body = _front(text)
        if fm is None:
            log(f"  {card.name}: no front matter; skipped"); continue
        stack = _stack(nodes, r['parent']) if r.get('parent') is not None else 'official FM'
        m = r.get('metrics') or {}; conf = r.get('seed_confirmation') or r.get('grey_confirmation') or {}
        if m:
            d1 = r.get('realized_delta'); dm = conf.get('delta_mean')
            line = (f"- {ref} on [{stack}]: primary {m['primary']:.4f}, single-seed Δ {d1:+.4f}"
                    + (f", seed-mean Δ {dm:+.4f} (t {conf.get('t')})" if dm is not None else '')
                    + f" — {'ACCEPTED' if r.get('accepted') else 'rejected'}; {r.get('diff_lines')} changed lines")
            if r.get('accepted'):
                status = 'alive'
            else:
                status = f"dead_under {{run: {run_id}, stack: {stack}, delta: {dm if dm is not None else d1:+.4f}}}"
        else:
            line = f"- {ref} on [{stack}]: FAILED at {r.get('failure_stage')} — {str(r.get('error'))[:120]} (recovery: {r.get('recovery')})"
            status = None
        # front matter: status + evidence
        if status:
            fm = re.sub(r'^status:.*$', f'status: {status}', fm, flags=re.M)
        ev = re.search(r'^evidence:\s*\[(.*?)\]\s*$', fm, flags=re.M)
        refs = [x.strip() for x in (ev.group(1).split(',') if ev and ev.group(1).strip() else [])] + [ref]
        fm = re.sub(r'^evidence:.*$', f"evidence: [{', '.join(refs)}]", fm, flags=re.M)
        # body: Measured section
        if '## Measured' in body:
            head, tail = body.split('## Measured', 1)
            tail = tail.replace('\n(none yet)', '', 1)
            body = head + '## Measured' + tail.rstrip('\n') + '\n' + line + '\n'
        else:
            body = body.rstrip('\n') + '\n\n## Measured\n' + line + '\n'
        card.write_text(summarize(f'---\n{fm}\n---\n{body}'))
        touched[card.name] = status or 'noted'
        log(f"  {card.name}: {status or 'noted'}  <- {ref}")
    return touched

if __name__ == '__main__':
    rebuild() if sys.argv[1] == '--rebuild' else print(distill(sys.argv[1]))
