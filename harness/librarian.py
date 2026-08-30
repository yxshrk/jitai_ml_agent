"""ADR-0013: the Librarian extends the menu from the literature. It reads the same cached prefix as every role (spec,
foundations, facts, the menu with statuses) plus the run journal block, searches the web through the provider's
web_search tool, and returns n new cards; code validates them, forces status/evidence, and refreshes the menu."""
from __future__ import annotations
import re, sys
from pathlib import Path
from . import config as C, prompts as P
from .journal import Journal

def run_librarian(brain, n=2, run_id=None, methods_dir=None, log=print, extra=''):
    sys.path.insert(0, str(C.KB / 'methods'))
    from validate import validate_text
    methods_dir = Path(methods_dir or C.KB / 'methods')
    existing = {p.stem: p.read_text() for p in methods_dir.glob('*.md') if p.name != 'README.md'}
    example = existing.get('loss-bpr-pairwise-within-user') or next(iter(existing.values()))
    if run_id:   # standalone use: give it the run's journal the way the loop does
        brain.set_context_block(P.run_block(f'after run {run_id}', Journal(C.RUNS / run_id).digest()))
    ctx = {'n': n, 'card_ids': sorted(existing), 'untried': P.untried_cards(), 'has_digest': bool(getattr(brain, '_block', '')),
           'extra': extra}
    cards = brain.librarian(ctx, example) or []
    written = []
    for c in cards:
        cid, text = (c.get('id') or '').strip(), c.get('card') or ''
        m = re.match(r'---\n(.*?)\n---\n', text, re.S)
        if not m:
            log(f'  librarian: card {cid!r} has no front matter; skipped'); continue
        fm, body = m.group(1), text[m.end():]
        fm = re.sub(r'^id:.*$', f'id: {cid}', fm, flags=re.M)
        fm = re.sub(r'^status:.*$', 'status: untried', fm, flags=re.M)
        fm = re.sub(r'^evidence:.*$', 'evidence: []', fm, flags=re.M)
        if c.get('source_url') and c['source_url'] not in fm:
            fm = re.sub(r'^source:(.*)$', lambda mm: f"source:{mm.group(1)} ({c['source_url']})", fm, count=1, flags=re.M)
        text = f'---\n{fm}\n---\n{body.rstrip()}\n'
        if cid in existing:
            log(f'  librarian: {cid} already exists; skipped'); continue
        errors = validate_text(cid, text, set(existing) | {cid})
        if errors:
            log(f'  librarian: {cid} rejected by the validator: {errors[:3]}'); continue
        (methods_dir / f'{cid}.md').write_text(text); existing[cid] = text; written.append(cid)
        log(f"  librarian: new card {cid} — {str(c.get('why_now', ''))[:140]}")
    if written:
        P.refresh_menu()
    return written
