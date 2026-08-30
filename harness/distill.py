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

MEASURED_RE = re.compile(r'^- (?P<ref>[\w-]+:node_\d+) on \[(?P<stack>[^\]]+)\](?P<variant> \(variant: [^)]*\))?: (?P<rest>.*)$')   # variant = a deepen of the card
DELTA_RE = re.compile(r'seed-mean Δ ([+-]?\d+\.\d+)|single-seed Δ ([+-]?\d+\.\d+)')
# ADR-0015: a feature screen measured on valid before any node — evidence, but not a training measurement
SCREEN_RE = re.compile(r'^- (?P<ref>[\w-]+:screen-g\d+) on \[(?P<stack>[^\]]+)\]: SCREENED (?P<kept>kept|DROPPED) best_gain (?P<gain>[+-]?\d+\.\d+)')

def summarize(text):
    """Recompute `status` and the one-line verdict at the top of ## Measured from all Measured lines."""
    fm, body = _front(text)
    if fm is None or '## Measured' not in body:
        return text
    head, tail = body.split('## Measured', 1)
    lines = [l for l in tail.splitlines() if l.startswith('- ')]
    per_stack, accepted, failed, screens = {}, [], [], []
    for l in lines:
        sm_ = SCREEN_RE.match(l)
        if sm_:
            screens.append((sm_.group('ref'), sm_.group('stack'), sm_.group('kept'), float(sm_.group('gain')))); continue
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
        status = 'proven — accepted on [' + '], ['.join(sorted({a[1] for a in accepted})) + ']'
        verdict = (f"ACCEPTED {len(accepted)}x (" + '; '.join(f"{r} on [{st}] Δ {d:+.4f}" for r, st, d in accepted if d is not None) + ')')
    elif per_stack:
        parts = [f"{st} x{len(ds)} (best Δ {max(x for x in ds if x is not None):+.4f})" for st, ds in per_stack.items() if any(x is not None for x in ds)]
        status = 'dead_under [' + '; '.join(parts) + ']'
        verdict = f"never accepted in {sum(len(v) for v in per_stack.values())} measurements on {len(per_stack)} stack(s); " + '; '.join(parts)
    else:
        dropped = [s for s in screens if s[2] == 'DROPPED']
        existing = (re.search(r'^status:\s*(.*)$', fm, flags=re.M) or [None, ''])[1].strip()
        if existing and not existing.startswith('untried'):
            status = existing                                     # only screens here: a screen never changes a trained status
        else:
            status = 'untried' + (f" (implementation failed: {', '.join(failed)})" if failed else '') + (
                f" (screened out on [{dropped[-1][1]}]: best_gain {max(s[3] for s in dropped):+.4f} on valid, no node built)" if dropped else '')
        verdict = ('no measurement yet' if not screens else
                   f"screened {len(screens)}x before any node (" + ', '.join(f"{r} {k} {g:+.4f}" for r, _, k, g in screens) + ')') \
            + (f"; implementation failed in {', '.join(failed)}" if failed else '')
    if failed and per_stack:
        verdict += f"; implementation failed in {', '.join(failed)}"
    if screens and (per_stack or accepted):
        verdict += f"; screened {len(screens)}x (" + ', '.join(f"{r} {k} {g:+.4f}" for r, _, k, g in screens) + ')'
    fm = re.sub(r'^status:.*$', f'status: {status}', fm, flags=re.M)
    tail_lines = [l for l in tail.splitlines() if not l.startswith('_Verdict:_') and l.strip() != '(none yet)']
    body_tail = '\n_Verdict:_ ' + verdict + '\n' + '\n'.join(l for l in tail_lines if l.strip())
    return f'---\n{fm}\n---\n{head}## Measured{body_tail}\n'

def _card_for(method, methods_dir):
    """The card a node's method name refers to. A Selector deepen names its card as '<card id> — <variant>' (live_06:
    'loss-bpr-pairwise-within-user — same-tab long-duration stream'); such a node is a measurement of the base card, not a
    new card (ADR-0014). Returns (card path or None, variant text or None)."""
    method = str(method or '')
    direct = methods_dir / f"{method}.md"
    if direct.exists():
        return direct, None
    for sep in (' — ', ' – ', ' -- '):
        if sep in method:
            base, variant = method.split(sep, 1)
            if (methods_dir / f"{base.strip()}.md").exists():
                return methods_dir / f"{base.strip()}.md", variant.strip()
    return None, None

def _measured_line(ref, r, stack, run_id, variant=None):
    """One exact evidence line for a card's ## Measured section, plus the provisional status it implies."""
    m = r.get('metrics') or {}; conf = r.get('seed_confirmation') or {}
    sv = f" (variant: {variant})" if variant else ''
    if not m:
        return (f"- {ref} on [{stack}]{sv}: FAILED at {r.get('failure_stage')} — {str(r.get('error'))[:120]} (recovery: {r.get('recovery')})", None)
    d1 = r.get('realized_delta'); dm = conf.get('delta_mean')
    line = (f"- {ref} on [{stack}]{sv}: primary {m['primary']:.4f}, single-seed Δ {d1:+.4f}"
            + ((f", seed-mean Δ {dm:+.4f} (z {conf['z']})" if 'z' in conf else f", seed-mean Δ {dm:+.4f} (t {conf.get('t')})") if dm is not None else '')
            + (' — NO-OP (predictions identical to the parent)' if r.get('identical_to_parent') else '')
            + f" — {'ACCEPTED' if r.get('accepted') else 'rejected'}; {r.get('diff_lines')} changed lines")
    status = 'proven' if r.get('accepted') else f"dead_under {{run: {run_id}, stack: {stack}, delta: {dm if dm is not None else d1:+.4f}}}"
    return line, status

def _has_ref(text, ref):
    """A card carries a measurement for run:node iff the reference is in its evidence list (the Archivist may cite
    the node in `source:`, which is not a measurement)."""
    fm, _ = _front(text)
    ev = re.search(r'^evidence:\s*\[(.*?)\]\s*$', fm or '', flags=re.M)
    return bool(ev) and ref in [x.strip() for x in ev.group(1).split(',')]

def _add_measurement(card, ref, line, status, log=print):
    """Append one evidence line + reference to a card and recompute its status; idempotent per ref."""
    text = card.read_text()
    if _has_ref(text, ref):
        return False
    fm, body = _front(text)
    if fm is None:
        log(f"  {card.name}: no front matter; skipped"); return False
    ev = re.search(r'^evidence:\s*\[(.*?)\]\s*$', fm, flags=re.M)
    refs = [x.strip() for x in (ev.group(1).split(',') if ev and ev.group(1).strip() else [])] + [ref]
    fm = re.sub(r'^evidence:.*$', f"evidence: [{', '.join(refs)}]", fm, flags=re.M)
    if status:
        fm = re.sub(r'^status:.*$', f'status: {status}', fm, flags=re.M)
    if '## Measured' in body:
        head, tail = body.split('## Measured', 1)
        tail = tail.replace('\n(none yet)', '', 1)
        body = head + '## Measured' + tail.rstrip('\n') + '\n' + line + '\n'
    else:
        body = body.rstrip('\n') + '\n\n## Measured\n' + line + '\n'
    card.write_text(summarize(f'---\n{fm}\n---\n{body}'))
    return True

def _clamp_expected(fm, log=print, cid=''):
    """expected_delta is what the mechanism could give where its preconditions hold, never below 0: a measured loss
    belongs in ## Measured. Archivists write honest negatives for dead wildcards; clamp them to [0, 0.0005]."""
    m = re.search(r'^expected_delta:\s*\[([^\]]*)\]', fm, flags=re.M)
    if not m:
        return fm
    nums = re.findall(r'-?\d+(?:\.\d+)?', m.group(1))
    if len(nums) != 2:
        return fm
    lo, hi = float(nums[0]), float(nums[1])
    if lo < 0 or hi < 0:
        lo2, hi2 = max(0.0, lo), max(0.0005, hi)
        log(f"  {cid}: expected_delta [{lo}, {hi}] clamped to [{lo2}, {hi2}] (measured losses live in ## Measured)")
        fm = fm[:m.start()] + f'expected_delta: [{lo2:g}, {hi2:g}]' + fm[m.end():]
    return fm

def _archivable(r, methods_dir):
    """Wildcards, and Selector candidates naming a card that does not exist, once they have a measurement."""
    if r.get('n') is None or not r.get('metrics') or r.get('action') not in ('improve', 'merge', 'retest', 'explore', 'deepen'):
        return False
    return bool(r.get('wildcard')) or (bool(r.get('method')) and _card_for(r['method'], methods_dir)[0] is None)

def archive(run_id, brain, methods_dir=None, log=print):
    """ADR-0013: every measured wildcard becomes a card, written by the Archivist role from the node's actual diff and
    numbers (status / evidence / Measured are filled by code), so the next run's Selector can pick, retest or compose
    it. A wildcard the Archivist recognises as an existing card's mechanism is filed as a measurement of that card."""
    import sys
    sys.path.insert(0, str(C.KB / 'methods'))
    from validate import validate_text
    methods_dir = Path(methods_dir or C.KB / 'methods'); run_dir = C.RUNS / run_id
    recs = [json.loads(l) for l in (run_dir / 'journal.jsonl').read_text().splitlines() if l.strip()]
    nodes = {str(r['n']): r for r in recs if r.get('n') is not None}
    existing = {p.stem: p.read_text() for p in methods_dir.glob('*.md') if p.name != 'README.md'}
    example = existing.get('loss-bpr-pairwise-within-user') or next(iter(existing.values()))
    from . import prompts as P
    from .journal import Journal
    brain.set_context_block(P.run_block(f'run {run_id} (complete)', Journal(run_dir).digest()))   # siblings' results: honest attribution
    made = []
    for r in recs:
        if not _archivable(r, methods_dir):
            continue
        ref = f"{run_id}:node_{r['n']:03d}"
        if any(_has_ref(t, ref) for t in existing.values()):
            continue
        diff_text = (run_dir / r['diff_path']).read_text() if r.get('diff_path') and (run_dir / r['diff_path']).exists() else ''
        stack = _stack(nodes, r['parent']) if r.get('parent') is not None else 'official FM'
        line, status = _measured_line(ref, r, stack, run_id)
        out = brain.archive({'run_id': run_id}, r, diff_text, sorted(existing), example, stack)
        if not out:
            log(f"  node_{r['n']:03d}: archivist returned nothing; skipped"); continue
        if out.get('duplicate_of') and out['duplicate_of'] in existing:
            card = methods_dir / f"{out['duplicate_of']}.md"
            if _add_measurement(card, ref, line, status, log):
                existing[out['duplicate_of']] = card.read_text()
                log(f"  node_{r['n']:03d}: filed as a measurement of {out['duplicate_of']}")
            continue
        cid, text = out.get('id') or '', out.get('card') or ''
        fm, body = _front(text)
        if fm is None or cid in existing:
            log(f"  node_{r['n']:03d}: unusable card ({'exists' if cid in existing else 'no front matter'}); skipped"); continue
        fm = re.sub(r'^id:.*$', f'id: {cid}', fm, flags=re.M)
        fm = re.sub(r'^status:.*$', 'status: untried', fm, flags=re.M)
        fm = re.sub(r'^evidence:.*$', 'evidence: []', fm, flags=re.M)
        if not re.search(r'^source:.*$', fm, flags=re.M):
            fm += f"\nsource: {run_id} Explorer (wildcard)"
        fm = _clamp_expected(fm, log, cid)
        text = f'---\n{fm}\n---\n{body}'
        errors = validate_text(cid, text, set(existing) | {cid})
        if errors:
            log(f"  node_{r['n']:03d}: card {cid} rejected by the validator: {errors[:3]}"); continue
        card = methods_dir / f'{cid}.md'; card.write_text(text)
        _add_measurement(card, ref, line, status, log)
        existing[cid] = card.read_text(); made.append(cid)
        log(f"  node_{r['n']:03d} -> new card {cid} ({status})")
    return made

def _measured_gains(text):
    """Every seed-mean (else single-seed) Δ in a card's ## Measured section."""
    fm, body = _front(text)
    if fm is None or '## Measured' not in body:
        return []
    out = []
    for l in body.split('## Measured', 1)[1].splitlines():
        m = MEASURED_RE.match(l)
        if not m or m.group('rest').startswith('FAILED'):
            continue
        rest = m.group('rest')
        sm = re.search(r'seed-mean Δ ([+-]?\d+\.\d+)', rest); ss = re.search(r'single-seed Δ ([+-]?\d+\.\d+)', rest)
        if sm or ss:
            out.append(float((sm or ss).group(1)))          # the seed-mean when there is one, else the single seed
    return out

def calibrate(methods_dir=None, bounds_path=None, log=print):
    """ADR-0018: the record replaces the promise. (a) A card with measurements gets expected_delta = [0, max measured
    seed-mean gain] (0 if never positive) — the Selector's calibration line showed predicted 0.006/0.004/0.003 realised
    +0.0022/+0.0005/−0.0003 while every accepted gain in five runs was +0.0009…+0.0017. (b) A card without measurements in a
    signal family with an oracle bound (facts §11, family_bounds.json) gets its upper capped at the bound. (c) Every card
    mapped to a bounded family carries one `ceiling:oracle` Measured line so the bound is visible where the evidence is.
    Idempotent; run at the end of every distill."""
    methods_dir = Path(methods_dir or C.KB / 'methods')
    bp = Path(bounds_path or methods_dir / 'family_bounds.json')
    bounds = json.loads(bp.read_text()) if bp.exists() else {}
    bounds = {'bounds': bounds.get('signal_families') or bounds.get('bounds') or {}, 'cards': bounds.get('cards') or {},
              'stack': bounds.get('stack', 'official FM + loss-bpr-pairwise-within-user + ensembling-seed-average')}   # the ledger's schema (research session)
    changed = {}
    for card in sorted(methods_dir.glob('*.md')):
        if card.name == 'README.md':
            continue
        text = card.read_text(); fm, body = _front(text)
        if fm is None:
            continue
        cid = (re.search(r'^id:\s*(.*)$', fm, re.M) or [None, card.stem])[1].strip()
        gains = _measured_gains(text)
        ed = re.search(r'^expected_delta:\s*\[(.*?)\]\s*$', fm, re.M)
        basis = re.search(r'^expected_delta_basis:(.*)$', fm, re.M)
        new_fm = fm
        if gains:
            hi = round(max(0.0, max(gains)), 4)
            new_fm = re.sub(r'^expected_delta:.*$', f'expected_delta: [0.0, {hi:.4f}]', new_fm, flags=re.M)
            tag = f'measured (ADR-0018): best seed-mean gain {max(gains):+.4f} over {len(gains)} measurement(s), so the promise is capped at the record'
            if basis:
                was = basis.group(1).strip()
                if 'measured (ADR-0018)' in was:                   # refresh a stale tag: keep only the original promise after '; was:'
                    was = was.split('; was:', 1)[1].strip() if '; was:' in was else ''
                new_fm = re.sub(r'^expected_delta_basis:.*$', f'expected_delta_basis: {tag}' + (f'; was: {was}' if was else ''), new_fm, count=1, flags=re.M)
        sig = bounds.get('cards', {}).get(cid)
        if sig and sig in bounds.get('bounds', {}) and bounds['bounds'][sig].get('kind', 'oracle') == 'oracle':   # measured headroom never caps a promise
            b = bounds['bounds'][sig]
            if not gains and ed:
                nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', ed.group(1))]
                if nums and max(nums) > b['bound']:
                    new_fm = re.sub(r'^expected_delta:.*$', f"expected_delta: [0.0, {b['bound']:.4f}]", new_fm, flags=re.M)
                    if basis and 'bounded (ADR-0018)' not in basis.group(1):
                        new_fm = re.sub(r'^expected_delta_basis:.*$', f"expected_delta_basis: bounded (ADR-0018) at {b['bound']:+.4f} by the oracle for '{sig}' — {b['source']}; was: " + basis.group(1).strip(), new_fm, count=1, flags=re.M)
        if new_fm != fm:
            card.write_text(f'---\n{new_fm}\n---\n{body}'); changed[card.name] = 'calibrated'
        if sig and sig in bounds.get('bounds', {}) and bounds['bounds'][sig].get('kind', 'oracle') == 'oracle':
            b = bounds['bounds'][sig]
            line = (f"- ceiling:oracle on [{bounds.get('stack', 'official FM')}]: BOUNDED <= {b['bound']:+.4f} for the signal family '{sig}' — "
                    f"{b['source']} (facts §11, kb/data/screens/CEILING.md)")
            if _add_measurement(card, 'ceiling:oracle', line, None, log):
                changed[card.name] = changed.get(card.name, '') + '+oracle'
    for name, what in changed.items():
        log(f'  {name}: {what}')
    return changed

def rebuild(methods_dir=None, log=print):
    methods_dir = Path(methods_dir or C.KB / 'methods')
    for card in sorted(methods_dir.glob('*.md')):
        if card.name == 'README.md':
            continue
        new = summarize(card.read_text())
        if new != card.read_text():
            card.write_text(new); log(f'  {card.name}: status -> {re.search(r"^status: (.*)$", _front(new)[0], re.M).group(1)[:110]}')

def _screen_line(ref, r, stack):
    """One evidence line for a feature screen (ADR-0015): what the probe measured on valid against the champion."""
    bg = r.get('best_gain'); cols = r.get('columns') or {}
    detail = '; '.join(f"{n}: varies {c.get('varies')}, GAUC {c.get('gauc')}, additive {c.get('additive'):+.4f}" for n, c in cols.items())
    return (f"- {ref} on [{stack}]: SCREENED {'kept' if r.get('kept') else 'DROPPED'} best_gain {bg:+.4f} ({r.get('best_column')}); "
            f"stack {r.get('stack_gain') if r.get('stack_gain') is None else format(r.get('stack_gain'), '+.4f')}; {detail}"
            + (f" — new_signal: {str(r.get('new_signal'))[:100]}" if r.get('new_signal') else ''))

def distill(run_id, methods_dir=None, log=print):
    """Fold every measured card-node of a run into its card, and every feature screen of a card (ADR-0015) as evidence
    (idempotent per run:node / run:screen-gN reference)."""
    methods_dir = Path(methods_dir or C.KB / 'methods')
    run_dir = C.RUNS / run_id
    recs = [json.loads(l) for l in (run_dir / 'journal.jsonl').read_text().splitlines() if l.strip()]
    nodes = {str(r['n']): r for r in recs if r.get('n') is not None}
    champion_at = {r.get('generation'): r.get('champion') for r in recs if r.get('action') == 'generation'}   # after each close
    touched = {}
    for r in recs:
        if r.get('action') == 'screen' and r.get('card') and r.get('best_gain') is not None:
            card, variant = _card_for(r['card'], methods_dir)
            if card is None:
                continue                                              # a wildcard's screen: the Archivist's business if a node followed
            g = r.get('generation') or 1
            stack = _stack(nodes, champion_at.get(g - 1, 0))          # the champion the probe was measured against
            ref = f"{run_id}:screen-g{g:02d}"
            if _add_measurement(card, ref, _screen_line(ref, r, stack), None, log):
                touched[card.name] = 'screened'; log(f"  {card.name}: screen {'kept' if r.get('kept') else 'DROPPED'}  <- {ref}")
            continue
        if r.get('action') not in ('improve', 'merge', 'retest', 'explore', 'deepen') or not r.get('method'):
            continue
        card, variant = _card_for(r['method'], methods_dir)
        if card is None:
            log(f"  no card for method {r['method']!r} (node_{r['n']:03d}); the Archivist handles wildcards"); continue
        ref = f"{run_id}:node_{r['n']:03d}"
        stack = _stack(nodes, r['parent']) if r.get('parent') is not None else 'official FM'
        line, status = _measured_line(ref, r, stack, run_id, variant)
        if _add_measurement(card, ref, line, status, log):
            touched[card.name] = status or 'noted'
            log(f"  {card.name}: {status or 'noted'}  <- {ref}")
    calibrate(methods_dir, log=log)              # ADR-0018: promises capped at the record / the oracle bounds
    return touched

if __name__ == '__main__':
    rebuild() if sys.argv[1] == '--rebuild' else print(distill(sys.argv[1]))   # archive needs a brain: use the CLI
