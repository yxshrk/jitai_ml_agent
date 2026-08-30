"""Append-only run journal (JSONL) + code diffs + markdown rendering. This file IS deliverable 3."""
import difflib, json, time
from pathlib import Path

class Journal:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        for d in ('nodes', 'diffs', 'outputs'):
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / 'journal.jsonl'

    def append(self, rec):
        rec = dict(rec); rec.setdefault('ts', time.time())
        with open(self.path, 'a') as fh:
            fh.write(json.dumps(rec, default=str) + '\n')
        return rec

    def records(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def node_path(self, n): return self.run_dir / 'nodes' / f'{n:03d}.py'
    def out_dir(self, n, tag=''): return self.run_dir / 'outputs' / (f'{n:03d}{tag}')

    def write_diff(self, n, parent_n, parent_code, code):
        patch = ''.join(difflib.unified_diff(parent_code.splitlines(True), code.splitlines(True),
                                             fromfile=f'node_{parent_n:03d}.py', tofile=f'node_{n:03d}.py'))
        p = self.run_dir / 'diffs' / f'{n:03d}.patch'
        p.write_text(patch)
        return str(p.relative_to(self.run_dir)), sum(1 for l in patch.splitlines() if l[:1] in '+-' and l[:3] not in ('+++', '---'))

    def compact_lines(self):
        """One line per iteration -- what the proposer sees as history (never a growing transcript)."""
        out = []
        for r in self.records():
            if r.get('action') in ('event',):
                continue
            m = r.get('metrics') or {}
            if r.get('error'):
                tail = f"ERROR at {r.get('failure_stage')}: {str(r['error'])[:90]} (recovery: {r.get('recovery')})"
            else:
                d = r.get('realized_delta')
                tail = f"primary {m.get('primary', float('nan')):.4f} GAUC {m.get('gauc', float('nan')):.4f} nDCG@5 {m.get('ndcg5', float('nan')):.4f}" \
                       + (f" (Δ{d:+.4f} vs champion, {'ACCEPTED' if r.get('accepted') else 'rejected'})" if d is not None else '')
            out.append(f"n={r['n']} node_{r['n']:03d} <- {r.get('parent')} [{r.get('action')}] {r.get('method') or ''}: "
                       f"{(r.get('hypothesis') or '')[:140]} | {tail}")
        return out

    def render_md(self, summary=None):
        L = [f'# Run journal — {self.run_dir.name}', '']
        if summary:
            L += ['## Summary', '```json', json.dumps(summary, indent=1, default=str), '```', '']
        L += ['## Iterations', '']
        for r in self.records():
            if r.get('action') == 'event':
                L.append(f"- _event_ (after n={r.get('n')}): {r.get('note')}"); continue
            m = r.get('metrics') or {}
            L += [f"### n={r['n']} — node_{r['n']:03d} ({r.get('action')}, parent {r.get('parent')})",
                  f"**Hypothesis:** {r.get('hypothesis')}",
                  f"**Method:** {r.get('method')} · expected Δ {r.get('expected_delta')} ({r.get('expected_delta_basis')})",
                  (f"**Result:** GAUC {m.get('gauc'):.4f} · nDCG@5 {m.get('ndcg5'):.4f} · primary {m.get('primary'):.4f} · "
                   f"realized Δ {r.get('realized_delta'):+.4f} · {'ACCEPTED' if r.get('accepted') else 'rejected'}") if m else
                  f"**Result:** ERROR at stage `{r.get('failure_stage')}`: {r.get('error')} — recovery: {r.get('recovery')}",
                  f"**Diff:** `{r.get('diff_path')}` ({r.get('diff_lines')} changed lines) · duration {r.get('duration_s', 0):.0f}s · "
                  f"tokens in/out {r.get('tokens_in', 0)}/{r.get('tokens_out', 0)} · intervention: {r.get('intervention', False)}", '']
        return '\n'.join(L)
