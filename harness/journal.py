"""Append-only run journal (JSONL) + code diffs + markdown rendering. This file IS deliverable 3."""
import difflib, json, time
from pathlib import Path

def diff_lines(a, b):
    """Number of changed lines between two scripts (what the judges would read)."""
    return sum(1 for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='') if l[:1] in '+-' and l[:3] not in ('+++', '---'))

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
        """One line per node plus a marker per generation -- what the proposer sees as history."""
        out = []
        for r in self.records():
            a = r.get('action')
            if a == 'event':
                continue
            if a == 'generation':
                out.append(f"--- generation {r.get('generation')}: {'improved' if r.get('improved') else 'no improvement'}, "
                           f"streak {r.get('streak')}, champion node_{(r.get('champion') or 0):03d}, best {r.get('best', 0):.4f} ---")
                continue
            m = r.get('metrics') or {}
            if r.get('error'):
                tail = f"ERROR at {r.get('failure_stage')}: {str(r['error'])[:90]} (recovery: {r.get('recovery')})"
            else:
                d = r.get('realized_delta')
                tail = (f"primary {m.get('primary', 0):.4f} GAUC {m.get('gauc', 0):.4f} nDCG@5 {m.get('ndcg5', 0):.4f}"
                        + (f" (\u0394{d:+.4f} vs champion, {'ACCEPTED' if r.get('accepted') else 'rejected'})" if d is not None else ''))
            parent = r.get('parent'); parent = f"node_{parent:03d}" if isinstance(parent, int) else 'root'
            out.append(f"n={r['n']} node_{r['n']:03d} <- {parent} [{a}/{r.get('target_component')}] {r.get('method') or ''}: "
                       f"{(r.get('hypothesis') or '')[:140]} | {tail}")
        return out

    def render_md(self, summary=None):
        L = [f'# Run journal \u2014 {self.run_dir.name}', '']
        if summary:
            L += ['## Summary', '```json', json.dumps(summary, indent=1, default=str), '```', '']
        L += ['## Iterations', '']
        for r in self.records():
            a = r.get('action')
            if a == 'event':
                L.append(f"- _event_ (generation {r.get('generation')}): {r.get('note')}"); L.append(''); continue
            if a == 'generation':
                L += [f"#### generation {r.get('generation')} closed \u2014 {'improved' if r.get('improved') else 'no improvement'}; "
                      f"streak {r.get('streak')}; champion node_{(r.get('champion') or 0):03d}; best {r.get('best', 0):.4f}; "
                      f"tokens {r.get('tokens_in', 0)}/{r.get('tokens_out', 0)}; {r.get('duration_s', 0):.0f}s",
                      f"_Diagnosis:_ {r.get('diagnosis')}", f"_Plan for next generation:_ `{json.dumps(r.get('plan'), default=str)}`", '']
                continue
            m = r.get('metrics') or {}
            d = r.get('realized_delta')
            result = (f"**Result:** GAUC {m.get('gauc', 0):.4f} \u00b7 nDCG@5 {m.get('ndcg5', 0):.4f} \u00b7 primary {m.get('primary', 0):.4f}"
                      + (f" \u00b7 realized \u0394 {d:+.4f} \u00b7 {'ACCEPTED' if r.get('accepted') else 'rejected'}" if d is not None else '')
                      + (f" \u00b7 seed confirmation {r.get('seed_confirmation')}" if r.get('seed_confirmation') else '')
                      + (f" \u00b7 recovery: {r.get('recovery')}" if r.get('recovery') else '')) if m else \
                     f"**Result:** ERROR at stage `{r.get('failure_stage')}`: {r.get('error')} \u2014 recovery: {r.get('recovery')}"
            L += [f"### n={r['n']} \u2014 node_{r['n']:03d} ({a}, parent {r.get('parent')}{', merge of ' + str(r.get('merge_parents')) if r.get('merge_parents') else ''})",
                  f"**Hypothesis:** {r.get('hypothesis')}",
                  f"**Method:** {r.get('method')} \u00b7 target `{r.get('target_component')}` \u00b7 expected \u0394 {r.get('expected_delta')} ({r.get('expected_delta_basis')})",
                  result,
                  f"**Diff:** `{r.get('diff_path')}` ({r.get('diff_lines')} changed lines) \u00b7 duration {r.get('duration_s') or 0:.0f}s \u00b7 "
                  f"tokens in/out {r.get('tokens_in', 0)}/{r.get('tokens_out', 0)} \u00b7 intervention: {r.get('intervention', False)}", '']
        return '\n'.join(L)
