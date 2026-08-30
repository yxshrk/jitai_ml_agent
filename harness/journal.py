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

    @staticmethod
    def diff_text(parent_code, code, name_a="parent", name_b="candidate"):
        return "\n".join(difflib.unified_diff(parent_code.splitlines(), code.splitlines(), name_a, name_b, lineterm="", n=2))

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
            elif r.get('identical_to_parent'):
                tail = f"NO-OP: predictions byte-identical to the parent's (primary {(r.get('metrics') or {}).get('primary', 0):.4f}) → rejected"
            else:
                d = r.get('realized_delta'); c = r.get('seed_confirmation') or {}; e = r.get('expected_delta')
                tail = (f"primary {m.get('primary', 0):.4f} GAUC {m.get('gauc', 0):.4f} nDCG@5 {m.get('ndcg5', 0):.4f}"
                        + (f" (\u0394{d:+.4f}" + (f", seed-mean \u0394{c['delta_mean']:+.4f}" if c else '') + (f", expected {e:+.4f}" if isinstance(e, (int, float)) else '')
                           + f" \u2192 {'ACCEPTED' if r.get('accepted') else 'rejected'})" if d is not None else ''))
            parent = r.get('parent'); parent = f"node_{parent:03d}" if isinstance(parent, int) else 'root'
            out.append(f"n={r['n']} node_{r['n']:03d} <- {parent} [{a}/{r.get('target_component')}{' WILDCARD' if r.get('wildcard') else ''}] {r.get('method') or ''}: "
                       f"{(r.get('hypothesis') or '')[:140]} | {tail}")
        return out

    def digest(self, max_diff_lines=250):
        """The exact run record for the LLM roles (nothing summarised): per generation the diagnosis and the
        consolidator's plan; per node the hypothesis, expected vs realised delta, change summary, critic rounds,
        metrics, learning curve, seed confirmation, failure + recovery, and the diff itself."""
        out = [f'# Run journal — {self.run_dir.name} (exact record of every node; diffs are against the parent script)']
        for r in self.records():
            a = r.get('action')
            if a == 'event':
                out.append(f"- event: {r.get('note')}"); continue
            if a == 'generation':
                plan = r.get('plan') or {}
                out.append(f"\n--- generation {r.get('generation')} closed: {'IMPROVED' if r.get('improved') else 'no improvement'}; "
                           f"streak {r.get('streak')}; champion node_{(r.get('champion') or 0):03d}; best-so-far {r.get('best', 0):.4f}; "
                           f"{r.get('duration_s')} s; ${r.get('cost_usd')}")
                if r.get('diagnosis'):
                    out.append(f"diagnosis given to this generation:\n{r['diagnosis']}")
                if plan:
                    out.append(f"consolidator after this generation: {plan.get('note', '')}\nplan: {json.dumps(plan.get('plan', []), default=str)}")
                continue
            parent = r.get('parent'); parent = f"node_{parent:03d}" if isinstance(parent, int) else 'root'
            mp = r.get('merge_parents')
            out.append(f"\n## node_{r['n']:03d} <- {parent}{' + node_%03d (merge)' % mp[1] if mp else ''} "
                       f"[{a}/{r.get('target_component')}{' WILDCARD' if r.get('wildcard') else ''}] card: {r.get('method') or '-'}")
            out.append(f"- hypothesis: {r.get('hypothesis')}")
            e = r.get('expected_delta')
            if isinstance(e, (int, float)):
                out.append(f"- expected Δ {e:+.4f} (basis: {r.get('expected_delta_basis')}); rejected alternative: {r.get('rejected_alternative')}")
            if r.get('change_summary'):
                out.append(f"- change summary: {r['change_summary']}")
            if r.get('critic'):
                out.append('- critic rounds: ' + json.dumps(r['critic'], default=str)[:1500])
            m = r.get('metrics') or {}
            if m:
                d = r.get('realized_delta'); c = r.get('seed_confirmation')
                line = (f"- result: primary {m.get('primary', 0):.4f} (GAUC {m.get('gauc', 0):.4f}, nDCG@5 {m.get('ndcg5', 0):.4f}, "
                        f"nDCG@5 on mixed-label users {m.get('ndcg5_disc', 0):.4f})")
                if r.get('identical_to_parent'):
                    line += "; NO-OP — predictions byte-identical to the parent's, rejected without seeds"
                elif d is not None:
                    line += f"; Δ {d:+.4f} vs the champion; " + ('ACCEPTED' if r.get('accepted') else 'rejected')
                out.append(line)
                if c:
                    out.append('- seed confirmation: ' + json.dumps(c, default=str))
                h = r.get('history') or []
                if h:
                    best = max(h, key=lambda x: x.get('val_primary', 0))
                    out.append(f"- curve: {len(h)} epochs, best epoch {best.get('epoch')} (train loss there {best.get('train_loss', 0):.4f}); "
                               'valid primary by epoch: ' + ' '.join(f"{x.get('val_primary', 0):.4f}" for x in h[:40]))
                out.append(f"- runtime {r.get('duration_s', 0):.0f} s; tokens {r.get('tokens_in')}/{r.get('tokens_out')}")
            else:
                out.append(f"- FAILED at {r.get('failure_stage')}: {str(r.get('error'))[:400]}; recovery: {r.get('recovery')}")
            dp = r.get('diff_path')
            if dp and (self.run_dir / dp).exists():
                lines = (self.run_dir / dp).read_text().splitlines()
                shown = lines[:max_diff_lines]
                out.append(f"- diff ({r.get('diff_lines')} changed lines" + (f", first {max_diff_lines} shown" if len(lines) > max_diff_lines else '') + '):')
                out.append('```diff\n' + '\n'.join(shown) + '\n```')
        return '\n'.join(out)

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
