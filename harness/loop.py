"""The loop (ADR-0008/0009): deterministic orchestrator running generations of k parallel branches.

    node 0 = reproduce the baseline
    repeat: diagnose -> select k -> implement/critique each -> smoke (fixer x1) -> full runs in parallel
            -> referee (accept >= EPS, grey-zone reseed) -> journal -> champion/convergence per generation
            -> consolidate (plan merges / retests / explores for the next generation)
    until converged (N non-improving generations), node cap, generation cap, wall-clock, or LLM budget.
The LLM never judges a score; every decision about numbers is in referee.py."""
from __future__ import annotations
import json, os, shutil, statistics, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import config as C, referee as R, prompts as P
from .brain import Usage, ParseError
from .data_access import build as build_workspace
from .journal import Journal

class Loop:
    def __init__(self, run_id, brain, k=3, max_nodes=C.MAX_ITERS, max_generations=None, seed=C.DEFAULT_SEED,
                 wall_clock_s=C.WALL_CLOCK_S, parallel=True, reseed_grey=True, seed_script=None, log=print, final_reseed=True,
                 iteration_unit='node'):
        self.run_id, self.brain, self.k, self.seed = run_id, brain, k, seed
        self.max_nodes, self.max_generations = max_nodes, max_generations or max_nodes
        self.wall_clock_s, self.parallel, self.reseed_grey = wall_clock_s, parallel, reseed_grey
        self.final_reseed = final_reseed
        assert iteration_unit in ('node', 'generation'); self.iteration_unit = iteration_unit   # ADR-0006: what the 50 cap counts
        self.seed_script = Path(seed_script or C.SEEDS / 'node_000_fm.py')
        self.run_dir = C.RUNS / run_id; self.j = Journal(self.run_dir); self._log = log
        self.state_path = self.run_dir / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'run_id': run_id, 'n_next': 0, 'generation': 0, 'champion': None, 'best': None, 'streak': 0,
            'nodes': {}, 'plan': None, 'parked': [], 'start': time.time(), 'interventions': 0, 'stop_reason': None,
            'usage': Usage().snapshot(), 'champion_seeds': {}}
        self.threads = max(1, (os.cpu_count() or 2) // max(1, k)) if parallel else (os.cpu_count() or 2)

    # ---------------- helpers ----------------
    def log(self, msg): self._log(f'[{self.run_id}] {msg}')
    def save(self): self.state_path.write_text(json.dumps(self.state, indent=1, default=str))
    def node(self, n): return self.state['nodes'][str(n)]
    @property
    def champion(self): return self.node(self.state['champion'])
    def elapsed(self): return time.time() - self.state['start']
    def code_of(self, n): return self.j.node_path(n).read_text()

    def ctx(self, **extra):
        ch = self.champion
        d = {'run_id': self.run_id, 'generation': self.state['generation'], 'k': self.k,
             'max_generations': self.max_generations, 'max_nodes': self.max_nodes, 'nodes_used': self.state['n_next'],
             'champion': ch, 'best': self.state['best'], 'streak': self.state['streak'],
             'journal_lines': self.j.compact_lines(), 'plan': self.state.get('plan'), 'parked': self.state.get('parked', []),
             'last_generation': self.state.get('last_generation', []), 'parent_code': self.code_of(ch['n'])}
        d.update(extra); return d

    def _brain(self, fn, *args, what=''):
        """Call a brain role with one retry on transient errors; returns (result, error)."""
        for attempt in (1, 2):
            try:
                return fn(*args), None
            except (ParseError, RuntimeError, Exception) as e:   # noqa: BLE001 — anything from the API layer
                err = f'{type(e).__name__}: {str(e)[:300]}'
                self.log(f'  {what} failed (attempt {attempt}): {err}')
                if attempt == 2 or 'budget exhausted' in err:
                    return None, err
                time.sleep(2)

    def _tokens_since(self, snap):
        d = Usage.delta(snap, self.brain.usage.snapshot())
        return d['tokens_in'], d['tokens_out'], d['cost_usd']

    # ---------------- node 0 ----------------
    def start(self):
        if self.state['champion'] is not None:
            self.log(f"resuming at generation {self.state['generation']}, champion node_{self.state['champion']:03d}")
            return
        build_workspace()
        n = self._new_node_id()
        shutil.copy(self.seed_script, self.j.node_path(n))
        self.log(f'node_{n:03d}: reproducing the official baseline ({self.seed_script.name})')
        res = R.run_script(self.j.node_path(n), self.j.out_dir(n), seed=self.seed, threads=os.cpu_count() or 2)
        rec = self._record(n, generation=0, parent=None, selection={'type': 'reproduce_baseline', 'card': 'official FM',
                           'target_component': None, 'hypothesis': 'Reproduce the official FM baseline under the harness contract.',
                           'expected_delta': 0.0, 'expected_delta_basis': f'published valid primary {C.BASELINE_VALID_PRIMARY}'},
                           res=res, diff=None, tokens=(0, 0), critic=None, recovery=None)
        if not res.ok:
            raise RuntimeError(f'baseline reproduction failed: {res.error}\n{res.log_tail}')
        rec['accepted'] = True; rec['realized_delta'] = None
        self.state['champion'] = n; self.state['best'] = res.metrics['primary']
        self.j.append(rec); self.save()
        self.log(f"node_{n:03d}: valid primary {res.metrics['primary']:.4f} (published {C.BASELINE_VALID_PRIMARY})")

    def _new_node_id(self):
        n = self.state['n_next']; self.state['n_next'] = n + 1; return n

    def _record(self, n, generation, parent, selection, res, diff, tokens, critic, recovery, merge_parents=None):
        rec = {'n': n, 'generation': generation, 'parent': parent, 'merge_parents': merge_parents,
               'action': selection.get('type', 'improve'), 'method': selection.get('card'),
               'target_component': selection.get('target_component'), 'hypothesis': selection.get('hypothesis'),
               'expected_delta': selection.get('expected_delta'), 'expected_delta_basis': selection.get('expected_delta_basis'),
               'rejected_alternative': selection.get('rejected_alternative'), 'change_summary': selection.get('change_summary'),
               'code_path': str(self.j.node_path(n).relative_to(self.run_dir)),
               'diff_path': diff[0] if diff else None, 'diff_lines': diff[1] if diff else None,
               'metrics': res.metrics if res else None, 'history': res.history if res else [],
               'failure_stage': None if (res and res.ok) else (res.stage if res else 'implement'),
               'error': None if (res and res.ok) else (res.error if res else 'no runnable script produced'),
               'log_tail': (res.log_tail[-800:] if res and not res.ok else ''),
               'duration_s': res.duration_s if res else 0.0, 'tokens_in': tokens[0], 'tokens_out': tokens[1],
               'critic': critic, 'recovery': recovery, 'intervention': False,
               'realized_delta': None, 'accepted': False, 'grey_confirmation': None}
        self.state['nodes'][str(n)] = {k: v for k, v in rec.items() if k not in ('log_tail',)}
        return rec

    # ---------------- one generation ----------------
    def generation(self):
        g = self.state['generation'] + 1; self.state['generation'] = g
        t_gen = time.time(); snap = self.brain.usage.snapshot()
        self.log(f'=== generation {g}: champion node_{self.champion["n"]:03d} ({self.champion["metrics"]["primary"]:.4f}), streak {self.state["streak"]} ===')

        # 1. diagnose + select
        diagnosis, err = self._brain(self.brain.diagnose, self.ctx(), what='diagnose')
        diagnosis = diagnosis or f'(diagnosis unavailable: {err})'
        selections, err = self._brain(self.brain.select, self.ctx(diagnosis=diagnosis), self.k, what='select')
        if not selections:
            self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'generation {g} aborted: selector failed ({err})'})
            return self._close_generation(g, [], diagnosis, snap, t_gen)
        selections = self._diversify(selections)
        gen_tokens_in, gen_tokens_out, _ = self._tokens_since(snap)

        # 2. implement + critique each candidate (sequential LLM calls), write node files
        nodes = []
        for sel in selections:
            n = self._new_node_id(); node_snap = self.brain.usage.snapshot()
            parent_n, merge_parents = self._resolve_parents(sel)
            sel['parent_n'] = parent_n
            parent_code = self.code_of(parent_n); extra = self.code_of(merge_parents[1]) if merge_parents else None
            code, critic, err = self._implement_with_critic(sel, parent_code, extra)
            tokens = self._tokens_since(node_snap)[:2]
            if code is None:
                res = None
                self.j.node_path(n).write_text(parent_code)          # keep the tree consistent
                nodes.append({'n': n, 'sel': sel, 'parent': parent_n, 'merge_parents': merge_parents, 'res': None,
                              'diff': None, 'tokens': tokens, 'critic': critic, 'recovery': None,
                              'error': err or 'implementer produced no script'})
                continue
            self.j.node_path(n).write_text(code)
            diff = self.j.write_diff(n, parent_n, parent_code, code)
            nodes.append({'n': n, 'sel': sel, 'parent': parent_n, 'merge_parents': merge_parents, 'res': None,
                          'diff': diff, 'tokens': list(tokens), 'critic': critic, 'recovery': None, 'error': None,
                          'parent_code': parent_code})

        # 3. smoke tests (parallel) with one fixer attempt each
        runnable = [nd for nd in nodes if nd['error'] is None]
        self._parallel(runnable, smoke=True)
        for nd in runnable:
            if not nd['res'].ok:
                self._try_fix(nd, stage='smoke')
        # 4. full runs (parallel) with one fixer attempt on failure
        runnable = [nd for nd in nodes if nd['error'] is None and nd['res'] is not None and nd['res'].ok]
        self._parallel(runnable, smoke=False)
        for nd in runnable:
            if not nd['res'].ok and nd['recovery'] is None:
                self._try_fix(nd, stage='full')

        # 5. referee + journal
        results = []
        for nd in nodes:
            rec = self._record(nd['n'], g, nd['parent'], nd['sel'], nd['res'], nd['diff'], tuple(nd['tokens']),
                               nd['critic'], nd['recovery'], merge_parents=nd['merge_parents'])
            if nd['res'] is not None and nd['res'].ok:
                accepted, delta = R.accept(self.champion['metrics']['primary'], nd['res'].metrics['primary'])
                rec['realized_delta'] = round(delta, 5)
                if not accepted and 0 < delta < C.EPS and self.reseed_grey:
                    accepted, conf = self._confirm_grey(nd['n']); rec['grey_confirmation'] = conf
                rec['accepted'] = accepted
            else:
                rec['error'] = rec['error'] or nd['error']
            self.state['nodes'][str(nd['n'])].update({k: rec[k] for k in ('realized_delta', 'accepted', 'grey_confirmation', 'error')})
            self.j.append(rec); results.append(self._result_view(rec))
            self.log(f"node_{nd['n']:03d} [{rec['target_component']}] " + (
                f"primary {rec['metrics']['primary']:.4f} (Δ{rec['realized_delta']:+.4f}) {'ACCEPTED' if rec['accepted'] else 'rejected'}"
                if rec['metrics'] else f"ERROR at {rec['failure_stage']}: {rec['error']}"))
        return self._close_generation(g, results, diagnosis, snap, t_gen)

    def _close_generation(self, g, results, diagnosis, snap, t_gen):
        # champion + convergence are decided per generation, in code
        ok = [r for r in results if r.get('metrics')]
        gen_best = max(ok, key=lambda r: r['metrics']['primary']) if ok else None
        improved = False
        if gen_best and gen_best['accepted'] and gen_best['metrics']['primary'] > self.champion['metrics']['primary']:
            self.state['champion'] = gen_best['n']; self.state['champion_seeds'] = {}
        if gen_best and gen_best['metrics']['primary'] > self.state['best'] + C.EPS:
            self.state['best'] = gen_best['metrics']['primary']; self.state['streak'] = 0; improved = True
        else:
            self.state['streak'] += 1
        # parked ideas: rejected-but-plausible nodes stay available for a justified retest (ADR-0004)
        for r in results:
            if r.get('metrics') and not r['accepted']:
                self.state['parked'].append({'node': r['n'], 'card': r['method'], 'target_component': r['target_component'],
                                             'delta': r['realized_delta'], 'on_champion': r['parent']})
        self.state['parked'] = self.state['parked'][-12:]
        # consolidator plans the next generation's slots
        plan, err = self._brain(self.brain.consolidate, self.ctx(), results, what='consolidate')
        self.state['plan'] = plan if plan else None
        self.state['last_generation'] = results
        tin, tout, cost = self._tokens_since(snap)
        self.j.append({'n': None, 'generation': g, 'action': 'generation', 'diagnosis': diagnosis,
                       'improved': improved, 'streak': self.state['streak'], 'champion': self.state['champion'],
                       'best': self.state['best'], 'plan': self.state['plan'], 'tokens_in': tin, 'tokens_out': tout,
                       'cost_usd': round(cost, 4), 'duration_s': round(time.time() - t_gen, 1),
                       'llm_calls': getattr(self.brain, 'calls', [])[-40:]})
        if hasattr(self.brain, 'calls'):
            self.brain.calls = []
        self.state['usage'] = self.brain.usage.snapshot(); self.save()
        self.log(f"generation {g} done: champion node_{self.state['champion']:03d} best {self.state['best']:.4f} "
                 f"streak {self.state['streak']} | tokens {tin}/{tout} | {time.time() - t_gen:.0f}s")
        return improved

    # ---------------- pieces ----------------
    def _diversify(self, selections):
        """Enforce distinct target_components (portfolio diversity); keep the first of each."""
        seen, out = set(), []
        for s in selections:
            tc = s.get('target_component')
            if tc in seen and s.get('type') != 'merge':
                self.log(f'  dropping duplicate target_component {tc!r}: {s.get("hypothesis", "")[:60]}')
                continue
            seen.add(tc); out.append(s)
        return out[:self.k]

    def _resolve_parents(self, sel):
        ch = self.state['champion']; mp = sel.get('merge_parents') or []
        if sel.get('type') == 'merge' and len(mp) >= 2 and all(str(p) in self.state['nodes'] for p in mp[:2]):
            return int(mp[0]), [int(mp[0]), int(mp[1])]
        p = sel.get('parent', 'champion')
        if isinstance(p, int) and str(p) in self.state['nodes'] and self.node(p).get('metrics'):
            return p, None
        return ch, None

    def _implement_with_critic(self, sel, parent_code, extra):
        """Implementer -> static firewall -> Critic; up to two 'revise' rounds; veto ends the candidate."""
        critic_log = []
        for round_ in range(3):
            out, err = self._brain(self.brain.implement, self.ctx(), sel, parent_code, extra, what='implement')
            if out is None:
                return None, critic_log, err
            code = out['code']; sel['change_summary'] = out.get('change_summary')
            hits = R.static_check(code)
            if hits:
                critic_log.append({'round': round_, 'verdict': 'veto', 'reasons': [f'static firewall: {hits}']})
                sel['critic_instructions'] = f'Remove every reference to {hits}; only --data-dir files may be read.'
                continue
            verdict, err = self._brain(self.brain.critique, self.ctx(), code, sel, what='critique')
            verdict = verdict or {'verdict': 'ok', 'reasons': [f'critic unavailable: {err}'], 'instructions': ''}
            critic_log.append({'round': round_, **verdict})
            if verdict['verdict'] == 'ok':
                return code, critic_log, None
            if verdict['verdict'] == 'veto':
                return None, critic_log, 'vetoed by critic: ' + '; '.join(verdict.get('reasons', []))[:300]
            sel['critic_instructions'] = verdict.get('instructions') or '; '.join(verdict.get('reasons', []))
        return None, critic_log, 'critic requested revisions three times'

    def _parallel(self, nodes, smoke):
        if not nodes:
            return
        def run(nd):
            tag = '_smoke' if smoke else ''
            return R.run_script(self.j.node_path(nd['n']), self.j.out_dir(nd['n'], tag), seed=self.seed, smoke=smoke,
                                threads=self.threads)
        if self.parallel and len(nodes) > 1:
            with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
                for nd, res in zip(nodes, ex.map(run, nodes)):
                    nd['res'] = res
        else:
            for nd in nodes:
                nd['res'] = run(nd)

    def _try_fix(self, nd, stage):
        """One fixer attempt; if it fails too, the node is abandoned (recorded as an error with recovery='abandoned')."""
        res = nd['res']; n = nd['n']
        self.log(f"node_{n:03d}: {stage} failed ({res.error}); asking the fixer")
        fix, err = self._brain(self.brain.fix, self.ctx(parent_code=nd['parent_code']), self.code_of(n), res.error, res.log_tail, what='fix')
        if not fix or R.static_check(fix['code']):
            nd['recovery'] = 'abandoned (fixer unavailable or unsafe)'; return
        self.j.node_path(n).write_text(fix['code'])
        nd['diff'] = self.j.write_diff(n, nd['parent'], nd['parent_code'], fix['code'])
        nd['res'] = R.run_script(self.j.node_path(n), self.j.out_dir(n, '_smoke2'), seed=self.seed, smoke=True, threads=self.threads)
        if nd['res'].ok:
            nd['res'] = R.run_script(self.j.node_path(n), self.j.out_dir(n), seed=self.seed, smoke=False, threads=self.threads)
        nd['recovery'] = f"patched by fixer ({fix.get('note', '')[:120]}) -> " + ('ok' if nd['res'].ok else f"failed again: {nd['res'].error}")
        if not nd['res'].ok:
            nd['recovery'] += ' -> abandoned'

    def _confirm_grey(self, n):
        """Grey zone (0 < delta < EPS): compare 3-seed means of the node and the champion."""
        seeds = [self.seed + 1, self.seed + 2]
        def mean_for(m, cache):
            vals = [self.node(m)['metrics']['primary']]
            for s in seeds:
                key = f'{m}:{s}'
                if key not in cache:
                    r = R.run_script(self.j.node_path(m), self.j.out_dir(m, f'_seed{s}'), seed=s, threads=os.cpu_count() or 2)
                    cache[key] = r.metrics['primary'] if r.ok else None
                if cache[key] is not None:
                    vals.append(cache[key])
            return statistics.mean(vals), vals
        cache = self.state['champion_seeds']
        m_node, v_node = mean_for(n, cache); m_ch, v_ch = mean_for(self.state['champion'], cache)
        accepted = (m_node - m_ch) >= C.EPS
        self.log(f"node_{n:03d}: grey-zone confirmation — node mean {m_node:.4f} {v_node} vs champion {m_ch:.4f} {v_ch} -> {'ACCEPTED' if accepted else 'rejected'}")
        return accepted, {'node_seeds': v_node, 'champion_seeds': v_ch, 'delta_mean': round(m_node - m_ch, 5)}

    @staticmethod
    def _result_view(rec):
        return {k: rec.get(k) for k in ('n', 'parent', 'merge_parents', 'action', 'method', 'target_component', 'hypothesis',
                                        'change_summary', 'expected_delta', 'metrics', 'realized_delta', 'accepted',
                                        'grey_confirmation', 'failure_stage', 'error', 'recovery', 'duration_s')} | \
               {'curve': [round(h.get('val_primary', 0), 4) for h in rec.get('history', [])][:40]}

    # ---------------- driver ----------------
    def run(self):
        self.start()
        while True:
            if self.state['streak'] >= C.N_CONVERGE:
                self.state['stop_reason'] = f'converged: {C.N_CONVERGE} generations without > {C.EPS} improvement'; break
            if self.iteration_unit == 'node' and self.state['n_next'] + self.k > self.max_nodes:
                self.state['stop_reason'] = f'iteration cap {self.max_nodes} (counting nodes)'; break
            if self.iteration_unit == 'generation' and self.state['generation'] >= self.max_nodes:
                self.state['stop_reason'] = f'iteration cap {self.max_nodes} (counting generations)'; break
            if self.state['generation'] >= self.max_generations:
                self.state['stop_reason'] = f'generation cap {self.max_generations}'; break
            if self.elapsed() > self.wall_clock_s:
                self.state['stop_reason'] = f'wall-clock {self.wall_clock_s}s'; break
            b = getattr(self.brain, 'budget_usd', None)
            if b is not None and self.brain.usage.cost_usd > b:
                self.state['stop_reason'] = f'LLM budget ${b}'; break
            try:
                self.generation()
            except Exception:       # noqa: BLE001 — a generation must never kill the run
                tb = traceback.format_exc()
                self.log('generation crashed:\n' + tb)
                self.j.append({'n': None, 'generation': self.state['generation'], 'action': 'event', 'note': 'generation crashed: ' + tb[-1500:]})
                self.state['streak'] += 1; self.save()
        return self.finish()

    def designate_final(self, top_k=3, extra_seeds=(1, 2)):
        """Robust final selection (AIRA): among the top_k nodes by validation primary, re-rank by the mean over
        extra seeds so a lucky single seed cannot be the submission. Ties go to the higher single-seed score."""
        nodes = [v for v in self.state['nodes'].values() if v.get('metrics')]
        top = sorted(nodes, key=lambda v: -v['metrics']['primary'])[:top_k]
        cache = self.state.setdefault('final_seeds', {})
        ranking = []
        for v in top:
            vals = [v['metrics']['primary']]
            if self.final_reseed:
                for s in extra_seeds:
                    key = f"{v['n']}:{self.seed + s}"
                    if key not in cache:
                        r = R.run_script(self.j.node_path(v['n']), self.j.out_dir(v['n'], f'_seed{self.seed + s}'),
                                         seed=self.seed + s, threads=os.cpu_count() or 2)
                        cache[key] = r.metrics['primary'] if r.ok else None
                    if cache[key] is not None:
                        vals.append(cache[key])
            ranking.append({'n': v['n'], 'valid_primary': v['metrics']['primary'], 'seeds': vals,
                            'mean': statistics.mean(vals), 'std': statistics.pstdev(vals) if len(vals) > 1 else None})
        ranking.sort(key=lambda r: (-r['mean'], -r['valid_primary']))
        self.state['designated'] = ranking[0]['n'] if ranking else self.state['champion']
        return ranking

    def finish(self):
        ch = self.champion; nodes = [v for v in self.state['nodes'].values() if v.get('metrics')]
        top = sorted(nodes, key=lambda v: -v['metrics']['primary'])[:3]
        ranking = self.designate_final()
        summary = {'run_id': self.run_id, 'stop_reason': self.state['stop_reason'], 'generations': self.state['generation'],
                   'nodes': self.state['n_next'], 'champion': ch['n'], 'champion_metrics': ch['metrics'],
                   'baseline_valid_primary': self.node(0)['metrics']['primary'],
                   'delta_vs_baseline_valid': round(ch['metrics']['primary'] - self.node(0)['metrics']['primary'], 5),
                   'top3_valid': [{'n': v['n'], 'primary': v['metrics']['primary']} for v in top],
                   'designated': self.state.get('designated'), 'final_ranking': ranking,
                   'usage': self.brain.usage.snapshot(), 'wall_clock_s': round(self.elapsed(), 1),
                   'interventions': self.state['interventions'], 'k': self.k, 'eps': C.EPS, 'n_converge': C.N_CONVERGE,
                   'iteration_unit': self.iteration_unit, 'iterations_used': self.state['n_next'] if self.iteration_unit == 'node' else self.state['generation']}
        (self.run_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        (self.run_dir / 'journal.md').write_text(self.j.render_md(summary))
        self.save(); self.log(f'run finished: {summary["stop_reason"]} — champion node_{ch["n"]:03d} {ch["metrics"]["primary"]:.4f}')
        return summary
