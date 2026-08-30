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
from .journal import Journal, diff_lines

def pick_champion(results):
    """The accepted node with the largest seed-mean gain (falls back to the single-seed delta), or None.
    A rejected node with the best single seed must not block an accepted one (ADR-0012)."""
    acc = [r for r in results if r.get('metrics') and r.get('accepted')]
    if not acc:
        return None
    def gain(r):
        c = r.get('seed_confirmation') or {}
        return c.get('delta_mean', r.get('realized_delta') or 0.0)
    return max(acc, key=lambda r: (gain(r), r['metrics']['primary']))['n']

def pooled_sigma(samples, prior=C.SEED_SD_PRIOR, prior_df=C.SEED_SD_PRIOR_DF):
    """Seed-to-seed SD of the primary pooled over every node with >= 2 fresh seeds (Bessel), blended with the prior:
    sigma^2 = (sum (n_i - 1) s_i^2 + prior_df * prior^2) / (sum (n_i - 1) + prior_df). Seed noise is a property of
    the data + model family (every node measured so far: 0.0002-0.0005), so pooling turns a 2-df t-test into a z-test."""
    ss, df = 0.0, 0
    for v in samples:
        if len(v) >= 2:
            ss += (len(v) - 1) * statistics.variance(v); df += len(v) - 1
    return ((ss + prior_df * prior ** 2) / (df + prior_df)) ** 0.5, df

def confirm_stats(v_node, v_ch, sigma):
    """z-test of the difference of fresh-seed means with the pooled seed SD: returns
    (mean_node, mean_champion, diff, se, z, accepted). Seeds are not paired across different scripts (they consume the
    RNG differently), so a paired test would be wrong; seed 0 is excluded from both means because it is the selected
    screen (a maximum of k draws) and would bias the candidate upward."""
    m_node, m_ch = statistics.mean(v_node), statistics.mean(v_ch)
    se = sigma * (1.0 / len(v_node) + 1.0 / len(v_ch)) ** 0.5
    diff = m_node - m_ch; z = diff / se if se else float('inf')
    return m_node, m_ch, diff, se, z, (diff >= C.MIN_EFFECT and z >= C.Z_CRIT)

class Loop:
    def __init__(self, run_id, brain, k=3, max_nodes=C.MAX_ITERS, max_generations=None, seed=C.DEFAULT_SEED,
                 wall_clock_s=C.WALL_CLOCK_S, parallel=True, confirm_seeds=True, seed_script=None, log=print, final_reseed=True,
                 iteration_unit='node', wildcard=True, librarian=True, auto_distill=True, convergence='confirmed', k_later=None):
        self.run_id, self.brain, self.k, self.seed = run_id, brain, k, seed
        # adaptive breadth: k branches in generation 1 (breadth pays when nothing is measured), k_later afterwards,
        # growing back toward k only for the Consolidator's concrete merge/retest slots (live_04: 4 of 5 accepted in
        # generation 1, then 20 nodes for one hit)
        self.k_first, self.k_later = k, (k_later if k_later is not None else k)
        assert convergence in ('confirmed', 'official'); self.convergence = convergence   # ADR-0012: which rule stops the run
        self.librarian_max = 2 if librarian else 0     # ADR-0013: web-searched cards after flat generations, at most twice a run
        self.auto_distill = auto_distill               # ADR-0013: fold the journal into the cards when the run ends
        self.max_nodes, self.max_generations = max_nodes, max_generations or max_nodes
        self.wall_clock_s, self.parallel, self.confirm_seeds = wall_clock_s, parallel, confirm_seeds
        self.final_reseed = final_reseed
        assert iteration_unit in ('node', 'generation'); self.iteration_unit = iteration_unit   # ADR-0006: what the 50 cap counts
        self.wildcard = wildcard     # ADR-0011: one slot per generation goes to the Explorer role
        self.seed_script = Path(seed_script or C.SEEDS / 'node_000_fm.py')
        self.run_dir = C.RUNS / run_id; self.j = Journal(self.run_dir); self._log = log
        self.state_path = self.run_dir / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'run_id': run_id, 'n_next': 0, 'generation': 0, 'champion': None, 'best': None, 'streak': 0,
            'nodes': {}, 'plan': None, 'parked': [], 'start': time.time(), 'interventions': 0, 'stop_reason': None,
            'usage': Usage().snapshot(), 'seed_cache': {}, 'best_single': None, 'librarian_calls': 0, 'elapsed_before': 0.0}
        # one seed cache keyed 'node:seed', never cleared (ADR-0012); older states kept two that were thrown away
        sc = self.state.setdefault('seed_cache', {})
        for k_ in ('champion_seeds', 'final_seeds'):
            sc.update({k2: v for k2, v in (self.state.pop(k_, {}) or {}).items() if v is not None})
        if self.state_path.exists():   # resumed: the wall clock counts running time, not the calendar
            self.state['elapsed_before'] = self.state.get('elapsed_before', 0.0) + max(0.0, self.state.get('last_save', self.state['start']) - self.state['start'])
            self.state['start'] = time.time()
        self.threads = max(1, (os.cpu_count() or 2) // max(1, k)) if parallel else (os.cpu_count() or 2)

    # ---------------- helpers ----------------
    def log(self, msg): self._log(f'[{self.run_id}] {msg}')
    def save(self):
        self.state['last_save'] = time.time(); self.state_path.write_text(json.dumps(self.state, indent=1, default=str))
    def node(self, n): return self.state['nodes'][str(n)]
    @property
    def champion(self): return self.node(self.state['champion'])
    def elapsed(self): return time.time() - self.state['start'] + self.state.get('elapsed_before', 0.0)
    def stack_of(self, n):
        """Accepted method chain from node 0 to n, e.g. 'official FM + loss-bpr-pairwise-within-user' — what is actually in a script."""
        from .distill import _stack
        return _stack(self.state['nodes'], n)
    def fresh_seeds(self, n):
        """Cached validation primaries of node n on seeds other than the screening seed, in seed order."""
        pre = f'{n}:'
        return [v for k_, v in sorted(self.state['seed_cache'].items(), key=lambda kv: int(kv[0].split(':')[1]) if ':' in kv[0] else 0)
                if k_.startswith(pre) and v is not None and int(k_.split(':')[1]) != self.seed]
    def node_mean(self, n):
        """Fresh-seed mean of node n (the statistic acceptance and convergence use); its seed-0 primary if none yet."""
        v = self.fresh_seeds(n)
        return statistics.mean(v) if v else self.node(n)['metrics']['primary']
    def champion_mean(self):
        return self.node_mean(self.state['champion'])
    def _sigma(self):
        """Seed SD pooled over every node of this run with >= 2 fresh seeds, blended with the prior."""
        return pooled_sigma([self.fresh_seeds(int(k)) for k in self.state['nodes']])
    def code_of(self, n): return self.j.node_path(n).read_text()

    def ctx(self, **extra):
        ch = self.champion
        d = {'run_id': self.run_id, 'generation': self.state['generation'], 'k': self.k,
             'max_generations': self.max_generations, 'max_nodes': self.max_nodes, 'nodes_used': self.state['n_next'],
             'champion': ch, 'best': self.state['best'], 'streak': self.state['streak'],
             'journal_lines': self.j.compact_lines(), 'plan': self.state.get('plan'), 'parked': self.state.get('parked', []),
             'last_generation': self.state.get('last_generation', []), 'parent_code': self.code_of(ch['n']),
             'champion_stack': self.stack_of(ch['n'])}
        d.update(extra); return d

    def _brain(self, fn, *args, what='', attempts=2):
        """Call a brain role with retries on transient errors; returns (result, error)."""
        for attempt in range(1, attempts + 1):
            try:
                return fn(*args), None
            except (ParseError, RuntimeError, Exception) as e:   # noqa: BLE001 — anything from the API layer
                err = f'{type(e).__name__}: {str(e)[:300]}'
                self.log(f'  {what} failed (attempt {attempt}): {err}')
                if attempt == attempts or 'budget exhausted' in err:
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
        self.state['champion'] = n; self.state['best'] = res.metrics['primary']; self.state['best_single'] = res.metrics['primary']
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
               'wildcard': bool(selection.get('wildcard', False)),
               'code_path': str(self.j.node_path(n).relative_to(self.run_dir)),
               'diff_path': diff[0] if diff else None, 'diff_lines': diff[1] if diff else None,
               'metrics': res.metrics if res else None, 'history': res.history if res else [],
               'failure_stage': None if (res and res.ok) else (res.stage if res else 'implement'),
               'error': None if (res and res.ok) else (res.error if res else 'no runnable script produced'),
               'log_tail': (res.log_tail[-800:] if res and not res.ok else ''),
               'duration_s': res.duration_s if res else 0.0, 'tokens_in': tokens[0], 'tokens_out': tokens[1],
               'critic': critic, 'recovery': recovery, 'intervention': False,
               'pred_hash': res.pred_hash if res else None, 'identical_to_parent': False,
               'realized_delta': None, 'accepted': False, 'seed_confirmation': None}
        self.state['nodes'][str(n)] = {k: v for k, v in rec.items() if k not in ('log_tail',)}
        return rec

    # ---------------- one generation ----------------
    def generation(self):
        g = self.state['generation'] + 1; self.state['generation'] = g
        planned = sum(1 for s_ in ((self.state.get('plan') or {}).get('plan') or []) if isinstance(s_, dict) and s_.get('type') in ('merge', 'retest'))
        self.k = self.k_first if g == 1 else max(self.k_later, min(self.k_first, self.k_later + planned))
        self.threads = max(1, (os.cpu_count() or 2) // max(1, self.k)) if self.parallel else (os.cpu_count() or 2)
        t_gen = time.time(); snap = self.brain.usage.snapshot()
        self.log(f'=== generation {g}: champion node_{self.champion["n"]:03d} ({self.champion["metrics"]["primary"]:.4f}), streak {self.state["streak"]} ===')
        # the exact journal so far, frozen for this generation, in the cached prefix of the roles that PLAN (ADR-0013);
        # full diffs for the champion lineage, accepted nodes and the last generation, stubs for older rejected nodes
        self.brain.set_context_block(P.run_block(g, self.j.digest(full_diff_nodes=self._diff_focus())), roles=P.PLANNING_ROLES)

        # 1. diagnose + select
        diagnosis, err = self._brain(self.brain.diagnose, self.ctx(), what='diagnose')
        diagnosis = diagnosis or f'(diagnosis unavailable: {err})'
        # With the wildcard on, the Selector still lists k candidates in priority order: the last is a reserve that
        # fills the slot only if one of its own collides with the Explorer's target_component (live_04 lost one
        # branch per generation to that collision when it asked for k-1).
        n_sel = self.k
        # Selector and Explorer both depend only on the diagnosis, so they run concurrently (each is ~1 min at xhigh).
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sel = ex.submit(self._brain, self.brain.select, self.ctx(diagnosis=diagnosis, k=n_sel), n_sel, what='select')
            f_wild = (ex.submit(self._brain, self.brain.explore, self.ctx(diagnosis=diagnosis), what='explore', attempts=1)
                      if self.wildcard else None)
            selections, err = f_sel.result()
            wild, werr = f_wild.result() if f_wild else (None, None)
        selections = selections or []
        if self.wildcard:
            if wild:
                wild['type'] = 'explore'; wild['wildcard'] = True
                selections = [wild] + selections          # first, so _diversify keeps it
                self.log(f"  wildcard: [{wild.get('target_component')}] {str(wild.get('hypothesis'))[:120]}")
            elif werr:
                self.log(f'  wildcard unavailable: {werr}')
        if not selections:
            self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'generation {g} aborted: selector failed ({err})'})
            return self._close_generation(g, [], diagnosis, snap, t_gen)
        selections = self._diversify(selections)
        gen_tokens_in, gen_tokens_out, _ = self._tokens_since(snap)

        # 2. implement + critique each candidate — the k chains run in parallel threads; tokens are attributed per node
        nodes = []
        for sel in selections:
            n = self._new_node_id()
            parent_n, merge_parents = self._resolve_parents(sel)
            sel['parent_n'] = parent_n
            nodes.append({'n': n, 'sel': sel, 'parent': parent_n, 'merge_parents': merge_parents, 'res': None,
                          'diff': None, 'tokens': [0, 0], 'critic': None, 'recovery': None, 'error': None,
                          'parent_code': self.code_of(parent_n),
                          'extra_code': self.code_of(merge_parents[1]) if merge_parents else None})
        def build(nd):
            self.brain.set_tag(nd['n'])
            hist = self._history_for(nd['sel'])
            code, critic, err = self._implement_with_critic(nd['sel'], nd['parent_code'], nd['extra_code'], hist)
            nd['critic'] = critic
            if code is None:
                self.j.node_path(nd['n']).write_text(nd['parent_code'])          # keep the tree consistent
                nd['error'] = err or 'implementer produced no script'
            else:
                self.j.node_path(nd['n']).write_text(code)
                nd['diff'] = self.j.write_diff(nd['n'], nd['parent'], nd['parent_code'], code)
            return nd
        if self.parallel and len(nodes) > 1:
            with ThreadPoolExecutor(max_workers=len(nodes)) as ex:
                list(ex.map(build, nodes))
        else:
            for nd in nodes:
                build(nd)
        self.brain.set_tag(None)
        for nd in nodes:   # token attribution from tagged calls
            calls = [c for c in getattr(self.brain, 'calls', []) if c.get('tag') == nd['n']]
            nd['tokens'] = [sum(c['tokens_in'] for c in calls), sum(c['tokens_out'] for c in calls)]

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

        # 5. referee + journal (confirmation seeds for every positive delta are prefetched in parallel)
        champ_p = self.champion['metrics']['primary']
        def noop(nd):   # byte-identical predictions to the parent: the change did nothing (ADR-0012)
            ph = self.node(nd['parent']).get('pred_hash')
            return bool(ph) and nd['res'] is not None and nd['res'].ok and nd['res'].pred_hash == ph
        if self.confirm_seeds:
            self._prefetch_seeds([nd['n'] for nd in nodes if nd['res'] is not None and nd['res'].ok and nd['res'].metrics['primary'] > champ_p and not noop(nd)])
        results = []
        for nd in nodes:
            rec = self._record(nd['n'], g, nd['parent'], nd['sel'], nd['res'], nd['diff'], tuple(nd['tokens']),
                               nd['critic'], nd['recovery'], merge_parents=nd['merge_parents'])
            if nd['res'] is not None and nd['res'].ok:
                single_ok, delta = R.accept(self.champion['metrics']['primary'], nd['res'].metrics['primary'])
                rec['realized_delta'] = round(delta, 5); rec['single_seed_accept'] = single_ok
                if noop(nd):
                    rec['identical_to_parent'] = True; accepted = False
                else:
                    accepted = single_ok
                    if delta > 0 and self.confirm_seeds:   # the best of k branches on one seed is biased upward: confirm with more seeds
                        accepted, conf = self._confirm_with_seeds(nd['n']); rec['seed_confirmation'] = conf
                rec['accepted'] = accepted
            else:
                rec['error'] = rec['error'] or nd['error']
            self.state['nodes'][str(nd['n'])].update({k: rec.get(k) for k in ('realized_delta', 'accepted', 'seed_confirmation', 'single_seed_accept', 'error', 'identical_to_parent')})
            self.j.append(rec); results.append(self._result_view(rec))
            self.log(f"node_{nd['n']:03d} [{rec['target_component']}] " + (
                (f"primary {rec['metrics']['primary']:.4f} NO-OP (predictions identical to node_{nd['parent']:03d}) rejected" if rec['identical_to_parent'] else
                 f"primary {rec['metrics']['primary']:.4f} (Δ{rec['realized_delta']:+.4f}) {'ACCEPTED' if rec['accepted'] else 'rejected'}")
                if rec['metrics'] else f"ERROR at {rec['failure_stage']}: {rec['error']}"))
        return self._close_generation(g, results, diagnosis, snap, t_gen)

    def _close_generation(self, g, results, diagnosis, snap, t_gen):
        # champion + convergence are decided per generation, in code
        ok = [r for r in results if r.get('metrics')]
        new_champion = pick_champion(results)                       # best seed-mean gain among the accepted nodes (ADR-0012)
        if new_champion is not None:
            self.state['champion'] = new_champion
        if ok:                                                      # the literal single-seed best, reported alongside
            self.state['best_single'] = max(self.state.get('best_single') or 0.0, max(r['metrics']['primary'] for r in ok))
        conv = R.Convergence(self.state['streak'], self.state.get('conv_ref'))
        improved = conv.update(self.champion_mean())                 # ADR-0012: cumulative rise of the champion's fresh-seed mean >= RESET_MIN_GAIN resets
        self.state['conv_ref'] = conv.ref; self.state['best'] = round(self.champion_mean(), 5)
        off = self.state.setdefault('official_rule', {'best_single_seed': self.node(0)['metrics']['primary'], 'streak': 0,
                                                      'converged_at_generation': None, 'champion_at_stop': None})
        o = R.OfficialRule(off['best_single_seed'], off['streak'], off['converged_at_generation'])
        o.update(max((r['metrics']['primary'] for r in ok), default=None), g)
        off.update(o.to_dict())
        if off['converged_at_generation'] == g and off.get('champion_at_stop') is None:
            off['champion_at_stop'] = self.state['champion']         # what the literal rule would have submitted
        self.state['official_rule'] = off
        if self.convergence == 'official':                           # the literal rule stops the run (switch for the judges)
            self.state['streak'] = o.streak; improved = (o.streak == 0)
        else:
            self.state['streak'] = conv.streak
        self._maybe_librarian(g, improved, results)
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

    def _diff_focus(self):
        """Nodes whose diffs the planning roles see in full: the champion's lineage, every accepted node, the last generation."""
        keep = set(); n = self.state['champion']
        while n is not None:
            keep.add(n); n = self.node(n).get('parent')
        keep |= {v['n'] for v in self.state['nodes'].values() if v.get('accepted')}
        keep |= {r['n'] for r in self.state.get('last_generation', []) if r.get('n') is not None}
        return keep

    def _maybe_librarian(self, g, improved, results):
        """After a flat generation, when fewer than k untried cards remain, ask the Librarian (web search) for two
        new cards; at most librarian_max times per run (ADR-0013)."""
        if improved or self.state.get('librarian_calls', 0) >= self.librarian_max or len(P.untried_cards()) >= self.k:
            return
        from .librarian import run_librarian
        self.state['librarian_calls'] = self.state.get('librarian_calls', 0) + 1
        try:
            made = run_librarian(self.brain, n=2, log=self.log, extra='\n'.join(self.j.compact_lines()[-len(results) - 1:]))
        except Exception as e:      # noqa: BLE001 — the menu simply stays as it is
            self.log(f'  librarian failed: {type(e).__name__}: {str(e)[:200]}'); made = []
        self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'librarian (web search) added cards: {made or "none"}'})

    # ---------------- pieces ----------------
    def _history_for(self, sel, limit=8):
        """Journal lines of earlier nodes with the same target_component or method — what the Implementer should learn from."""
        out = []
        for r in self.j.records():
            if r.get('n') is None or r.get('action') in ('event', 'generation'):
                continue
            if r.get('target_component') == sel.get('target_component') or (r.get('method') and r.get('method') == sel.get('card')):
                m = r.get('metrics') or {}; c = r.get('seed_confirmation') or {}
                res = (f"primary {m['primary']:.4f}, Δ {r.get('realized_delta'):+.4f}" + (f", seed-mean {c['delta_mean']:+.4f}" if c else '')
                       + f", {'accepted' if r.get('accepted') else 'rejected'}") if m else f"FAILED: {str(r.get('error'))[:100]}"
                out.append(f"- node_{r['n']:03d} [{r.get('target_component')}] {r.get('method')}: {str(r.get('change_summary') or r.get('hypothesis'))[:160]} -> {res}")
                if r.get('diff_path') and (self.run_dir / r['diff_path']).exists():
                    out.append((r['n'], self.run_dir / r['diff_path']))
        lines = [x for x in out if isinstance(x, str)][-limit:]
        diffs = [x for x in out if not isinstance(x, str)][-2:]      # the two most recent relevant diffs, in full (<= 120 lines)
        for n, dp in diffs:
            body = dp.read_text().splitlines()[:120]
            lines.append(f"diff of node_{n:03d} (what that attempt changed):\n```diff\n" + '\n'.join(body) + '\n```')
        return lines

    def _prefetch_seeds(self, node_ids):
        """Run all missing confirmation seeds (nodes + champion) in parallel; _confirm_with_seeds then reads the cache."""
        cache = self.state['seed_cache']; jobs = []
        for m in list(node_ids) + [self.state['champion']]:
            for i in range(1, C.CONFIRM_SEEDS + 1):
                sd = self.seed + i
                if f'{m}:{sd}' not in cache:
                    jobs.append((m, sd))
        if not jobs:
            return
        def run(job):
            m, sd = job
            r = R.run_script(self.j.node_path(m), self.j.out_dir(m, f'_seed{sd}'), seed=sd, threads=max(1, (os.cpu_count() or 2) // max(1, min(len(jobs), 5))))
            return job, (r.metrics['primary'] if r.ok else None)
        workers = min(len(jobs), 5) if self.parallel else 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for (m, sd), val in ex.map(run, jobs):
                cache[f'{m}:{sd}'] = val

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

    def _implement_with_critic(self, sel, parent_code, extra, hist=None):
        """Implementer -> static firewall -> Critic; up to two 'revise' rounds; veto ends the candidate."""
        critic_log = []
        for round_ in range(3):
            ictx = self.ctx(history_for_implementer=hist, parent_stack=self.stack_of(sel.get('parent_n', self.state['champion'])))
            out, err = self._brain(self.brain.implement, ictx, sel, parent_code, extra, what='implement')
            if out is None:
                return None, critic_log, err
            code = out['code']; sel['change_summary'] = out.get('change_summary')
            hits = R.static_check(code)
            if hits:
                critic_log.append({'round': round_, 'verdict': 'veto', 'reasons': [f'static firewall: {hits}']})
                sel['critic_instructions'] = f'Remove every reference to {hits} (also from comments and docstrings); only --data-dir files may be read.'
                continue
            n_diff = diff_lines(parent_code, code); limit = 400 if sel.get('type') == 'merge' else C.MAX_DIFF_LINES
            if n_diff > limit and round_ < 2:
                critic_log.append({'round': round_, 'verdict': 'revise', 'reasons': [f'diff too large: {n_diff} changed lines (limit {limit}) - the parent was rewritten instead of edited']})
                sel['critic_instructions'] = (f'Your last version changed {n_diff} lines. Return the PARENT script byte-for-byte except for the '
                                              f'lines this hypothesis needs (typically 5-80). Do not touch the docstring, imports, comments or output code.')
                self.log(f'  diff too large ({n_diff} lines); sending back to the implementer')
                continue
            diff_text = self.j.diff_text(parent_code, code, f"node_{sel.get('parent_n', self.state['champion']):03d} (parent)", 'candidate')
            cctx = self.ctx(parent_stack=self.stack_of(sel.get('parent_n', self.state['champion'])), parent_doc=parent_code[:1200])
            verdict, err = self._brain(self.brain.critique, cctx, code, sel, diff_text, what='critique')
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

    def _ensure_seeds(self, m, seeds):
        cache = self.state['seed_cache']
        for sd in seeds:
            key = f'{m}:{sd}'
            if key not in cache:
                r = R.run_script(self.j.node_path(m), self.j.out_dir(m, f'_seed{sd}'), seed=sd, threads=os.cpu_count() or 2)
                cache[key] = r.metrics['primary'] if r.ok else None

    def _confirm_with_seeds(self, n):
        """ADR-0012 statistics: FRESH seeds (1..CONFIRM_SEEDS) for the candidate and the champion (cached), a z-test of
        the difference of their means with the seed SD pooled over the whole run (prior 0.0003), acceptance iff the gain
        is >= MIN_EFFECT and z >= Z_CRIT; a borderline z gets two more seeds first. Seed 0 is the selected screen and is
        reported but not counted. Measured reason: the best of k single-seed branches is biased upward — +0.0022 on one
        seed was +0.0017 over three; a 3-vs-3 t-test at 2.5 passed 3-6 % of null candidates."""
        base = [self.seed + i for i in range(1, C.CONFIRM_SEEDS + 1)]
        ch = self.state['champion']
        self._ensure_seeds(n, base); self._ensure_seeds(ch, base)
        sigma, df = self._sigma()
        v_node, v_ch = self.fresh_seeds(n), self.fresh_seeds(ch)
        if not v_node or not v_ch:
            return False, {'error': 'confirmation seeds failed', 'node_seeds': v_node, 'champion_seeds': v_ch}
        m_node, m_ch, diff, se, z, accepted = confirm_stats(v_node, v_ch, sigma)
        adaptive = False
        if C.Z_BORDER <= z < C.Z_CRIT and len(v_node) < C.MAX_CONFIRM_SEEDS:
            adaptive = True
            self._ensure_seeds(n, [self.seed + i for i in range(C.CONFIRM_SEEDS + 1, C.MAX_CONFIRM_SEEDS + 1)])
            sigma, df = self._sigma(); v_node = self.fresh_seeds(n)
            m_node, m_ch, diff, se, z, accepted = confirm_stats(v_node, v_ch, sigma)
        self.log(f"node_{n:03d}: seed confirmation — fresh seeds {[round(v, 5) for v in v_node]} mean {m_node:.5f} vs champion "
                 f"{[round(v, 5) for v in v_ch]} mean {m_ch:.5f}: diff {diff:+.5f}, z {z:.1f} (sigma {sigma:.5f}, {df} df"
                 f"{', adaptive' if adaptive else ''}) -> {'ACCEPTED' if accepted else 'rejected'}")
        return accepted, {'node_seed0': self.node(n)['metrics']['primary'], 'node_seeds': v_node, 'champion_seeds': v_ch,
                          'delta_mean': round(diff, 5), 'se': round(se, 6), 'z': round(z, 2), 'sigma_pooled': round(sigma, 6),
                          'sigma_df': df, 'adaptive': adaptive,
                          'rule': f'fresh-seed mean gain >= {C.MIN_EFFECT} and z >= {C.Z_CRIT} with the pooled seed SD'}

    def _result_view(self, rec):
        m = rec.get('metrics') or {}
        view = {k: rec.get(k) for k in ('n', 'parent', 'merge_parents', 'action', 'method', 'target_component', 'hypothesis',
                                        'change_summary', 'expected_delta', 'realized_delta', 'accepted',
                                        'seed_confirmation', 'failure_stage', 'error', 'recovery', 'duration_s')}
        view['metrics'] = {k: v for k, v in m.items() if k != 'by_group'}
        view['curve'] = [round(h.get('val_primary', 0), 4) for h in rec.get('history', [])][:40]
        cg = (self.champion.get('metrics') or {}).get('by_group') or {}
        if m.get('by_group') and cg:   # where the node moved relative to the champion, per tab and duration band
            view['by_group_delta'] = {g: round(m['by_group'][g]['primary'] - cg[g]['primary'], 4) for g in m['by_group'] if g in cg}
        return view

    # ---------------- driver ----------------
    def run(self):
        self.start()
        while True:
            if self.state['streak'] >= C.N_CONVERGE:
                self.state['stop_reason'] = (f'converged: {C.N_CONVERGE} generations without a >= {C.RESET_MIN_GAIN} cumulative rise of the champion fresh-seed mean (ADR-0012)'
                                             if self.convergence == 'confirmed' else f'converged: official rule (single-seed best, eps {C.EPS}, N {C.N_CONVERGE})'); break
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
        summary = self.finish()
        if self.auto_distill:
            try:
                from .distill import distill, archive
                self.log('distilling the journal into the cards'); distill(self.run_id, log=self.log); archive(self.run_id, self.brain, log=self.log)
            except Exception as e:   # noqa: BLE001
                self.log(f'distill/archive failed: {type(e).__name__}: {str(e)[:200]}')
        return summary

    def designate_final(self, top_k=3):
        """Robust final selection (AIRA): among the top_k nodes by validation primary, re-rank by the FRESH-seed mean
        (seeds 1..CONFIRM_SEEDS, cached) so a lucky single seed cannot be the submission. Ties go to the higher seed-0 score."""
        nodes = [v for v in self.state['nodes'].values() if v.get('metrics')]
        top = sorted(nodes, key=lambda v: -v['metrics']['primary'])[:top_k]
        ranking = []
        for v in top:
            if self.final_reseed:
                self._ensure_seeds(v['n'], [self.seed + i for i in range(1, C.CONFIRM_SEEDS + 1)])
            vals = self.fresh_seeds(v['n']) or [v['metrics']['primary']]
            ranking.append({'n': v['n'], 'valid_primary': v['metrics']['primary'], 'fresh_seeds': vals,
                            'mean': statistics.mean(vals), 'std': statistics.stdev(vals) if len(vals) > 1 else None})
        ranking.sort(key=lambda r: (-r['mean'], -r['valid_primary']))
        self.state['designated'] = ranking[0]['n'] if ranking else self.state['champion']
        return ranking

    def _official_submission(self):
        """The node the organizers' literal rule would have submitted: the champion at the generation it converged."""
        off = self.state.get('official_rule') or {}; n = off.get('champion_at_stop')
        if n is None:
            return {'note': 'the literal single-seed rule had not converged when the run ended', 'node': None}
        return {'node': n, 'generation': off.get('converged_at_generation'), 'valid_primary': self.node(n)['metrics']['primary'],
                'fresh_seed_mean': round(self.node_mean(n), 5), 'fresh_seeds': len(self.fresh_seeds(n))}

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
                   'champion_seed_mean': round(self.champion_mean(), 5), 'best_single_seed': self.state.get('best_single'),
                   'convergence_rule': f'ADR-0012 (revised): {C.N_CONVERGE} generations without a seed-confirmed champion change',
                   'official_rule': self.state.get('official_rule'), 'official_rule_submission': self._official_submission(),
                   'convergence_switch': self.convergence,
                   'tokens': {'in_uncached': self.brain.usage.snapshot().get('tokens_in'), 'in_cached': self.brain.usage.snapshot().get('cache_read'),
                              'out': self.brain.usage.snapshot().get('tokens_out')},
                   'interventions': self.state['interventions'], 'k': self.k, 'eps': C.EPS, 'n_converge': C.N_CONVERGE,
                   'iteration_unit': self.iteration_unit, 'iterations_used': self.state['n_next'] if self.iteration_unit == 'node' else self.state['generation']}
        (self.run_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        (self.run_dir / 'journal.md').write_text(self.j.render_md(summary))
        self.save(); self.log(f'run finished: {summary["stop_reason"]} — champion node_{ch["n"]:03d} {ch["metrics"]["primary"]:.4f}')
        return summary
