"""The loop (ADR-0008/0009): deterministic orchestrator running generations of k parallel branches.

    node 0 = reproduce the baseline
    repeat: diagnose -> select k -> implement/critique each -> smoke (fixer x1) -> full runs in parallel
            -> referee (accept >= EPS, grey-zone reseed) -> journal -> champion/convergence per generation
            -> consolidate (plan merges / retests / explores for the next generation)
    until converged (N non-improving generations), node cap, generation cap, wall-clock, or LLM budget.
The LLM never judges a score; every decision about numbers is in referee.py."""
from __future__ import annotations
import json, os, re, shutil, statistics, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from . import config as C, referee as R, prompts as P
from .brain import Usage, ParseError, Brain
from . import screen as S
from .data_access import build as build_workspace
from .journal import Journal, diff_lines

def _slug(x):
    """Normalise a mechanism tag for comparison: lower-case, non-alphanumerics collapsed to '-'."""
    s = re.sub(r'[^a-z0-9]+', '-', str(x or '').lower()).strip('-')
    return s or None

def _gkey(x):
    """Normalise a breakdown-group name for comparison: 'dur>180s', 'dur > 180 s' and 'DUR>180S' are one group."""
    return re.sub(r'[^a-z0-9]', '', str(x or '').lower()) or None

def _stack_key(stack):
    """A stack string 'official FM + a + b' as a set of its parts — equality, never substring containment."""
    return frozenset(x.strip() for x in str(stack or '').split(' + ') if x.strip())

def _dead_stacks(status):
    """The stacks named in a 'dead_under [stack xN (best Δ); stack xN (...)]' status."""
    m = re.search(r'dead_under \[(.*)\]\s*$', str(status or ''))
    if not m:
        return []
    return [re.sub(r'\s+[x×]\d+.*$', '', part).strip() for part in m.group(1).split(';') if part.strip()]

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

def pooled_sigma(samples):
    """Seed-to-seed SD of the primary pooled over every node of THIS RUN with >= 2 fresh seeds (Bessel):
    sigma^2 = sum (n_i - 1) s_i^2 / sum (n_i - 1). Seed noise is a property of the data + model family (every node
    measured so far: 0.0002-0.0005), so pooling turns a 2-df t-test into a z-test.

    ADR-0020: no prior from earlier runs is blended in. A run's acceptance decisions must rest on evidence the run
    itself gathered, so that what is submitted is the product of one autonomous search rather than of knowledge
    carried between runs. The prior it replaces (0.0003 at 4 df, from live_01/02) sat below every run's true seed SD
    (0.00035-0.00044 measured), so it shrank sigma and inflated every z; removing it makes the test stricter.
    Returns (sigma, df); sigma is None when no node yet has two fresh seeds, and the caller falls back to the
    candidate's own SD."""
    ss, df = 0.0, 0
    for v in samples:
        if len(v) >= 2:
            ss += (len(v) - 1) * statistics.variance(v); df += len(v) - 1
    return ((ss / df) ** 0.5 if df else None), df

def needs_more_seeds(z, accepted, df, n_seeds):
    """Whether a candidate must spend its remaining fresh seeds before the verdict stands. Two reasons: the z-score is
    borderline (Z_BORDER <= z < Z_CRIT), or the verdict ACCEPTS on a pooled seed SD this run has barely estimated
    (df < MIN_SIGMA_DF, ADR-0020) — with no prior carried in from other runs the opening confirmation pools only 4 df,
    where a fixed Z_CRIT is really a ~2 % test rather than 0.13 %. The fix is degrees of freedom, not a higher bar:
    measured on live_08/09's BPR nodes, two more seeds take df 4 -> 6 and z 3.39 -> 4.42 and 5.47 -> 4.19, both still
    accepted. Normally only the first candidate of a run pays, since its extra seeds raise df for everyone after it."""
    return (C.Z_BORDER <= z < C.Z_CRIT or (accepted and df < C.MIN_SIGMA_DF)) and n_seeds < C.MAX_CONFIRM_SEEDS

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
                 iteration_unit='node', wildcard=True, librarian=True, auto_distill=True, convergence='confirmed', k_later=None,
                 screen=True, campaigns=True, designation=None, frontier=True):
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
        # ADR-0015: feature candidates are probed and measured on valid before a node is spent; needs a brain with a probe role
        self.screen = bool(screen) and type(brain).probe is not Brain.probe
        self.campaigns = campaigns   # ADR-0016: one card family per generation from CAMPAIGNS_FROM_GENERATION on
        self.designation = designation or C.DESIGNATION_DEFAULT   # ADR-0012 amendment: strict | adaptive
        self.frontier_on = frontier    # ADR-0021: a frontier of progressing nodes and a persistent proposal queue
        assert self.designation in ('strict', 'adaptive')
        self.seed_script = Path(seed_script or C.SEEDS / 'node_000_fm.py')
        self.run_dir = C.RUNS / run_id; self.j = Journal(self.run_dir); self._log = log
        self.state_path = self.run_dir / 'state.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'run_id': run_id, 'n_next': 0, 'generation': 0, 'champion': None, 'best': None, 'streak': 0,
            'nodes': {}, 'plan': None, 'parked': [], 'start': time.time(), 'interventions': 0, 'stop_reason': None,
            'usage': Usage().snapshot(), 'seed_cache': {}, 'best_single': None, 'librarian_calls': 0, 'elapsed_before': 0.0,
            'families': {}, 'campaign': None, 'frontier': {}, 'queue': []}
        # one seed cache keyed 'node:seed', never cleared (ADR-0012); older states kept two that were thrown away
        sc = self.state.setdefault('seed_cache', {})
        for k_ in ('champion_seeds', 'final_seeds'):
            sc.update({k2: v for k2, v in (self.state.pop(k_, {}) or {}).items() if v is not None})
        if self.state_path.exists():   # resumed: the wall clock counts running time, not the calendar
            last = self.state.get('last_save') or self.state_path.stat().st_mtime
            self.state['elapsed_before'] = self.state.get('elapsed_before', 0.0) + max(0.0, last - self.state['start'])
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
        """Seed SD pooled over every node of this run with >= 2 fresh seeds — this run's evidence only (ADR-0020)."""
        return pooled_sigma([self.fresh_seeds(int(k)) for k in self.state['nodes']])
    def code_of(self, n): return self.j.node_path(n).read_text()

    def ctx(self, **extra):
        ch = self.champion
        d = {'run_id': self.run_id, 'generation': self.state['generation'], 'k': self.k,
             'max_generations': self.max_generations, 'max_nodes': self.max_nodes, 'nodes_used': self.state['n_next'],
             'champion': ch, 'best': self.state['best'], 'streak': self.state['streak'],
             'journal_lines': self.j.compact_lines(), 'plan': self.state.get('plan'), 'parked': self.state.get('parked', []),
             'last_generation': self.state.get('last_generation', []), 'parent_code': self.code_of(ch['n']),
             'champion_stack': self.stack_of(ch['n']),
             # ADR-0014 planning state: free-slot candidates, closed mechanisms, hard groups, the champion's inputs
             'untried': P.untried_cards(), 'proven_not_on_stack': self._proven_not_on_stack(),
             'closed_mechanisms': self._closed_mechanisms(), 'hard_groups': self._hard_groups(),
             'champion_inputs': self.inputs_of(ch['n']),
             'screened': self.state.get('screened', []),
             # ADR-0016: the campaign family of this generation and every family's status
             'campaign': self.state.get('campaign'), 'families': self.state.get('families') or {},
             'campaign_cards': self._family_cards(self.state.get('campaign')),
             # ADR-0021: the nodes worth building on and the proposals already waiting
             'frontier': self.frontier_view(), 'queue': self.queue_view()}
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
        self.state['conv_ref'] = res.metrics['primary']      # ADR-0012: the cumulative-rise reference starts at the baseline
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
               'mechanism': selection.get('mechanism'), 'target_group': selection.get('target_group'),
               'new_signal': selection.get('new_signal'), 'rebased_from': selection.get('rebased_from'),
               'screen': selection.get('screen'),
               'code_path': str(self.j.node_path(n).relative_to(self.run_dir)),
               'diff_path': diff[0] if diff else None, 'diff_lines': diff[1] if diff else None,
               'metrics': res.metrics if res else None, 'history': res.history if res else [],
               'failure_stage': None if (res and res.ok) else (res.stage if res else 'implement'),
               'error': None if (res and res.ok) else (res.error if res else None),
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
        self.state['campaign'] = self._campaign_family(g)          # ADR-0016: one family per generation, chosen in code
        if self.state['campaign']:
            fams = self.state.get('families') or {}
            self.log(f"  campaign: {self.state['campaign']} (open: {[f for f, v in fams.items() if v['status'] == 'open']}, "
                     f"closed: {[f for f, v in fams.items() if v['status'] != 'open']})")
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
        if selections and g >= C.FREE_SLOT_FROM_GENERATION and not self._free_slot_ok(selections):   # ADR-0014
            self.log('  selector: no free-slot candidate (an untried / not-yet-stacked card); asking once more')
            again, err2 = self._brain(self.brain.select, self.ctx(diagnosis=diagnosis, k=n_sel, free_slot_violation=True), n_sel,
                                      what='select', attempts=1)
            if again and self._free_slot_ok(again):
                selections = again
            else:
                self.log(f'  selector: still no free-slot candidate; proceeding with its first answer ({err2})')
                self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': 'free-slot rule violated twice by the Selector; proceeded'})
        if self.wildcard:
            if wild:
                wild['type'] = 'explore'; wild['wildcard'] = True
                selections = [wild] + selections          # first, so _diversify keeps it
                self.log(f"  wildcard: [{wild.get('target_component')}] {str(wild.get('hypothesis'))[:120]}")
            elif werr:
                self.log(f'  wildcard unavailable: {werr}')
        selections = self._apply_rules(selections)                  # ADR-0014: closed mechanisms, hard groups, information
        if self.frontier_on:
            # ADR-0021: proposals go into a persistent queue over the FRONTIER (not just the champion) and this
            # generation runs the best eligible ones; the rest wait instead of being thrown away with the generation.
            added = self.queue_add(selections, g)
            selections = self.queue_pop(self.k + 2, g)
            names = ', '.join('node_%03d%s' % (e['n'], '*' if e['champion'] else '') for e in self.frontier_view()[:C.FRONTIER_MAX])
            self.log(f"  queue: +{added} proposals, {len(selections)} popped, {len(self.state.get('queue', []))} waiting; frontier {names}")
        if not selections:
            self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'generation {g} aborted: selector failed ({err})'})
            return self._close_generation(g, [], diagnosis, snap, t_gen)
        selections = self._diversify(selections)
        selections = self._screen(selections, g)[:self.k]           # ADR-0015: measure feature candidates before building them; then k
        if self.frontier_on and len(selections) < len(self.state.get('queue', [])) + len(selections):
            for extra in [s for s in selections if s.get('popped') == g][self.k:]:
                self.state.setdefault('queue', []).append(extra)     # popped but unused: back to the queue, not lost
        if not selections:
            self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'generation {g} aborted: every candidate was screened out'})
            return self._close_generation(g, [], diagnosis, snap, t_gen)
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
            if nd['sel'].get('parent_n') != nd['parent']:          # the Critic rebased the candidate onto another node (ADR-0014)
                nd['parent'] = nd['sel']['parent_n']; nd['parent_code'] = self.code_of(nd['parent'])
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
                rec['error'] = rec['error'] or nd['error'] or 'no runnable script produced'
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
        if self.state.get('conv_ref') is None:                      # older states: the reference is the baseline's fresh-seed mean
            self.state['conv_ref'] = self.node_mean(0)
        conv = R.Convergence(self.state['streak'], self.state['conv_ref'])
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
        self._frontier_book(g, results)                              # ADR-0021: children and barren counts per frontier node
        self.frontier_update(g)                                      # … then add new near-misses and retire the barren
        self._campaign_update(g, results)                            # ADR-0016: a family closes after flat generations
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
                       'campaign': self.state.get('campaign'), 'families': json.loads(json.dumps(self.state.get('families') or {})),
                       'frontier': self.frontier_view(), 'queue_pending': len(self.state.get('queue', [])),
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
        if improved or self.state.get('librarian_calls', 0) >= self.librarian_max:
            return
        if len(P.untried_cards()) >= self.k and self.state.get('streak', 0) < 2:   # ADR-0014: two flat generations also call it
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
        """Portfolio diversity: distinct target_components — except that the free-slot candidate (ADR-0014, tagged by
        _free_slot_ok) outranks the wildcard on a collision (an information-adding wildcard and an untried features/history
        card share a component more often than not, and the free slot is the rule that keeps untried cards from being
        locked out), and that inside a campaign (ADR-0016) the family's candidates are distinct by MECHANISM slug, since
        they share a component by design."""
        fam = self.state.get('campaign'); seen, out = {}, []
        for s in selections:
            tc = s.get('target_component')
            in_campaign = bool(fam) and not s.get('wildcard') and s.get('type') not in ('merge', 'retest')
            if in_campaign and _slug(s.get('mechanism')):
                key = ('mechanism', _slug(s.get('mechanism')))
            elif in_campaign:      # no mechanism slug: never collide it with the whole family on the component key
                key = ('card', s.get('card')) if s.get('card') else ('hypothesis', (_slug(s.get('hypothesis')) or '')[:40])
                self.log(f"  campaign candidate without a mechanism slug ({s.get('card')!r}); keyed by {key[0]}")
            else:
                key = ('component', tc)
            if key in seen and s.get('type') != 'merge':
                prev = out[seen[key]]
                if s.get('free_slot') and prev.get('wildcard'):
                    self.log(f"  free-slot candidate {s.get('card')!r} outranks the wildcard on {tc!r}; dropping the wildcard")
                    out[seen[key]] = s; continue
                self.log(f'  dropping duplicate {key[0]} {key[1]!r}: {s.get("hypothesis", "")[:60]}')
                continue
            seen[key] = len(out); out.append(s)
        return out

    def _node_ref(self, x):
        """Parse a node reference the roles may write: 12, "12", "node_012", "champion"; None if unknown."""
        if isinstance(x, bool) or x is None:
            return None
        if isinstance(x, int):
            n = x
        else:
            m = re.search(r'(\d+)', str(x))
            if str(x).strip().lower() == 'champion' or not m:
                return None
            n = int(m.group(1))
        return n if str(n) in self.state['nodes'] and self.node(n).get('metrics') else None

    # ---------------- ADR-0014 slot rules (code, not prose) ----------------
    INPUT_COLUMNS = ('user_id', 'video_id', 'date', 'hourmin', 'time_ms', 'tab', 'duration_ms', 'is_rand', 'author_id', 'video_type',
                     'upload_dt', 'upload_type', 'visible_status', 'video_duration', 'server_width', 'server_height', 'music_id',
                     'music_type', 'tag', 'user_active_degree', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
                     'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days', 'onehot_feat', 'is_click', 'is_like',
                     'is_follow', 'is_comment', 'is_forward', 'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time',
                     'is_profile_enter', 'long_view')

    def inputs_of(self, n):
        """Columns and side-table fields a node's script references — the Explorer must add a signal outside this set."""
        try:
            code = self.code_of(n)
        except Exception:      # noqa: BLE001
            return []
        return [c for c in self.INPUT_COLUMNS if re.search(r'\b' + re.escape(c) + r'(\d+|_range)?\b', code)]

    def _proven_not_on_stack(self):
        parts = _stack_key(self.stack_of(self.state['champion']))
        return [c for c in P.proven_cards() if c not in parts]

    def _rejected_deepens(self):
        return [r for r in self.state['nodes'].values() if r.get('action') == 'deepen' and r.get('metrics') and not r.get('accepted')]

    def _closed_mechanisms(self):
        """mechanism slug -> rejected deepen nodes; a rejected mechanism is not deepened again this run."""
        out = {}
        for r in self._rejected_deepens():
            m = _slug(r.get('mechanism'))
            if m:
                out.setdefault(m, []).append(r['n'])
        return out

    def _hard_groups(self):
        """breakdown group -> rejected deepen nodes, for groups with >= HARD_GROUP_REJECTS of them."""
        cnt, names = {}, {}
        for r in self._rejected_deepens():
            g_ = str(r.get('target_group') or 'all').strip(); k_ = _gkey(g_)
            if g_.lower() not in ('all', '', 'none') and k_:
                names.setdefault(k_, g_); cnt.setdefault(k_, []).append(r['n'])   # 'dur>180s' and 'dur > 180 s' are one group
        return {names[k_]: ns for k_, ns in cnt.items() if len(ns) >= C.HARD_GROUP_REJECTS}

    def _free_slot_ok(self, selections):
        """From FREE_SLOT_FROM_GENERATION on, one candidate must take an untried card or a proven card not on the stack."""
        eligible = set(P.untried_cards()) | set(self._proven_not_on_stack())
        fam = self.state.get('campaign')
        if fam:                                   # ADR-0016: the free slot stays inside the campaign family
            idx = P.card_index(); eligible = {c for c in eligible if idx.get(c, {}).get('family') == fam}
        if not eligible:
            return True
        for s in selections:
            if s.get('card') in eligible and s.get('type') in ('improve', 'retest', 'explore'):
                s['free_slot'] = True          # _diversify lets it outrank the wildcard on a component collision
                return True
        return False

    def _apply_rules(self, selections):
        """Drop candidates the rules forbid: a deepen of a closed mechanism or of a hard group, a wildcard without new_signal."""
        closed, hard, out = self._closed_mechanisms(), self._hard_groups(), []
        fam = self.state.get('campaign'); idx = P.card_index() if fam else {}
        for s in selections:
            h = str(s.get('hypothesis'))[:80]
            if fam and not s.get('wildcard') and s.get('type') not in ('merge', 'retest'):
                cf = P.family_of(s.get('card'), idx)
                if cf is not None and cf != fam:
                    self.log(f"  dropping {s.get('card')!r}: outside the campaign family {fam!r} (it is {cf!r})"); continue
            if s.get('type') == 'deepen':
                m = _slug(s.get('mechanism'))
                if m and m in closed:
                    self.log(f"  dropping deepen of closed mechanism {m!r} (rejected in node(s) {closed[m]}): {h}"); continue
                g_ = str(s.get('target_group') or 'all').strip(); hit = next((k_ for k_ in hard if _gkey(k_) == _gkey(g_)), None)
                if hit is not None:
                    self.log(f"  dropping deepen on hard group {g_!r} (rejected deepens {hard[hit]}): {h}"); continue
            if s.get('wildcard'):
                sig = str(s.get('new_signal') or '').strip()
                if len(sig) < 8 or sig.lower() in ('none', 'n/a', 'null'):
                    self.log(f"  dropping wildcard without a new_signal (capacity-only proposals are closed, ADR-0014): {h}"); continue
            out.append(s)
        return out

    # ---------------- ADR-0015 feature screen ----------------
    def _screenable(self, s):
        if s.get('type') in ('merge', 'retest'):
            return False
        # a wildcard is probed only when its signal is a column of the row (features / encoding / history); a model-family
        # wildcard (attention over the history, a tree model) has no column to compute, and a null probe would kill it
        return s.get('target_component') in C.SCREEN_COMPONENTS and (not s.get('wildcard') or bool(s.get('new_signal')))

    @staticmethod
    def _family_of(s):
        """The card's family (front matter, also for '<card> — <variant>' names), else the candidate's component."""
        return P.family_of(s.get('card')) or s.get('target_component')

    def _screen(self, selections, g):
        """Probe every feature candidate (a Probe-role script computes the signal on the label-stripped valid split), measure it
        against the champion's predictions, and drop the slot when best_gain < SCREEN_MIN_GAIN. A failed probe never blocks."""
        if not self.screen or g < C.SCREEN_FROM_GENERATION:
            return selections
        todo = [s for s in selections if self._screenable(s)]
        champ_pred = self.j.out_dir(self.state['champion']) / 'predictions.csv'
        if not todo or not champ_pred.exists():
            return selections
        threads = max(1, (os.cpu_count() or 2) // max(1, len(todo)))
        def probe(s):
            code, err = self._brain(self.brain.probe, self.ctx(), s, what='probe', attempts=1)
            if not code:
                return s, None, err or 'the Probe declined: not a per-row column signal'
            out = self.run_dir / 'screens' / f"g{g:02d}_{(_slug(s.get('card')) or 'candidate')[:48]}"
            out.mkdir(parents=True, exist_ok=True); (out / 'probe.py').write_text(code)
            return s, S.run_probe(out / 'probe.py', out, champ_pred, threads=threads), None
        if self.parallel and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=len(todo)) as ex:
                outcomes = list(ex.map(probe, todo))
        else:
            outcomes = [probe(s) for s in todo]
        dropped = set()
        for s, res, err in outcomes:
            fam = self._family_of(s); card = s.get('card')
            if res is None or not res.ok:
                why = err or (res.error if res else 'unknown')
                self.log(f"  screen: probe of {card!r} failed ({why}); candidate proceeds unscreened")
                self.j.append({'n': None, 'generation': g, 'action': 'event', 'note': f'screen of {card!r} failed: {why}; candidate proceeded'})
                continue
            kept = S.passes(res); s['screen'] = res.summary()
            rec = {'n': None, 'generation': g, 'action': 'screen', 'kept': kept, 'card': card, 'family': fam,
                   'target_component': s.get('target_component'), 'hypothesis': s.get('hypothesis'), 'new_signal': s.get('new_signal'),
                   'wildcard': bool(s.get('wildcard')), 'best_gain': res.best_gain, 'best_column': res.best_column, 'stack_gain': res.stack_gain,
                   'columns': res.columns, 'duration_s': round(res.duration_s, 1), 'text': res.text()}
            self.j.append(rec)
            self.state.setdefault('screened', []).append({'generation': g, 'card': card, 'family': fam, 'best_gain': res.best_gain, 'kept': kept})
            self.log(f"  screen {'kept' if kept else 'DROPPED'} {card!r} [{fam}]: {res.text()[:160]}")
            if not kept:
                dropped.add(id(s))
        return [s for s in selections if id(s) not in dropped]
    # ---------------- ADR-0016 family campaigns ----------------
    def _families_init(self):
        """state['families']: every card family, open until it has been campaigned flat or has nothing left to measure."""
        fams = self.state.setdefault('families', {})
        for cid, f in P.card_index().items():
            fams.setdefault(f['family'], {'status': 'open', 'generations': [], 'nodes': [], 'best_gain': None, 'flat_streak': 0, 'evidence': ''})
        return fams

    def _family_cards(self, fam):
        if not fam:
            return []
        return sorted(f"{c} [{v['status'].split(' ')[0]}]" for c, v in P.card_index().items() if v['family'] == fam)

    def _ledger(self):
        """The family ledger built from the cards at the start of each generation (ADR-0018; never a stale file)."""
        g = self.state.get('generation')
        if getattr(self, '_ledger_cache', (None, None))[0] != g:
            self._ledger_cache = (g, P.ledger())
        return self._ledger_cache[1]

    def _family_score(self, fam):
        """Ordering key of an open family (ADR-0016 + ADR-0018): (not a last family, has a kept screen gain, value) where
        value = a kept screen gain if any, else the best card_value() among the family's cards still measurable on this
        stack — the record for measured cards, the discounted promise capped by an ORACLE bound for unmeasured ones;
        None when nothing is left to measure. Falls back to the cards' own expected_delta without a ledger."""
        idx = P.card_index(); stack = _stack_key(self.stack_of(self.state['champion']))
        kept = [s['best_gain'] for s in (self.state.get('screened') or [])
                if s.get('family') == fam and s.get('best_gain') is not None and s.get('kept')]   # dropped screens are evidence AGAINST
        last = 0 if fam in C.CAMPAIGN_LAST_FAMILIES else 1
        if kept:
            return (last, 1, max(kept))
        def measurable(c, v):
            if c in stack:
                return False
            if v['status'].startswith('dead_under'):
                return stack not in {_stack_key(s) for s in _dead_stacks(v['status'])}
            return True
        led = self._ledger(); rows = (led or {}).get('cards', {})
        if led is not None:                   # with a ledger, evidence orders every family (no forced-last family)
            last = 1
            fam_screens = [s.get('best_gain') for s in (led.get('families', {}).get(fam, {}).get('screen_gains') or [])
                           if s.get('kept') and s.get('best_gain') is not None]
            if fam_screens:
                return (last, 1, max(fam_screens))
        vals = []
        for c, v in idx.items():
            if v['family'] != fam or not measurable(c, v):
                continue
            if c in rows:
                vals.append(P.card_value(rows[c]))
            elif v['expected_hi'] is not None:
                vals.append(v['expected_hi'])
        return (last, 0, max(vals)) if vals else None

    def _campaign_family(self, g):
        """The family this generation's Selector slots belong to: the current one while it is open, else the best-scoring
        open family; None before CAMPAIGNS_FROM_GENERATION, with --no-campaigns, or when every family is closed."""
        if not self.campaigns or g < C.CAMPAIGNS_FROM_GENERATION:
            return None
        fams = self._families_init(); cur = self.state.get('campaign')
        if cur and fams.get(cur, {}).get('status') == 'open':
            if self._family_score(cur) is not None:
                return cur
            fams[cur]['status'] = 'exhausted'; fams[cur]['evidence'] = (fams[cur]['evidence'] + ' nothing left to measure on this stack').strip()
            self.log(f"  campaign {cur!r} exhausted: nothing left to measure on this stack")
        best = None
        for fam, v in fams.items():
            if v['status'] != 'open':
                continue
            key = self._family_score(fam)
            if key is None:
                v['status'] = 'exhausted'; v['evidence'] = (v['evidence'] + ' nothing left to measure on this stack').strip(); continue
            if best is None or key > best[0]:
                best = (key, fam)
        return best[1] if best else None

    def _campaign_update(self, g, results):
        """Book this generation's nodes to the campaign family; close it after CAMPAIGN_FLAT_GENERATIONS flat generations."""
        fam = self.state.get('campaign')
        if not fam or self._families_init().get(fam, {}).get('status') != 'open':
            return
        v = self._families_init()[fam]; idx = P.card_index()
        mine = [r for r in results if r.get('metrics') and r.get('action') != 'merge' and P.family_of(r.get('method'), idx) == fam]
        v['generations'].append(g); v['nodes'] += [r['n'] for r in mine]
        gains = [(r.get('seed_confirmation') or {}).get('delta_mean', r.get('realized_delta')) for r in mine]
        gains = [x for x in gains if x is not None]
        if gains:
            v['best_gain'] = max(gains + ([v['best_gain']] if v['best_gain'] is not None else []))
        dropped_here = any(s.get('family') == fam and not s.get('kept') and s.get('generation') == g for s in (self.state.get('screened') or []))
        if any(r.get('accepted') for r in mine):
            v['flat_streak'] = 0
        elif not mine and not dropped_here:
            self.log(f"  campaign {fam!r}: no node this generation (slots went to merges/retests); flat streak unchanged")
        else:
            v['flat_streak'] += 1
            if v['flat_streak'] >= C.CAMPAIGN_FLAT_GENERATIONS:
                v['status'] = 'closed'
                v['evidence'] = f"closed at generation {g}: {v['flat_streak']} campaign generations without an accepted node (nodes {v['nodes']}, best gain {v['best_gain']})"
                self.log(f"  campaign {fam!r} closed: {v['flat_streak']} flat generations (nodes {v['nodes']})")

    # ---------------- ADR-0021: the frontier and the proposal queue ----------------
    def _se(self):
        """One standard error of a three-seed mean, from this run's pooled seed SD (its own evidence, ADR-0020)."""
        sigma, _ = self._sigma()
        return (sigma or C.SEED_SD) / (C.CONFIRM_SEEDS ** 0.5)

    def frontier_update(self, g):
        """Recompute the frontier: the champion, every accepted node, and every node whose fresh-seed mean is within
        one standard error of the champion's — the near-misses three runs threw away. A node retires after
        FRONTIER_RETIRE_GENERATIONS generations without an accepted descendant."""
        if not self.frontier_on:
            return {}
        fr = self.state.setdefault('frontier', {})
        ch = self.state['champion']; ch_mean = self.champion_mean() or 0.0
        margin = C.FRONTIER_MARGIN_SE * self._se()
        for v in self.state['nodes'].values():
            if not v.get('metrics') or v['n'] == ch:
                continue
            m = self.node_mean(v['n'])
            if m is None or m + margin < ch_mean:
                continue
            fr.setdefault(str(v['n']), {'n': v['n'], 'added': g, 'barren': 0, 'children': 0, 'accepted': bool(v.get('accepted'))})
        fr[str(ch)] = {**fr.get(str(ch), {'added': g, 'barren': 0, 'children': 0}), 'n': ch, 'accepted': True}
        for key, e in list(fr.items()):
            e['mean'] = self.node_mean(e['n']); e['n_seeds'] = len(self.fresh_seeds(e['n']))
            if e['n'] == ch:
                e['barren'] = 0; continue
            if e.get('added') != g and e.get('accepted_child_gen') != g:
                pass
            if e['barren'] >= C.FRONTIER_RETIRE_GENERATIONS:
                self.log(f"  frontier: retiring node_{e['n']:03d} ({e['barren']} generations without an accepted child)")
                fr.pop(key); self.state['queue'] = [q for q in self.state.get('queue', []) if q.get('parent') != e['n']]
        if len(fr) > C.FRONTIER_MAX:      # keep the champion and the best means
            keep = {str(ch)} | {k for k, _ in sorted(((k, v) for k, v in fr.items() if v['n'] != ch),
                                                     key=lambda kv: -(kv[1].get('mean') or 0))[:C.FRONTIER_MAX - 1]}
            for key in [k for k in fr if k not in keep]:
                fr.pop(key)
        return fr

    def frontier_view(self):
        """The frontier as the planning roles see it: node, fresh-seed mean, accepted, children, barren generations."""
        if not self.frontier_on:
            return []
        out = []
        for e in (self.state.get('frontier') or {}).values():
            v = self.state['nodes'].get(str(e['n'])) or {}
            out.append({'n': e['n'], 'mean': round(e.get('mean') or self.node_mean(e['n']) or 0.0, 5),
                        'accepted': bool(v.get('accepted')), 'method': v.get('method'), 'stack': self.stack_of(e['n']),
                        'children': e.get('children', 0), 'barren_generations': e.get('barren', 0),
                        'champion': e['n'] == self.state['champion']})
        return sorted(out, key=lambda r: -r['mean'])

    def queue_view(self, limit=12):
        """The pending proposals, best first — what the planners should extend rather than repeat."""
        q = sorted(self.state.get('queue', []), key=lambda x: -x.get('score', 0))
        return [{k: x.get(k) for k in ('parent', 'card', 'mechanism', 'target_component', 'hypothesis', 'expected_delta', 'score', 'added')}
                for x in q[:limit]]

    def _queue_score(self, item):
        """A proposal's priority: how far its parent has come, plus what the idea is worth on the record (ADR-0018)."""
        parent_mean = self.node_mean(item.get('parent')) or self.node_mean(0) or 0.0
        base = self.node_mean(0) or 0.0
        led = self._ledger() or {}
        row = (led.get('cards') or {}).get(str(item.get('card') or '').split(' — ')[0].strip())
        value = P.card_value(row) if row else min(float(item.get('expected_delta') or 0.0), 0.002)
        wild = 0.0005 if item.get('wildcard') else 0.0
        return round((parent_mean - base) + value + wild, 6)

    def queue_add(self, selections, g):
        """Queue this generation's proposals (parents resolved, duplicates of a pending idea dropped) — ADR-0021: an
        idea outlives the generation that proposed it, so a full slate no longer throws away the runner-up."""
        q = self.state.setdefault('queue', [])
        pending = {(x.get('parent'), _slug(x.get('mechanism')) or str(x.get('card'))) for x in q}
        added = 0
        for s in selections:
            item = dict(s); item['parent'], item['merge_parents'] = self._resolve_parents(item)
            key = (item['parent'], _slug(item.get('mechanism')) or str(item.get('card')))
            if key in pending:
                continue
            item['added'] = g; item['score'] = self._queue_score(item)
            q.append(item); pending.add(key); added += 1
        q.sort(key=lambda x: -x.get('score', 0))
        if len(q) > C.QUEUE_MAX:
            del q[C.QUEUE_MAX:]
        return added

    def queue_pop(self, k, g):
        """Take the k best eligible proposals: parent still on the frontier, mechanism not closed, group not hard,
        inside the campaign family, not stale. Popped items leave the queue; the rest wait for the next generation."""
        q = self.state.setdefault('queue', [])
        fr = {e['n'] for e in (self.state.get('frontier') or {}).values()} or {self.state['champion']}
        closed, hard = self._closed_mechanisms(), self._hard_groups()
        fam = self.state.get('campaign'); idx = P.card_index() if fam else {}
        out, keep = [], []
        for item in sorted(q, key=lambda x: -x.get('score', 0)):
            card = str(item.get('card'))[:40]
            if g - item.get('added', g) >= C.QUEUE_STALE_GENERATIONS:
                self.log(f"  queue: dropping stale proposal {card!r} (waited {g - item.get('added', g)} generations)")
                continue
            if item.get('type') != 'merge' and item.get('parent') not in fr:
                self.log(f"  queue: dropping {card!r} — its parent node_{item.get('parent'):03d} left the frontier")
                continue
            m = _slug(item.get('mechanism'))
            if m and m in closed:
                self.log(f"  queue: dropping {card!r} — mechanism {m!r} is closed this run"); continue
            if len(out) >= k:
                keep.append(item); continue
            g_ = str(item.get('target_group') or 'all').strip()
            if item.get('type') == 'deepen' and any(_gkey(h) == _gkey(g_) for h in hard):
                keep.append(item); continue                       # a hard group may be re-opened by a new champion
            if fam and not item.get('wildcard') and item.get('type') not in ('merge', 'retest'):
                cf = P.family_of(item.get('card'), idx)
                if cf is not None and cf != fam:
                    keep.append(item); continue                   # out of this generation's campaign; waits for its own
            item['popped'] = g; out.append(item)
        self.state['queue'] = keep
        return out

    def _frontier_book(self, g, results):
        """Book this generation's children to their parents: an accepted child clears a frontier node's barren count,
        a generation of only rejected children increments it (retirement happens in frontier_update)."""
        fr = self.state.get('frontier') or {}
        by_parent = {}
        for r in results:
            if r.get('parent') is not None and r.get('metrics'):
                by_parent.setdefault(r['parent'], []).append(r)
        for pn, rs in by_parent.items():
            e = fr.get(str(pn))
            if not e:
                continue
            e['children'] = e.get('children', 0) + len(rs)
            if any(x.get('accepted') for x in rs):
                e['barren'] = 0; e['accepted_child_gen'] = g
            else:
                e['barren'] = e.get('barren', 0) + 1

    def _resolve_parents(self, sel):
        """The parent a candidate is built on. A deepen/retest of a specific node must branch from THAT node — live_05's
        node_014 named node_012 as a string, was silently built on the champion, and the Critic rejected it three times
        for being against the wrong parent. The candidate's `parent` field is rewritten to the resolved node so the
        Implementer, the Critic and the journal agree."""
        ch = self.state['champion']; mp = [self._node_ref(x) for x in (sel.get('merge_parents') or [])]
        if sel.get('type') == 'merge' and len(mp) >= 2 and mp[0] is not None and mp[1] is not None:
            sel['merge_parents'] = [mp[0], mp[1]]
            return mp[0], [mp[0], mp[1]]
        p = self._node_ref(sel.get('parent', 'champion'))
        if p is None and sel.get('parent') not in (None, 'champion'):
            self.log(f"  parent {sel.get('parent')!r} not resolvable; building on the champion node_{ch:03d}")
        sel['parent'] = p if p is not None else ch
        return sel['parent'], None

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
            nb = self._node_ref(verdict.get('rebase_to')) if verdict.get('rebase_to') not in (None, '', 'null') else None
            if nb is not None and nb != sel.get('parent_n') and str(nb) in self.state['nodes'] and self.j.node_path(nb).exists():
                # ADR-0014: the hypothesis edits a specific node but the candidate was built on another script — hand the
                # Implementer the right parent instead of letting it re-implement that node's mechanism (live_06 node_020)
                self.log(f"  critic: rebasing the candidate onto node_{nb:03d} (was node_{sel.get('parent_n', self.state['champion']):03d})")
                sel['rebased_from'] = sel.get('parent_n'); sel['parent'] = sel['parent_n'] = nb; parent_code = self.code_of(nb)
                sel['critic_instructions'] = (f"The parent script is now node_{nb:03d} (the harness rebased the candidate as the Critic asked): "
                                              f"edit THAT script for the hypothesis only. " + (verdict.get('instructions') or ''))
                continue
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
        the difference of their means with the seed SD pooled over this run's own seed runs (no outside prior, ADR-0020),
        acceptance iff the gain
        is >= MIN_EFFECT and z >= Z_CRIT; a borderline z — or an accepting one on a sigma with < MIN_SIGMA_DF df — gets
        two more seeds first. Seed 0 is the selected screen and is
        reported but not counted. Measured reason: the best of k single-seed branches is biased upward — +0.0022 on one
        seed was +0.0017 over three; a 3-vs-3 t-test at 2.5 passed 3-6 % of null candidates."""
        base = [self.seed + i for i in range(1, C.CONFIRM_SEEDS + 1)]
        ch = self.state['champion']
        self._ensure_seeds(n, base); self._ensure_seeds(ch, base)
        v_node, v_ch = self.fresh_seeds(n), self.fresh_seeds(ch)
        if not v_node or not v_ch:
            return False, {'error': 'confirmation seeds failed', 'node_seeds': v_node, 'champion_seeds': v_ch}
        def sigma_for(v):
            """Pooled sigma, unless the node's own seeds are clearly more unstable (sample SD > 2x pooled, p ~ 5 % under
            homogeneity); the node's own SD also stands in when this run has no pooled estimate yet (ADR-0020)."""
            s_pool, df = self._sigma(); s_own = statistics.stdev(v) if len(v) > 1 else 0.0
            if s_pool is None:
                return (s_own, 0, True)
            return (s_own, df, True) if s_own > 2 * s_pool else (s_pool, df, False)
        sigma, df, own = sigma_for(v_node)
        m_node, m_ch, diff, se, z, accepted = confirm_stats(v_node, v_ch, sigma)
        adaptive = False
        thin = accepted and df < C.MIN_SIGMA_DF
        if needs_more_seeds(z, accepted, df, len(v_node)):
            adaptive = True
            self._ensure_seeds(n, [self.seed + i for i in range(C.CONFIRM_SEEDS + 1, C.MAX_CONFIRM_SEEDS + 1)])
            v_node = self.fresh_seeds(n); sigma, df, own = sigma_for(v_node)
            m_node, m_ch, diff, se, z, accepted = confirm_stats(v_node, v_ch, sigma)
        self.log(f"node_{n:03d}: seed confirmation — fresh seeds {[round(v, 5) for v in v_node]} mean {m_node:.5f} vs champion "
                 f"{[round(v, 5) for v in v_ch]} mean {m_ch:.5f}: diff {diff:+.5f}, z {z:.1f} (sigma {sigma:.5f}, {df} df"
                 f"{', node-own SD' if own else ''}{', adaptive' if adaptive else ''}) -> {'ACCEPTED' if accepted else 'rejected'}")
        return accepted, {'node_seed0': self.node(n)['metrics']['primary'], 'node_seeds': v_node, 'champion_seeds': v_ch,
                          'delta_mean': round(diff, 5), 'se': round(se, 6), 'z': round(z, 2), 'sigma_pooled': round(sigma, 6),
                          'sigma_df': df, 'sigma_from_node_only': own, 'adaptive': adaptive, 'thin_df': thin,
                          'rule': f'fresh-seed mean gain >= {C.MIN_EFFECT} and z >= {C.Z_CRIT} with the seed SD pooled over '
                                  f'this run only (ADR-0020: no prior from other runs)'}

    def _result_view(self, rec):
        m = rec.get('metrics') or {}
        view = {k: rec.get(k) for k in ('n', 'parent', 'merge_parents', 'action', 'method', 'target_component', 'hypothesis',
                                        'change_summary', 'expected_delta', 'realized_delta', 'accepted',
                                        'seed_confirmation', 'failure_stage', 'error', 'recovery', 'duration_s')}
        view['metrics'] = {k: v for k, v in m.items() if k not in ('by_group', 'by_pair')}
        view['curve'] = [round(h.get('val_primary', 0), 4) for h in rec.get('history', [])][:40]
        cg = (self.champion.get('metrics') or {}).get('by_group') or {}
        if m.get('by_group') and cg:   # where the node moved relative to the champion, per tab and duration band
            view['by_group_delta'] = {g: round(m['by_group'][g]['primary'] - cg[g]['primary'], 4) for g in m['by_group'] if g in cg}
        cp = (self.champion.get('metrics') or {}).get('by_pair') or {}
        if m.get('by_pair') and cp:   # misordered-pair fraction per pair type vs the champion; negative = fewer misordered pairs
            view['by_pair_delta'] = {t: round(m['by_pair'][t]['err'] - cp[t]['err'], 4) for t in m['by_pair']
                                     if isinstance(m['by_pair'].get(t), dict) and isinstance(cp.get(t), dict)
                                     and m['by_pair'][t].get('err') is not None and cp[t].get('err') is not None}
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
        """Final selection for submission (AIRA-style robust choice, ADR-0012 and its amendment after live_07).
        Candidates: the top_k nodes by validation primary plus the confirmed champion, re-ranked by the FRESH-seed mean
        (cached; seeds are run only for a node with fewer than two) so a lucky single seed cannot be the submission.
        'strict' (default): only ACCEPTED nodes may be designated — the run never submits a node it rejected; the best
        unaccepted candidate is reported as best_unaccepted for the record. 'adaptive': an unaccepted leader is given
        MAX_CONFIRM_SEEDS fresh seeds and is designated only if its gain over the champion's fresh-seed mean is
        >= MIN_EFFECT at z >= Z_BORDER (the search's own borderline test), else it is excluded; every exclusion is journaled.
        Tie-break (both modes): within one standard error of the best mean, an accepted node is preferred (live_04's
        designation had picked a rejected near-no-op variant of the champion on a 0.0000045 gap)."""
        nodes = [v for v in self.state['nodes'].values() if v.get('metrics')]
        # the pool: every ACCEPTED node (few, each with fresh seeds) plus the champion — so an accepted node with a modest
        # seed-0 but the best fresh-seed mean can be designated; in adaptive mode also the top_k unaccepted by seed-0 valid
        pool = {v['n']: v for v in nodes if v.get('accepted')}
        if self.state.get('champion') is not None:
            pool[self.state['champion']] = self.node(self.state['champion'])
        unacc_top = [v for v in sorted(nodes, key=lambda v: -v['metrics']['primary'])[:top_k] if v['n'] not in pool]
        if self.designation == 'adaptive':
            for v in unacc_top:
                pool[v['n']] = v
        def entry(v, reseed):
            if reseed and self.final_reseed and len(self.fresh_seeds(v['n'])) < 2:
                self._ensure_seeds(v['n'], [self.seed + i for i in range(1, C.CONFIRM_SEEDS + 1)])
            vals = self.fresh_seeds(v['n']) or [v['metrics']['primary']]
            return {'n': v['n'], 'valid_primary': v['metrics']['primary'], 'fresh_seeds': vals, 'accepted': bool(v.get('accepted')),
                    'mean': statistics.mean(vals), 'std': statistics.stdev(vals) if len(vals) > 1 else None}
        ranking = [entry(v, True) for v in sorted(pool.values(), key=lambda v: -v['metrics']['primary'])]
        if self.designation == 'strict':      # reported for the record from cached seeds only — never re-run, never eligible
            ranking += [entry(v, False) for v in unacc_top]
        ranking.sort(key=lambda r: (-r['mean'], -r['valid_primary']))
        events = []
        for r in ranking:
            r['n_seeds'] = len(self.fresh_seeds(r['n']))
        unacc = [r for r in ranking if not r['accepted'] and r['n_seeds'] >= 2]      # a single seed is not a fresh-seed mean
        self.state['best_unaccepted'] = ({'n': unacc[0]['n'], 'mean': round(unacc[0]['mean'], 5), 'valid_primary': unacc[0]['valid_primary'],
                                          'n_seeds': unacc[0]['n_seeds']} if unacc else None)
        if self.designation == 'strict':
            for r in unacc:
                r['excluded'] = 'not accepted (strict designation: the run never submits a node it rejected)'
            eligible = [r for r in ranking if r['accepted']]
            if unacc and (not eligible or unacc[0]['mean'] > eligible[0]['mean']):
                events.append(f"designation (strict): node_{unacc[0]['n']:03d} leads on fresh-seed mean ({unacc[0]['mean']:.5f}) but was not "
                              f"accepted; excluded — accepted lineage only")
        else:
            eligible, ch = [], self.state.get('champion')
            for r in ranking:
                if r['accepted']:
                    eligible.append(r); continue
                if ch is None or r['n'] == ch:
                    continue
                if self.final_reseed:
                    self._ensure_seeds(r['n'], [self.seed + i for i in range(1, C.MAX_CONFIRM_SEEDS + 1)])
                    r['fresh_seeds'] = self.fresh_seeds(r['n']); r['mean'] = statistics.mean(r['fresh_seeds'])
                    r['std'] = statistics.stdev(r['fresh_seeds']) if len(r['fresh_seeds']) > 1 else None
                v_ch = self.fresh_seeds(ch)
                if not v_ch:
                    r['excluded'] = 'champion has no fresh seeds to compare against'; continue
                sigma, _ = self._sigma()
                _, _, diff, se, z, _ = confirm_stats(r['fresh_seeds'], v_ch, sigma)
                r['adaptive'] = {'seeds': len(r['fresh_seeds']), 'delta_mean': round(diff, 5), 'se': round(se, 6), 'z': round(z, 2)}
                if diff >= C.MIN_EFFECT and z >= C.Z_BORDER:
                    eligible.append(r)
                    events.append(f"designation (adaptive): node_{r['n']:03d} not accepted as champion but with {len(r['fresh_seeds'])} fresh seeds "
                                  f"beats the champion by {diff:+.5f} at z {z:.2f} (>= {C.Z_BORDER}); eligible")
                else:
                    r['excluded'] = f"adaptive test failed: {diff:+.5f} at z {z:.2f} with {len(r['fresh_seeds'])} seeds"
                    events.append(f"designation (adaptive): node_{r['n']:03d} excluded — {r['excluded']}")
        eligible.sort(key=lambda r: (-r['mean'], -r['valid_primary']))
        if eligible:
            sigma, _ = self._sigma(); best = eligible[0]['mean']
            se = sigma * (1.0 / max(1, len(eligible[0]['fresh_seeds']))) ** 0.5
            tied = [r for r in eligible if best - r['mean'] <= se]
            preferred = [r for r in tied if r['accepted']]
            if preferred and not eligible[0]['accepted']:
                eligible.remove(preferred[0]); eligible.insert(0, preferred[0])
                eligible[0]['tie_break'] = f"within one SE ({se:.5f}) of the best mean; accepted lineage preferred"
        self.state['designated'] = eligible[0]['n'] if eligible else self.state['champion']
        self.state['designation_events'] = events
        for e in events:
            self.log('  ' + e); self.j.append({'n': None, 'generation': self.state['generation'], 'action': 'event', 'note': e})
        ordered = eligible + [r for r in ranking if r not in eligible]
        return ordered

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
        usage = self.brain.usage.snapshot()
        if not usage.get('calls'):                                   # finished offline (resume): keep the run's recorded usage
            usage = self.state.get('usage', usage)
        summary = {'run_id': self.run_id, 'stop_reason': self.state['stop_reason'], 'generations': self.state['generation'],
                   'nodes': self.state['n_next'], 'champion': ch['n'], 'champion_metrics': ch['metrics'],
                   'baseline_valid_primary': self.node(0)['metrics']['primary'],
                   'delta_vs_baseline_valid': round(ch['metrics']['primary'] - self.node(0)['metrics']['primary'], 5),
                   'top3_valid': [{'n': v['n'], 'primary': v['metrics']['primary']} for v in top],
                   'designated': self.state.get('designated'), 'final_ranking': ranking,
                   'usage': usage, 'wall_clock_s': round(self.elapsed(), 1),
                   'champion_seed_mean': round(self.champion_mean(), 5), 'best_single_seed': self.state.get('best_single'),
                   'convergence_rule': f'ADR-0012 (revised): {C.N_CONVERGE} generations without a seed-confirmed champion change',
                   'official_rule': self.state.get('official_rule'), 'official_rule_submission': self._official_submission(),
                   'convergence_switch': self.convergence, 'campaigns': self.campaigns, 'families': self.state.get('families'),
                   'designation_rule': self.designation, 'designation_events': self.state.get('designation_events', []),
                   'frontier': self.frontier_view(), 'queue_pending': len(self.state.get('queue', [])), 'frontier_on': self.frontier_on,
                   'best_unaccepted': self.state.get('best_unaccepted'),
                   'tokens': {'in_total': usage.get('tokens_in'), 'in_cached': usage.get('cache_read'),
                              'in_uncached': (usage.get('tokens_in') or 0) - (usage.get('cache_read') or 0), 'out': usage.get('tokens_out')},
                   'interventions': self.state['interventions'], 'k': self.k_first, 'k_later': self.k_later, 'eps': C.EPS, 'n_converge': C.N_CONVERGE,
                   'iteration_unit': self.iteration_unit, 'iterations_used': self.state['n_next'] if self.iteration_unit == 'node' else self.state['generation']}
        (self.run_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        (self.run_dir / 'journal.md').write_text(self.j.render_md(summary))
        self.save(); self.log(f'run finished: {summary["stop_reason"]} — champion node_{ch["n"]:03d} {ch["metrics"]["primary"]:.4f}')
        return summary
