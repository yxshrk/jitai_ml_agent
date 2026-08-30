"""LLM roles behind one interface (`Brain`), a scripted `FakeBrain` for offline tests, and two API backends
(`OpenAIBrain`, `AnthropicBrain`) that differ only in how one call is made.

Every role is a single call with a narrow contract; the loop (code) decides the order and never lets a role judge
a score. Code-producing roles answer with a ```json header block followed by a ```python block, which avoids
escaping whole files inside JSON."""
from __future__ import annotations
import json, os, re, threading, time
from dataclasses import dataclass, asdict
from . import config as C
from . import prompts as P

FENCE_JSON = re.compile(r"```json\s*\n?(.*?)```", re.S)
FENCE_PY = re.compile(r"```python\s*\n(.*?)```", re.S)

class ParseError(ValueError):
    pass

def parse_header(text):
    m = FENCE_JSON.search(text)
    if not m:
        raise ParseError('no ```json block in the reply')
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError as e:
        raise ParseError(f'json header does not parse: {e}')

def parse_code(text):
    blocks = FENCE_PY.findall(text)
    if not blocks:
        raise ParseError('no ```python block in the reply')
    return max(blocks, key=len).rstrip() + '\n'      # the whole script is the longest block

# USD per million tokens: (uncached input, output, cached input, cache write)
PRICES = {'gpt-5.6-sol': (5.0, 30.0, 0.5, 0.0), 'gpt-5.6': (5.0, 30.0, 0.5, 0.0),
          'gpt-5.6-terra': (2.0, 12.0, 0.2, 0.0), 'gpt-5.6-luna': (0.2, 1.2, 0.02, 0.0),
          'claude-opus-5': (5.0, 25.0, 0.5, 6.25), 'claude-sonnet-5': (2.0, 10.0, 0.2, 2.5)}

@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0          # all input tokens, cached or not (what the organizers count)
    tokens_out: int = 0         # all output tokens, reasoning included
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    def add(self, model, tin_uncached, tout, cached=0, cache_write=0):
        p = PRICES.get(model) or PRICES.get(model.rsplit('-', 1)[0], PRICES['gpt-5.6-sol'])
        self.calls += 1; self.tokens_in += tin_uncached + cached; self.tokens_out += tout
        self.cache_read += cached; self.cache_write += cache_write
        self.cost_usd += (tin_uncached * p[0] + tout * p[1] + cached * p[2] + cache_write * p[3]) / 1e6
    def snapshot(self): return asdict(self)
    @staticmethod
    def delta(a, b): return {k: (b[k] - a[k]) for k in a}

class Brain:
    """Interface. Each method returns plain Python; raises ParseError/RuntimeError on unusable replies."""
    def __init__(self): self.usage = Usage(); self.calls = []
    def diagnose(self, ctx) -> str: raise NotImplementedError
    def select(self, ctx, k) -> list: raise NotImplementedError
    def implement(self, ctx, selection, parent_code, extra_parent_code=None) -> dict: raise NotImplementedError
    def critique(self, ctx, code, selection, diff_text='') -> dict: raise NotImplementedError
    def fix(self, ctx, code, error, log_tail) -> dict: raise NotImplementedError
    def consolidate(self, ctx, results) -> dict: raise NotImplementedError
    def explore(self, ctx): return None          # wildcard slot; backends may override
    def archive(self, ctx, rec, diff_text, card_ids, example, stack): return None   # wildcard -> card; LLM backends override
    def librarian(self, ctx, example): return []    # web-searched new cards; LLM backends override
    def probe(self, ctx, selection): return None    # ADR-0015 probe script for the feature screen; None = no screen
    def set_tag(self, tag): pass
    def set_context_block(self, text, roles=None): pass   # the generation-stable run journal block

class FakeBrain(Brain):
    """Scripted brain: `generations` is a list (one per generation) of lists of (selection, code) pairs.
    The fixer returns the parent's code (a 'revert' recovery); the critic always says ok."""
    def __init__(self, generations, diagnosis='fake diagnosis: the FM peaks at epoch 7 then overfits.'):
        super().__init__(); self.generations = generations; self.diagnosis = diagnosis
    def _gen(self, ctx): return self.generations[min(ctx['generation'] - 1, len(self.generations) - 1)]
    def diagnose(self, ctx): return self.diagnosis
    def select(self, ctx, k): return [dict(s) for s, _ in self._gen(ctx)][:k]
    def implement(self, ctx, selection, parent_code, extra_parent_code=None):
        for s, code in self._gen(ctx):
            if s['hypothesis'] == selection['hypothesis']:
                return {'code': code, 'change_summary': s['hypothesis']}
        return {'code': parent_code, 'change_summary': 'no scripted code; parent returned'}
    def critique(self, ctx, code, selection, diff_text=''): return {'verdict': 'ok', 'reasons': [], 'instructions': ''}
    def fix(self, ctx, code, error, log_tail): return {'code': ctx['parent_code'], 'note': 'fake fixer: reverted to the parent script'}
    def consolidate(self, ctx, results): return {'note': 'fake consolidator', 'plan': []}

class LLMBrain(Brain):
    """Shared role logic (prompt -> call -> parse -> validate, one format-reminder retry). Backends implement _call."""
    DEFAULT_EFFORT = {'diagnose': 'medium', 'select': 'xhigh', 'implement': 'xhigh', 'critique': 'medium',
                      'fix': 'medium', 'consolidate': 'medium', 'explore': 'xhigh', 'archive': 'medium', 'librarian': 'high',
                      'probe': 'medium'}
    MAX_TOKENS = {'diagnose': 3000, 'select': 16000, 'implement': 30000, 'critique': 4000, 'fix': 30000, 'consolidate': 5000,
                  'explore': 12000, 'archive': 8000, 'librarian': 20000, 'probe': 12000}   # reasoning tokens count against these (xhigh roles need room)
    ROLE_TOOLS = {'librarian': [{'type': 'web_search'}]}     # provider-side tools per role (OpenAI Responses API)

    def __init__(self, models=None, efforts=None, budget_usd=None, log=print):
        super().__init__()
        self._lock = threading.Lock(); self._tl = threading.local()   # calls may run from parallel branches
        self.models = dict(self.DEFAULT_MODELS, **(models or {}))
        self.efforts = dict(self.DEFAULT_EFFORT, **(efforts or {}))
        self.budget_usd = budget_usd; self.log = log; self._block = ''; self._block_roles = None

    def set_context_block(self, text, roles=None):
        self._block = text or ''; self._block_roles = set(roles) if roles else None

    def _block_for(self, role):
        return self._block if (self._block_roles is None or role in self._block_roles) else ''

    def set_tag(self, tag):
        """Tag subsequent calls from this thread (the loop uses the node id) so tokens can be attributed per node."""
        self._tl.tag = tag

    def _record(self, model, tin_uncached, tout, cached, cw, call):
        with self._lock:
            self.usage.add(model, tin_uncached, tout, cached, cw)
            call['tag'] = getattr(self._tl, 'tag', None); self.calls.append(call)

    def _check_budget(self):
        if self.budget_usd is not None and self.usage.cost_usd > self.budget_usd:
            raise RuntimeError(f'LLM budget exhausted: ${self.usage.cost_usd:.2f} > ${self.budget_usd:.2f}')

    def _call(self, role, user_text, retry_note=None) -> str: raise NotImplementedError

    def _with_retry(self, role, user_text, parse):
        try:
            return parse(self._call(role, user_text))
        except ParseError as e:
            self.log(f'  [{role}] parse failed ({e}); retrying once with a format reminder')
            return parse(self._call(role, user_text, retry_note=str(e)))

    def diagnose(self, ctx):
        return self._call('diagnose', P.user_diagnose(ctx)).strip()

    def select(self, ctx, k):
        def parse(t):
            sels = parse_header(t).get('selections')
            if not isinstance(sels, list) or not sels:
                raise ParseError('"selections" must be a non-empty list')
            for s in sels:
                for f in ('type', 'card', 'target_component', 'hypothesis', 'expected_delta', 'expected_delta_basis'):
                    if f not in s:
                        raise ParseError(f'selection missing field "{f}"')
                if s['target_component'] not in P.TARGET_COMPONENTS:
                    raise ParseError(f'unknown target_component {s["target_component"]!r}; use one of {P.TARGET_COMPONENTS}')
                s['expected_delta'] = float(s['expected_delta'])
            return sels[:k]
        return self._with_retry('select', P.user_select(ctx), parse)

    def explore(self, ctx):
        def parse(t):
            sels = parse_header(t).get('selections')
            if not isinstance(sels, list) or not sels:
                raise ParseError('"selections" must contain exactly one wildcard candidate')
            s = sels[0]
            for want, aliases in (('card', ('name', 'method', 'idea', 'title')), ('expected_delta_basis', ('basis', 'rationale', 'evidence')),
                                  ('hypothesis', ('description', 'proposal'))):
                if want not in s:
                    for a in aliases:
                        if a in s: s[want] = s[a]; break
            for f in ('card', 'target_component', 'hypothesis', 'expected_delta', 'expected_delta_basis'):
                if f not in s:
                    raise ParseError(f'wildcard missing field "{f}" (fields present: {sorted(s)})')
            if s['target_component'] not in P.TARGET_COMPONENTS:
                raise ParseError(f'unknown target_component {s["target_component"]!r}')
            s['expected_delta'] = float(s['expected_delta']); s['type'] = 'explore'
            return s
        return self._with_retry('explore', P.user_explore(ctx), parse)

    def archive(self, ctx, rec, diff_text, card_ids, example, stack):
        def parse(t):
            h = parse_header(t)
            if h.get('duplicate_of'):
                return {'id': h.get('id'), 'duplicate_of': h['duplicate_of'], 'card': None}
            blocks = re.findall(r'```card\n(.*?)\n```', t, re.S)
            if not blocks:
                raise ParseError('expected one ```card ... ``` block with the full card text')
            if not h.get('id'):
                raise ParseError('header must carry "id"')
            return {'id': str(h['id']).strip(), 'duplicate_of': None, 'card': blocks[0].strip() + '\n', 'family': h.get('family')}
        return self._with_retry('archive', P.user_archive(ctx, rec, diff_text, card_ids, example, stack), parse)

    def librarian(self, ctx, example):
        def parse(t):
            h = parse_header(t); cards = h.get('cards')
            if not isinstance(cards, list) or not cards:
                raise ParseError('"cards" must be a non-empty list')
            blocks = re.findall(r'```card\n(.*?)\n```', t, re.S)
            if len(blocks) < len(cards):
                raise ParseError(f'{len(cards)} cards announced but {len(blocks)} ```card``` blocks given')
            return [{'id': str(c.get('id', '')).strip(), 'source_url': c.get('source_url'), 'why_now': c.get('why_now'),
                     'card': b.strip() + '\n'} for c, b in zip(cards, blocks)]
        return self._with_retry('librarian', P.user_librarian(ctx, example), parse)

    def implement(self, ctx, selection, parent_code, extra_parent_code=None):
        def parse(t):
            return {'code': parse_code(t), 'change_summary': str(parse_header(t).get('change_summary', ''))}
        return self._with_retry('implement', P.user_implement(ctx, selection, parent_code, extra_parent_code), parse)

    def probe(self, ctx, selection):
        def parse(t):
            try:
                if parse_header(t).get('not_a_column'):
                    return None                       # the Probe declined: the signal is not a per-row column
            except ParseError:
                pass
            return parse_code(t)
        return self._with_retry('probe', P.user_probe(ctx, selection), parse)

    def critique(self, ctx, code, selection, diff_text=''):
        def parse(t):
            h = parse_header(t)
            if h.get('verdict') not in ('ok', 'revise', 'veto'):
                raise ParseError('verdict must be ok | revise | veto')
            return {'verdict': h['verdict'], 'reasons': h.get('reasons', []), 'instructions': h.get('instructions', ''),
                    'rebase_to': h.get('rebase_to')}   # ADR-0014: 'revise' onto another node's script
        return self._with_retry('critique', P.user_critique(ctx, code, selection, diff_text), parse)

    def fix(self, ctx, code, error, log_tail):
        def parse(t):
            return {'code': parse_code(t), 'note': str(parse_header(t).get('note', ''))}
        return self._with_retry('fix', P.user_fix(ctx, code, error, log_tail), parse)

    def consolidate(self, ctx, results):
        def parse(t):
            h = parse_header(t)
            return {'note': str(h.get('note', '')), 'plan': list(h.get('plan', []))[:ctx['k']]}
        return self._with_retry('consolidate', P.user_consolidate(ctx, results), parse)

class OpenAIBrain(LLMBrain):
    """Roles on the OpenAI Responses API (GPT-5.6 family). The stable prefix goes first in `instructions` so
    OpenAI's automatic prompt caching applies; reasoning effort is set per role; usage is metered per call."""
    DEFAULT_MODELS = {r: 'gpt-5.6-sol' for r in P.ROLE_SYSTEM}

    def __init__(self, models=None, efforts=None, budget_usd=None, log=print, api_key=None):
        super().__init__(models, efforts, budget_usd, log)
        from openai import OpenAI
        key = api_key or os.environ.get('OPENAI_API_KEY')
        if not key and (C.ROOT / '.env').exists():
            from dotenv import dotenv_values
            key = dotenv_values(C.ROOT / '.env').get('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY not found in the environment or .env')
        self.client = OpenAI(api_key=key, max_retries=3, timeout=900.0)

    def _call(self, role, user_text, retry_note=None):
        self._check_budget()
        model = self.models[role]; t0 = time.time()
        text_in = P.user_message(role, user_text) + (f"\n\nFORMAT REMINDER: {retry_note}" if retry_note else '')
        kw = dict(model=model, instructions=P.system_text(role, self._block_for(role)), input=text_in,
                  reasoning={'effort': self.efforts[role]}, max_output_tokens=self.MAX_TOKENS[role])
        tools = self.ROLE_TOOLS.get(role)
        if tools:
            kw['tools'] = tools
        try:
            r = self.client.responses.create(**kw)
        except Exception as e:   # noqa: BLE001 — older tool name on some deployments
            if tools and 'web_search' in str(e):
                kw['tools'] = [{'type': 'web_search_preview'}]; r = self.client.responses.create(**kw)
            else:
                raise
        searches = sum(1 for it in (getattr(r, 'output', None) or []) if getattr(it, 'type', '') == 'web_search_call')
        u = r.usage
        cached = getattr(getattr(u, 'input_tokens_details', None), 'cached_tokens', 0) or 0
        reasoning = getattr(getattr(u, 'output_tokens_details', None), 'reasoning_tokens', 0) or 0
        self._record(model, u.input_tokens - cached, u.output_tokens, cached, 0,
                     {'role': role, 'model': r.model, 'tokens_in': u.input_tokens, 'tokens_out': u.output_tokens,
                      'cached': cached, 'reasoning_tokens': reasoning, 'seconds': round(time.time() - t0, 1),
                      'status': r.status, 'response_id': r.id, 'web_searches': searches})
        if r.status != 'completed':
            why = getattr(getattr(r, 'incomplete_details', None), 'reason', r.status)
            raise ParseError(f'{role}: response {r.status} ({why}); max_output_tokens={self.MAX_TOKENS[role]}')
        text = r.output_text or ''
        if not text.strip():
            raise ParseError(f'{role}: empty reply')
        return text

class AnthropicBrain(LLMBrain):
    """Roles on the Anthropic Messages API (kept as an alternative backend)."""
    DEFAULT_MODELS = {r: 'claude-opus-5' for r in P.ROLE_SYSTEM}

    def __init__(self, models=None, efforts=None, budget_usd=None, log=print, use_fallbacks=True):
        super().__init__(models, efforts, budget_usd, log)
        import anthropic
        self.anthropic = anthropic; self.client = anthropic.Anthropic(max_retries=3); self.use_fallbacks = use_fallbacks

    def _call(self, role, user_text, retry_note=None):
        self._check_budget()
        model = self.models[role]; t0 = time.time()
        messages = [{'role': 'user', 'content': P.user_message(role, user_text) + (f"\n\nFORMAT REMINDER: {retry_note}" if retry_note else '')}]
        extra_body = {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': self.efforts[role]}}
        extra_headers = {}
        if self.use_fallbacks and model.startswith('claude-opus-5'):
            extra_body['fallbacks'] = 'default'; extra_headers['anthropic-beta'] = 'server-side-fallback-2026-07-01'
        try:
            with self.client.messages.stream(model=model, max_tokens=self.MAX_TOKENS[role], system=P.system_blocks(role, self._block_for(role)),
                                             messages=messages, extra_body=extra_body, extra_headers=extra_headers) as s:
                msg = s.get_final_message()
        except self.anthropic.BadRequestError as e:
            if self.use_fallbacks and 'fallback' in str(e).lower():
                self.use_fallbacks = False
                return self._call(role, user_text, retry_note)
            raise
        u = msg.usage
        cr = getattr(u, 'cache_read_input_tokens', 0) or 0; cw = getattr(u, 'cache_creation_input_tokens', 0) or 0
        self._record(model, u.input_tokens, u.output_tokens, cr, cw,
                     {'role': role, 'model': msg.model, 'tokens_in': u.input_tokens + cr + cw, 'tokens_out': u.output_tokens,
                      'cached': cr, 'seconds': round(time.time() - t0, 1), 'stop_reason': msg.stop_reason})
        if msg.stop_reason == 'refusal':
            raise RuntimeError(f'{role}: the model refused')
        if msg.stop_reason == 'max_tokens':
            raise ParseError(f'{role}: reply truncated at max_tokens={self.MAX_TOKENS[role]}')
        return ''.join(b.text for b in msg.content if b.type == 'text')
