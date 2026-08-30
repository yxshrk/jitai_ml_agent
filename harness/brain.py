"""LLM roles behind one interface (`Brain`), plus `FakeBrain` for offline tests.

Every role is a single call with a narrow contract; the loop (code) decides the order and never lets a role judge
scores. Code-producing roles answer with a ```json header block followed by a ```python block, which avoids
escaping whole files inside JSON."""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, asdict
from typing import Optional
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

# USD per million tokens: (input, output, cache_read, cache_write)
PRICES = {'claude-opus-5': (5.0, 25.0, 0.5, 6.25), 'claude-sonnet-5': (2.0, 10.0, 0.2, 2.5),
          'claude-haiku-4-5': (1.0, 5.0, 0.1, 1.25)}

@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost_usd: float = 0.0
    def add(self, model, tin, tout, cr=0, cw=0):
        p = PRICES.get(model, PRICES['claude-opus-5'])
        self.calls += 1; self.tokens_in += tin; self.tokens_out += tout; self.cache_read += cr; self.cache_write += cw
        self.cost_usd += (tin * p[0] + tout * p[1] + cr * p[2] + cw * p[3]) / 1e6
    def snapshot(self): return asdict(self)
    @staticmethod
    def delta(a, b): return {k: (b[k] - a[k]) for k in a}

class Brain:
    """Interface. Each method returns plain Python; raises ParseError/RuntimeError on unusable replies."""
    def __init__(self): self.usage = Usage()
    def diagnose(self, ctx) -> str: raise NotImplementedError
    def select(self, ctx, k) -> list: raise NotImplementedError
    def implement(self, ctx, selection, parent_code, extra_parent_code=None) -> dict: raise NotImplementedError
    def critique(self, ctx, code, selection) -> dict: raise NotImplementedError
    def fix(self, ctx, code, error, log_tail) -> dict: raise NotImplementedError
    def consolidate(self, ctx, results) -> dict: raise NotImplementedError

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
    def critique(self, ctx, code, selection): return {'verdict': 'ok', 'reasons': [], 'instructions': ''}
    def fix(self, ctx, code, error, log_tail): return {'code': ctx['parent_code'], 'note': 'fake fixer: reverted to the parent script'}
    def consolidate(self, ctx, results): return {'note': 'fake consolidator', 'plan': []}

class AnthropicBrain(Brain):
    """Roles on the Anthropic Messages API. Newer request fields go through extra_body so the code works on any SDK
    line; the stable prefix is cached with cache_control; usage is metered per call."""
    DEFAULT_MODELS = {r: 'claude-opus-5' for r in P.ROLE_SYSTEM}
    DEFAULT_EFFORT = {'diagnose': 'medium', 'select': 'high', 'implement': 'high', 'critique': 'medium',
                      'fix': 'medium', 'consolidate': 'medium'}
    MAX_TOKENS = {'diagnose': 2000, 'select': 6000, 'implement': 24000, 'critique': 3000, 'fix': 24000, 'consolidate': 4000}

    def __init__(self, models=None, efforts=None, budget_usd=None, use_fallbacks=True, log=print):
        super().__init__()
        import anthropic                      # resolved from ANTHROPIC_API_KEY (or an `ant auth login` profile)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3)
        self.models = dict(self.DEFAULT_MODELS, **(models or {}))
        self.efforts = dict(self.DEFAULT_EFFORT, **(efforts or {}))
        self.budget_usd = budget_usd; self.use_fallbacks = use_fallbacks; self.log = log
        self.calls = []                       # per-call metering, appended to the journal by the loop

    def _call(self, role, user_text, retry_note=None):
        if self.budget_usd is not None and self.usage.cost_usd > self.budget_usd:
            raise RuntimeError(f'LLM budget exhausted: ${self.usage.cost_usd:.2f} > ${self.budget_usd:.2f}')
        model = self.models[role]
        messages = [{'role': 'user', 'content': user_text + (f"\n\nFORMAT REMINDER: {retry_note}" if retry_note else '')}]
        extra_body = {'thinking': {'type': 'adaptive'}, 'output_config': {'effort': self.efforts[role]}}
        extra_headers = {}
        if self.use_fallbacks and model.startswith('claude-opus-5'):
            extra_body['fallbacks'] = 'default'; extra_headers['anthropic-beta'] = 'server-side-fallback-2026-07-01'
        t0 = time.time()
        try:
            with self.client.messages.stream(model=model, max_tokens=self.MAX_TOKENS[role], system=P.system_blocks(role),
                                             messages=messages, extra_body=extra_body, extra_headers=extra_headers) as s:
                msg = s.get_final_message()
        except self.anthropic.BadRequestError as e:
            if self.use_fallbacks and 'fallback' in str(e).lower():   # API rejected the fallback beta: retry without it
                self.use_fallbacks = False
                return self._call(role, user_text, retry_note)
            raise
        u = msg.usage
        cr = getattr(u, 'cache_read_input_tokens', 0) or 0; cw = getattr(u, 'cache_creation_input_tokens', 0) or 0
        self.usage.add(model, u.input_tokens, u.output_tokens, cr, cw)
        self.calls.append({'role': role, 'model': msg.model, 'tokens_in': u.input_tokens, 'tokens_out': u.output_tokens,
                           'cache_read': cr, 'cache_write': cw, 'seconds': round(time.time() - t0, 1),
                           'stop_reason': msg.stop_reason, 'request_id': getattr(msg, '_request_id', None)})
        if msg.stop_reason == 'refusal':
            raise RuntimeError(f'{role}: the model refused (category {getattr(msg.stop_details, "category", None)})')
        if msg.stop_reason == 'max_tokens':
            raise ParseError(f'{role}: reply truncated at max_tokens={self.MAX_TOKENS[role]}')
        return ''.join(b.text for b in msg.content if b.type == 'text')

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

    def implement(self, ctx, selection, parent_code, extra_parent_code=None):
        def parse(t):
            return {'code': parse_code(t), 'change_summary': str(parse_header(t).get('change_summary', ''))}
        return self._with_retry('implement', P.user_implement(ctx, selection, parent_code, extra_parent_code), parse)

    def critique(self, ctx, code, selection):
        def parse(t):
            h = parse_header(t)
            if h.get('verdict') not in ('ok', 'revise', 'veto'):
                raise ParseError('verdict must be ok | revise | veto')
            return {'verdict': h['verdict'], 'reasons': h.get('reasons', []), 'instructions': h.get('instructions', '')}
        return self._with_retry('critique', P.user_critique(ctx, code, selection), parse)

    def fix(self, ctx, code, error, log_tail):
        def parse(t):
            return {'code': parse_code(t), 'note': str(parse_header(t).get('note', ''))}
        return self._with_retry('fix', P.user_fix(ctx, code, error, log_tail), parse)

    def consolidate(self, ctx, results):
        def parse(t):
            h = parse_header(t)
            return {'note': str(h.get('note', '')), 'plan': list(h.get('plan', []))[:ctx['k']]}
        return self._with_retry('consolidate', P.user_consolidate(ctx, results), parse)
