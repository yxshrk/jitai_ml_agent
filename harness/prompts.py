"""Prompt assembly for the LLM roles. One stable, cacheable prefix (brief + rules + contract + data facts + method
menu) plus a short role-specific system block and a dynamic user message. Never a growing transcript."""
from __future__ import annotations
import json
from pathlib import Path
from . import config as C

TARGET_COMPONENTS = ['features', 'encoding', 'model', 'loss', 'training-schedule', 'regularization',
                     'aux-targets', 'history', 'data-weighting', 'ensembling']

def _read(p):
    p = Path(p)
    return p.read_text() if p.exists() else ''

FALLBACK_MENU = """\
(No method cards yet — this is the organizers' own ranked list of unexplored directions plus the measured data facts.)
1. loss — align the objective with the ranking metric: within-user pairwise (BPR) or listwise softmax, or a hybrid with
   logloss. Organizers' #1 lead. Cheap (~60 lines), same runtime.
2. features — `duration_unknown` flag (duration_ms = 0 rows are always negative and currently share bucket 0), finer
   duration buckets, tab x duration crosses. Mechanism-backed by facts §3. Cheap.
3. data-weighting — recency weighting of training rows (traffic and positive rate drift; the last train days resemble
   valid/test). Mechanism-backed by facts §5. Cheap.
4. aux-targets — is_click (corr 0.76 with the label) as an auxiliary target with a shared embedding. Cheap.
5. history — per-user aggregates from earlier rows (rates by author / tab / duration bucket); histories are short
   (median 35 rows), so aggregates before sequence models.
6. model — DCN / DeepFM-style head on the same embeddings; organizers measured that embedding size alone is flat.
7. training-schedule / regularization — the FM peaks at epoch ~7 then overfits: lr decay, dropout on embeddings,
   stronger L2, early-stop on the right metric.
8. watch-time modelling (loss) — censored regression on play_time_ms (CWM / D2Q / TPM); heavier.
9. ensembling — seed or node ensembles as a closing move.
Measured dead by the organizers on FM + logloss (retest only with a changed stack, ADR-0004): adding the 13 static
CWM fields; embedding size k = 8 / 16 / 32.
"""

def menu():
    cards = sorted((C.KB / 'methods').glob('*.md')) if (C.KB / 'methods').exists() else []
    if not cards:
        return FALLBACK_MENU
    return '\n\n'.join(p.read_text() for p in cards)

COMMON_PREAMBLE = """You are one role inside an autonomous ML-research harness for the KuaiRand-Pure within-user
ranking task. The harness (deterministic code) runs the loop, scores every script with the official evaluate.py,
applies the acceptance rule (a node must beat the champion by >= 0.002 on validation primary) and the convergence
rule. You perform exactly one role per call and answer in the required format. The facts below were measured on
this dataset; treat them as ground truth. There is no test data anywhere you can see; never look for it.
All code must follow the script contract. Only numpy and the standard library are available."""

ROLE_SYSTEM = {
 'diagnose': """Role: DIAGNOSTICIAN. Given the champion's learning curve and the last generation's results, write a compact
diagnosis (<= 8 lines, plain text, no code): (1) training dynamics — overfit (validation peaks early then falls),
underfit (still rising at stop) or flat; (2) which pipeline component the evidence points at, citing data facts;
(3) what the last generation taught — which mechanisms moved the metric and which did not; (4) one line on the
risk of validation overfitting given the streak.""",
 'select': """Role: SELECTOR. Choose exactly k candidates for the next generation.
Rules: each candidate targets ONE component (target_component in %s); the k candidates must have DIFFERENT
target_components; if the Consolidator's plan lists slots (merge / retest / explore), fill those first, then use the
remaining slots for the highest-expected-gain fresh improvements given the diagnosis; never repeat an idea already
measured on the same parent unless it is a planned retest; expected_delta is a number (validation primary) with a
one-sentence basis citing a menu entry, a data fact or a journal line; name one rejected alternative and why.
Output exactly one fenced block:
```json
{"selections": [{"type": "improve|merge|retest|explore", "card": "<menu entry or method name>",
  "target_component": "<one of the list>", "hypothesis": "<one sentence: what changes and why it should help>",
  "expected_delta": 0.004, "expected_delta_basis": "<one sentence>",
  "rejected_alternative": {"card": "...", "reason": "..."},
  "parent": "champion", "merge_parents": [<node ids, only for merge>]}]}
```""" % TARGET_COMPONENTS,
 'implement': """Role: IMPLEMENTER. Produce the WHOLE runnable script for one candidate. Start from the parent script and make
the smallest coherent change that tests the hypothesis; keep everything else identical (data loading, outputs,
early stopping on validation primary, seed handling, SMOKE_EPOCHS). Requirements: numpy + stdlib only; must finish
in < 30 minutes on CPU; predictions.csv and metrics.json exactly per contract (metrics.json must include the
per-epoch history); deterministic given --seed; read only from --data-dir; never use an outcome column as a
feature of the row being scored; history features only from rows strictly earlier in time. For a merge: combine
the two parents' changes; where they touch the same code prefer the champion's version and say so.
Output exactly: ```json {"change_summary": "<one line>"}``` followed by one ```python ... ``` block with the full script.""",
 'critique': """Role: CRITIC. Review the script before it runs. Check: (1) LEAKAGE — an outcome column (long_view, is_click,
is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, profile_stay_time, comment_stay_time,
is_profile_enter) used as a feature of the scored row; joins to future data; the statistic file; history features not
strictly earlier in time; any reference to test data. (2) CONTRACT — outputs, SMOKE_EPOCHS honoured, determinism,
runtime risk (pure-Python loops over a million rows inside the epoch loop, quadratic pair construction, etc.).
(3) FIDELITY — the change implements the stated hypothesis and nothing else. (4) NOISE — is the expected_delta
plausible against the 0.002 floor? Verdict: ok | revise (fixable — give exact instructions) | veto (leakage or
test access). Output exactly one fenced block:
```json {"verdict": "ok|revise|veto", "reasons": ["..."], "instructions": "<what to change, if revise>"}```""",
 'fix': """Role: FIXER. The script failed. Return the corrected WHOLE script with the minimal change that fixes the error
without altering the hypothesis. If the failure is a timeout, reduce cost (fewer epochs, vectorise the slow loop)
while keeping the method. Output exactly: ```json {"note": "<what was wrong and what you changed>"}``` followed by one
```python ... ``` block with the full script.""",
 'consolidate': """Role: CONSOLIDATOR. Read this generation's verdicts (deltas, accepted flags, learning curves, what each node
changed). Plan up to k slots for the NEXT generation: (a) "merge" — two nodes with different target_components that
both improved the champion by >= 0.001, to be combined on the champion; (b) "retest" — a parked idea whose context
has changed (say why: changed stack, weak evidence, suspected bug); (c) "explore" — when the generation did not
improve, one slot from the runner-up lineage or an untried family. Leave the remaining slots to the Selector.
Do NOT decide acceptance — the referee already did. Output exactly one fenced block:
```json {"note": "<two lines of reasoning>", "plan": [{"type": "merge", "merge_parents": [3, 5], "hypothesis": "..."},
 {"type": "retest", "parent": "champion", "card": "...", "hypothesis": "...", "reason": "..."},
 {"type": "explore", "parent": 2, "card": "...", "hypothesis": "..."}]}```""",
}

_STABLE = None
def stable_prefix():
    """Built once per process so the cached prefix is byte-identical across calls."""
    global _STABLE
    if _STABLE is None:
        _STABLE = '\n\n'.join([
            COMMON_PREAMBLE,
            '# Task specification\n' + _read(C.KB / 'spec' / 'task.md'),
            '# Scoring\n' + _read(C.KB / 'spec' / 'scoring.md'),
            '# Script contract\n' + _read(C.WORKSPACE / 'CONTRACT.md'),
            '# Measured data facts\n' + _read(C.KB / 'data' / 'facts.md'),
            '# Method menu\n' + menu(),
        ])
    return _STABLE

def system_blocks(role):
    return [{'type': 'text', 'text': stable_prefix(), 'cache_control': {'type': 'ephemeral'}},
            {'type': 'text', 'text': ROLE_SYSTEM[role]}]

# ---------- dynamic user messages ----------
def _state(ctx):
    ch = ctx['champion']
    curve = ', '.join(f"{h['epoch']}:{h['val_primary']:.4f}" for h in ch.get('history', [])[:40])
    return (f"Run {ctx['run_id']} — generation {ctx['generation']} of at most {ctx['max_generations']}; "
            f"nodes used {ctx['nodes_used']}/{ctx['max_nodes']}; k = {ctx['k']}.\n"
            f"Champion: node_{ch['n']:03d} — primary {ch['metrics']['primary']:.4f} "
            f"(GAUC {ch['metrics']['gauc']:.4f}, nDCG@5 {ch['metrics']['ndcg5']:.4f}); hypothesis: {ch.get('hypothesis')}\n"
            f"Champion learning curve (epoch:valid primary): {curve}\n"
            f"Best-so-far {ctx['best']:.4f}; non-improving generation streak {ctx['streak']} "
            f"(converged at {C.N_CONVERGE}); baseline valid primary {C.BASELINE_VALID_PRIMARY}.\n"
            f"Journal (one line per node):\n" + ('\n'.join(ctx['journal_lines']) or '(empty)'))

def user_diagnose(ctx):
    return _state(ctx) + '\n\nLast generation results:\n' + json.dumps(ctx.get('last_generation', []), indent=1, default=str)[:6000]

def user_select(ctx):
    plan = ctx.get('plan') or {}
    return (_state(ctx) + f"\n\nDiagnosis:\n{ctx.get('diagnosis', '(none)')}\n\n"
            f"Consolidator plan for this generation: {json.dumps(plan, default=str)}\n"
            f"Parked ideas (measured before, retest only with a reason): {json.dumps(ctx.get('parked', []), default=str)}\n"
            f"Choose exactly k = {ctx['k']} candidates.")

def user_implement(ctx, selection, parent_code, extra_parent_code=None):
    s = (f"Candidate to implement:\n{json.dumps(selection, indent=1, default=str)}\n\n"
         f"Parent script (node_{selection.get('parent_n', ctx['champion']['n']):03d}):\n```python\n{parent_code}\n```\n")
    if extra_parent_code:
        s += f"\nSecond parent script for the merge:\n```python\n{extra_parent_code}\n```\n"
    if selection.get('critic_instructions'):
        s += f"\nThe Critic asked for these changes to your previous version:\n{selection['critic_instructions']}\n"
    return s

def user_critique(ctx, code, selection):
    return f"Candidate:\n{json.dumps(selection, indent=1, default=str)}\n\nScript:\n```python\n{code}\n```"

def user_fix(ctx, code, error, log_tail):
    return f"Error: {error}\n\nLog tail:\n{log_tail}\n\nScript:\n```python\n{code}\n```"

def user_consolidate(ctx, results):
    return _state(ctx) + '\n\nThis generation:\n' + json.dumps(results, indent=1, default=str)[:12000] + f"\n\nk = {ctx['k']}."
