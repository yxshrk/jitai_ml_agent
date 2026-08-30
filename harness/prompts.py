"""Prompt assembly for the LLM roles. One stable, cacheable prefix (brief + rules + contract + data facts + method
menu) plus a short role-specific system block and a dynamic user message. Never a growing transcript."""
from __future__ import annotations
import json, re
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

def _front_fields(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            if line and not line.startswith(' ') and ':' in line:
                k, v = line.split(':', 1); out[k.strip()] = v.strip()
    return out

def menu():
    """All method cards, preceded by a status table so the Selector can see at a glance what is alive, what is dead
    on which stack, and what is untried."""
    cards = sorted(p for p in (C.KB / 'methods').glob('*.md') if p.name != 'README.md') if (C.KB / 'methods').exists() else []
    if not cards:
        return FALLBACK_MENU
    rows = ['| card | component | status | card expected Δ |', '|---|---|---|---|']
    for p in cards:
        f = _front_fields(p.read_text())
        st = f.get('status', 'untried').replace('official FM', 'FM').replace('loss-bpr-pairwise-within-user', 'BPR')
        rows.append(f"| {f.get('id', p.stem)} | {f.get('target_component', '?')} | {st[:140]} | {f.get('expected_delta', '?')} |")
    legend = ("Status legend: `proven` = seed-confirmed in some run on the listed stack — NOT necessarily in the current champion (see Champion stack); `dead_under [stack xN (best Δ)]` = measured N "
              "times on that stack and never accepted (best Δ = best seed-mean delta seen) — do not re-propose it on that stack; "
              "a retest needs a stack not listed AND a stated reason (ADR-0004); `untried` = never measured. In stacks, FM = the "
              "official FM baseline and BPR = loss-bpr-pairwise-within-user. Each card's `## Measured` section has the per-node "
              "evidence: single-seed and seed-mean Δ, t statistic, verdict, diff size.")
    return '## Card status at a glance\n' + '\n'.join(rows) + '\n\n' + legend + '\n\n## Cards\n\n' + '\n\n'.join(p.read_text() for p in cards)

COMMON_PREAMBLE = """You are one role inside an autonomous ML-research harness for the KuaiRand-Pure within-user
ranking task. The harness (deterministic code) runs the loop, scores every script with the official evaluate.py and
applies these rules — """ + C.rules_text() + """
You perform exactly one role per call and answer in the required format. The facts below were measured on this
dataset; treat them as ground truth. A card's status says what was measured in SOME run on SOME stack; what the
current champion script actually contains is stated as 'Champion stack' in the context — never assume a method is
in a script because its card is proven. There is no test data anywhere you can see; never look for it.
All code must follow the script contract. Only numpy and the standard library are available."""

ROLE_SYSTEM = {
 'diagnose': """Role: DIAGNOSTICIAN. Write <= 8 lines of plain text (no code) from the champion's learning curve, its metrics
(GAUC, nDCG@5, ndcg5_disc = nDCG among users with mixed labels) and the last generation's per-node results:
(1) dynamics: overfit (validation peaks early then falls) / underfit (still rising) / flat — cite epochs and values;
(2) for each node of the last generation, WHICH HALF of the metric moved (GAUC vs nDCG@5) and by how much — a
loss change that moves GAUC but not nDCG points at top-of-list ordering; a feature that moves neither is dead here;
(3) the single most informative next probe and the component it targets, citing a numbered data fact;
(4) one line on validation-overfitting risk given the streak and how many sub-0.002 "wins" have been accepted;
(5) from the per-group breakdown (tabs, duration bands) and each node's by_group_delta: WHERE the champion is weakest
and which node moved which group — the deepen slot should target that. Group metrics are computed within group and
within user on SHORTER lists, so a group's nDCG@5 is a map of relative weakness, not comparable to the overall number
(tab=0 or dur=0 users are mostly all-negative, so their nDCG is near zero whatever the model does).""",
 'select': """Role: SELECTOR. Choose exactly k candidates for the next generation.
Rules: each candidate targets ONE component (target_component in %s); the k candidates must have DIFFERENT
target_components; fill the Consolidator's slots (merge / retest / deepen / explore) first, then the highest expected
gain per cost given the diagnosis; from generation 2 on EVERY candidate of yours must DEEPEN (type "deepen") unless it fills a
Consolidator merge/retest slot: a specific variant of the champion's own mechanism, or of a near-miss (|delta| <=
0.0005, unconfirmed), driven by the per-group breakdown and the diagnosis — e.g. negatives per positive, tab-stratified
pair sampling, a schedule change on the champion, a second iteration of a feature that moved one group — never a new
single-shot idea (the Explorer's slot is for those); depth beats breadth here (four runs: the confirmed gains came from
iterating two families, the wide generations returned flat single shots); never repeat an idea already measured on the
same parent unless it is a planned retest;
prefer the cheaper implementation when expected gains tie; in generation 1 at least one candidate must be a
ranking-aligned loss (organizers' lead #1).
CALIBRATION (measured in this project): predicted 0.006 / 0.004 / 0.003 realised +0.0022 / +0.0005 / -0.0003. Cards
state ranges for the whole family; a single first attempt lands in the LOWER THIRD of the card's range unless the
diagnosis gives specific evidence for more. Seed-to-seed SD is ~0.0003; acceptance needs a seed-mean gain of at least
MIN_EFFECT on fresh seeds at z >= Z_CRIT with the pooled seed SD, so real +0.0008 effects pass and single-seed flukes do not; the entire
remaining headroom is ~0.25.
Output exactly one fenced block:
```json
{"selections": [{"type": "improve|deepen|merge|retest|explore", "card": "<card id or method name>",
  "target_component": "<one of the list>", "hypothesis": "<one sentence: what changes and why it should help>",
  "expected_delta": 0.002, "expected_delta_basis": "<one sentence citing a card range, a numbered data fact or a journal line>",
  "cheapest_test": "<the smallest code change that tests the hypothesis>",
  "rejected_alternative": {"card": "...", "reason": "..."},
  "parent": "champion", "merge_parents": [<node ids, only for merge>]}]}
```""" % TARGET_COMPONENTS,
 'explore': """Role: EXPLORER (the wildcard slot). Propose exactly ONE candidate that is NOT a card on the menu as it stands:
(a) a combination of two mechanisms the cards treat separately, (b) a technique from ranking / recommendation
research that no card covers (name the paper or idea), or (c) an unconventional idea grounded in a numbered data
fact — the tab x duration structure, the repeated (user, video) pairs, the 18-second threshold, the volume collapse
after 04-12, the closed catalogue. Constraints: one target_component; implementable in numpy in under 120 changed
lines starting from the champion; not a hyper-parameter tweak; not something the journal already measured. Prefer
mechanisms orthogonal to what the champion stack already contains (see 'Champion stack' in the context; do not
assume anything else is in it). Be bold on the idea, honest on
expected_delta (lower third of what a card in that family would promise). Output exactly one fenced block with
exactly these fields:
```json
{"selections": [{"type": "explore", "card": "<short name for the idea>", "target_component": "<one of %s>",
  "hypothesis": "<one sentence: what changes and why it should help>", "expected_delta": 0.002,
  "expected_delta_basis": "<one sentence citing a paper, a numbered data fact or a journal line>",
  "cheapest_test": "<the smallest code change that tests it>",
  "rejected_alternative": {"card": "<another idea you considered>", "reason": "..."}, "parent": "champion"}]}
```""" % TARGET_COMPONENTS,
 'implement': """Role: IMPLEMENTER. EDIT the parent script; do not rewrite it. Return the parent script with ONLY the lines the
hypothesis requires changed, added or removed — keep every other line byte-for-byte, including the module
docstring (you may append one line to it), comments, import order, function names and the output code. The diff is
what the judges read: a one-component change is typically 5-80 changed lines; a diff above ~150 lines for an
"improve" node is a defect and will be sent back. Never restructure, rename, reformat, or "clean up".
Requirements: numpy + stdlib only; must finish in < 30 minutes on CPU (the parent takes ~15 s); predictions.csv and
metrics.json exactly per contract (metrics.json must include the per-epoch history); deterministic given --seed;
read only from --data-dir; never mention any path outside --data-dir anywhere in the file, not even in a comment or
docstring; never use an outcome column as a feature of the row being scored; history features only from rows
strictly earlier in time (for valid rows every train row is earlier). Keep row-level work vectorised or precomputed
outside the epoch loop. Keep the per-epoch print line. For a merge: apply both parents' changes onto the champion;
where they touch the same lines prefer the champion's version and say so in change_summary. The hypothesis is
FIXED: if the Critic's instructions would change it, ignore that part and implement the hypothesis as stated with
the smallest compliant change. When prior attempts of the same idea are listed, learn from their outcomes (what
was changed, what it scored, why it failed) instead of repeating them.
Output exactly: ```json {"change_summary": "<one line>"}``` followed by one ```python ... ``` block with the full script.""",
 'critique': """Role: CRITIC. Review the script before it runs. The candidate's hypothesis is FIXED: never ask for a different
hypothesis and never judge whether the expected gain is worth testing — the referee measures that. Check, in order:
(1) LEAKAGE — an outcome column (long_view, is_click, is_like, is_follow, is_comment, is_forward, is_hate,
play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter) used as a feature of the scored row; joins to
future data; the statistic file; history features not strictly earlier in time; any reference to test data or to
paths outside --data-dir. (2) CONTRACT — outputs, SMOKE_EPOCHS caps EVERY training phase, determinism,
predictions.csv row order and row_id logic untouched, runtime risk (pure-Python loops over a million rows inside
the epoch loop, quadratic pair construction). (3) SCOPE — judge it from the UNIFIED DIFF you are given: the change
implements the stated hypothesis and nothing else. The parent's stack is stated in the context: the candidate must
not add, remove or swap a loss, sampler, feature, schedule or ensemble that the hypothesis does not name — whatever
the cards mark as proven, and whatever the candidate's own text claims the parent contains. If the diff is far larger
than the hypothesis needs, say "revise" with the instruction to return the parent script with only the necessary edits. A minor over-reach that does not change the hypothesis (e.g. a coefficient
also applied to a second matrix) is a NOTE in reasons, not a revise.
Be terse: if the verdict is ok, give at most two short reasons; spend words only on problems. Veto only for leakage
or test access; everything else is revise (code changes only) or ok. Output exactly one fenced block:
```json {"verdict": "ok|revise|veto", "reasons": ["..."], "instructions": "<exact code changes, if revise>"}```""",
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
 {"type": "deepen", "parent": "champion", "card": "...", "hypothesis": "<a specific variant of the champion's mechanism, driven by the per-group breakdown>"},
 {"type": "explore", "parent": 2, "card": "...", "hypothesis": "..."}]}```""",
}

ROLE_SYSTEM['archive'] = """Role: ARCHIVIST. Turn one measured wildcard node (an idea that was not on the menu) into a method card so later
runs can select, retest or compose it. Write the card from the node's ACTUAL diff and measurement, not from what the idea hoped to do.
Same schema as the menu cards: front matter id, family, target_component, source, applies_when, expected_delta, expected_delta_basis,
cost, composes_with, conflicts_with, status, evidence; sections ## Claim, ## Mechanism, ## How to implement on node_000, ## Risks,
## Measured. Rules: id = "<target_component>-<short-slug>" (lowercase, hyphens), not an existing id; expected_delta = [lo, hi] that
brackets what was measured (seed-mean first), never above it, with lo >= 0 (a measured loss goes in ## Measured, not here); How to implement = the concrete edit from the diff (functions, shapes,
lines), at most 12 lines; composes_with / conflicts_with only existing card ids; status: untried and evidence: [] (the harness fills
them); ## Measured contains only "(none yet)"; at most 60 lines. Attribute honestly using the run journal in your instructions: if the
diff also contains a mechanism a sibling node measured on the same parent (e.g. a loss change), say so in Risks and bracket
expected_delta by the part this card's mechanism can claim. If the node is really an existing card (same mechanism), answer with
duplicate_of and no card block. Output exactly: ```json {"id": "<id>", "duplicate_of": null, "family": "<family>"}``` followed by one
```card ... ``` block holding the complete card text (front matter + body)."""

ROLE_SYSTEM['librarian'] = """Role: LIBRARIAN. Extend the menu with PUBLISHED methods that fit this problem's measured facts and the evidence in
the run journal (what is alive, what died on which stack, what nobody tried). Use web search to find concrete sources — papers,
competition write-ups (KDD Cup, RecSys Challenge, Kaggle, WSDM Cup), open-source recommender libraries (RecBole, DeepCTR, LightFM,
xLearn, TorchRec) — then check every candidate against Foundations (a user-constant term cannot move the metric) and the constraints
(numpy only, one CPU, script under 30 min, no test access, no pretrained weights, edits of node_000). Propose exactly n cards, each a
DIFFERENT mechanism from every existing card and from each other; preconditions checkable against numbered facts; expected_delta
honest against the 0.0005-0.005 range the journal shows; target_component must be one of %s (an optimizer or
learning-rate change is training-schedule; a new feature is features; a new interaction model is model); How to implement = a
concrete numpy edit of node_000 in at most 12 lines;
source = citation with URL. Prefer methods a Selector would plausibly pick next generation over exotic ones. Same schema as the menu
cards; status: untried; evidence: []; ## Measured "(none yet)"; at most 60 lines each. Output exactly: ```json {"cards": [{"id": "<id>",
"source_url": "<url>", "why_now": "<one line tied to the journal evidence>"}]}``` followed by one ```card ... ``` block per card, in
the same order."""

ROLE_SYSTEM['select'] = ROLE_SYSTEM['select'].replace('MIN_EFFECT', str(C.MIN_EFFECT)).replace('Z_CRIT', str(C.Z_CRIT))
ROLE_SYSTEM['librarian'] = ROLE_SYSTEM['librarian'] % (TARGET_COMPONENTS,)

_STABLE = None
def stable_prefix():
    """Built once per process so the cached prefix is byte-identical across calls."""
    global _STABLE
    if _STABLE is None:
        _STABLE = '\n\n'.join([
            COMMON_PREAMBLE,
            '# Task specification\n' + _read(C.KB / 'spec' / 'task.md'),
            '# Scoring\n' + _read(C.KB / 'spec' / 'scoring.md'),
            '# Foundations (task-specific mathematics)\n' + _read(C.KB / 'spec' / 'foundations.md'),
            '# Script contract\n' + _read(C.WORKSPACE / 'CONTRACT.md'),
            '# Measured data facts\n' + _read(C.KB / 'data' / 'facts.md'),
            '# Method menu\n' + menu(),
        ])
    return _STABLE

PLANNING_ROLES = ('diagnose', 'select', 'explore', 'consolidate', 'librarian', 'archive')   # roles that read the run journal block

def refresh_menu():
    """Forget the cached prefix so cards added by the Librarian/Archivist appear in the next call."""
    global _STABLE
    _STABLE = None

def untried_cards():
    return sorted(p.stem for p in (C.KB / 'methods').glob('*.md') if p.name != 'README.md'
                  and _front_fields(p.read_text()).get('status', '').startswith('untried'))

def run_block(generation, digest):
    """The exact run journal, frozen at the start of a generation: identical for every call of that generation, so
    it sits right after the stable prefix and the provider's cache serves it after the first call."""
    return (f'# Run journal so far (frozen at the start of generation {generation}; every node and measurement, with full '
            f'diffs for the champion lineage, accepted nodes and the last generation, stubs for older rejected nodes — read it '
            f'before proposing)\n\n{digest}')

def system_text(role, block=''):
    """OpenAI `instructions`: the stable prefix (byte-identical for every role and generation) followed by the
    generation-stable run block; both are served by the provider's prompt cache after the first call. The role text
    goes at the top of the user message (see user_message)."""
    return stable_prefix() + ('\n\n' + block if block else '')

def system_blocks(role, block=''):
    out = [{'type': 'text', 'text': stable_prefix(), 'cache_control': {'type': 'ephemeral'}}]
    if block:
        out.append({'type': 'text', 'text': block, 'cache_control': {'type': 'ephemeral'}})
    return out

def user_message(role, text):
    return f"## Your role in this call\n{ROLE_SYSTEM[role]}\n\n## Context\n{text}"

# ---------- dynamic user messages ----------
def _state(ctx):
    ch = ctx['champion']
    curve = ', '.join(f"{h['epoch']}:{h['val_primary']:.4f}" for h in ch.get('history', [])[:40])
    return (f"Run {ctx['run_id']} — generation {ctx['generation']} of at most {ctx['max_generations']}; "
            f"nodes used {ctx['nodes_used']}/{ctx['max_nodes']}; k = {ctx['k']}.\n"
            f"Champion: node_{ch['n']:03d} — primary {ch['metrics']['primary']:.4f} "
            f"(GAUC {ch['metrics']['gauc']:.4f}, nDCG@5 {ch['metrics']['ndcg5']:.4f}); hypothesis: {ch.get('hypothesis')}\n"
            f"Champion stack (accepted methods actually in its script): [{ctx.get('champion_stack', 'official FM')}]\n"
            f"Champion learning curve (epoch:valid primary): {curve}\n"
            + _breakdown(ch)
            + f"Best-so-far {ctx['best']:.4f}; non-improving generation streak {ctx['streak']} "
            f"(converged at {C.N_CONVERGE}); baseline valid primary {C.BASELINE_VALID_PRIMARY}.\n"
            f"Journal index (one line per node; the full record with diffs is the '# Run journal' section of your instructions):\n"
            + ('\n'.join(ctx['journal_lines']) or '(empty)'))

def _breakdown(ch):
    bg = (ch.get('metrics') or {}).get('by_group') or {}
    if not bg:
        return ''
    return ('Champion by group (where it is wrong; primary / GAUC / nDCG@5, rows): '
            + '; '.join(f"{g} {v['primary']:.4f}/{v['gauc']:.4f}/{v['ndcg5']:.4f} ({v['rows']})" for g, v in bg.items()) + '\n')

def user_diagnose(ctx):
    return _state(ctx) + '\n\nLast generation results:\n' + json.dumps(ctx.get('last_generation', []), indent=1, default=str)[:24000]

def user_select(ctx):
    plan = ctx.get('plan') or {}
    return (_state(ctx) + f"\n\nDiagnosis:\n{ctx.get('diagnosis', '(none)')}\n\n"
            f"Consolidator plan for this generation: {json.dumps(plan, default=str)}\n"
            f"Parked ideas (measured before, retest only with a reason): {json.dumps(ctx.get('parked', []), default=str)}\n"
            f"Choose exactly k = {ctx['k']} candidates, in priority order. One generation slot belongs to an Explorer role "
            f"whose target_component you cannot see: if it collides with one of yours, that one is dropped and your last "
            f"candidate takes the slot — so make the last one a genuine reserve, not a throwaway.")

def user_explore(ctx):
    cards = sorted(p.stem for p in (C.KB / 'methods').glob('*.md') if p.name != 'README.md') if (C.KB / 'methods').exists() else []
    return (_state(ctx) + f"\n\nDiagnosis:\n{ctx.get('diagnosis', '(none)')}\n\n"
            f"Cards already on the menu (do not propose these as they stand): {', '.join(cards)}\n"
            f"Propose exactly one wildcard candidate.")

def user_implement(ctx, selection, parent_code, extra_parent_code=None):
    s = (f"Candidate to implement:\n{json.dumps(selection, indent=1, default=str)}\n\n"
         f"Parent script node_{selection.get('parent_n', ctx['champion']['n']):03d}; its stack (everything it contains beyond the "
         f"official FM): [{ctx.get('parent_stack', 'official FM')}]. Nothing else on the menu is in it, whatever a card's status says. "
         f"Edit it for the candidate only.\n```python\n{parent_code}\n```\n")
    if extra_parent_code:
        s += f"\nSecond parent script for the merge:\n```python\n{extra_parent_code}\n```\n"
    if ctx.get('history_for_implementer'):
        s += "\nPrior attempts relevant to this candidate (same component or method), with outcomes:\n" + '\n'.join(ctx['history_for_implementer']) + "\n"
    if selection.get('critic_instructions'):
        s += f"\nThe Critic asked for these changes to your previous version:\n{selection['critic_instructions']}\n"
    return s

def user_critique(ctx, code, selection, diff_text=''):
    return (f"Candidate:\n{json.dumps(selection, indent=1, default=str)}\n\n"
            f"Parent: node_{selection.get('parent_n', ctx['champion']['n']):03d}; parent stack (what it actually contains): "
            f"[{ctx.get('parent_stack', ctx.get('champion_stack', 'official FM'))}]\nParent docstring:\n{ctx.get('parent_doc', '')[:1200]}\n\n"
            f"Unified diff, parent -> candidate (judge SCOPE from this):\n```diff\n{diff_text[:20000]}\n```\n\n"
            f"Full candidate script (for LEAKAGE and CONTRACT):\n```python\n{code}\n```")

def user_fix(ctx, code, error, log_tail):
    return f"Error: {error}\n\nLog tail:\n{log_tail}\n\nScript:\n```python\n{code}\n```"

def user_consolidate(ctx, results):
    return _state(ctx) + '\n\nThis generation:\n' + json.dumps(results, indent=1, default=str)[:12000] + f"\n\nk = {ctx['k']}."

def user_archive(ctx, rec, diff_text, card_ids, example_card, stack):
    keep = {k: rec.get(k) for k in ('n', 'hypothesis', 'target_component', 'method', 'expected_delta', 'expected_delta_basis',
                                    'change_summary', 'critic', 'metrics', 'realized_delta', 'accepted', 'seed_confirmation',
                                    'diff_lines', 'duration_s')}
    h = rec.get('history') or []
    curve = ' '.join(f"{x.get('val_primary', 0):.4f}" for x in h[:40])
    return (f"Run {ctx.get('run_id')}: archive node_{rec['n']:03d}, measured on the stack [{stack}].\n"
            f"Node record: {json.dumps(keep, default=str)}\nValid primary by epoch: {curve}\n\n"
            f"Diff against the parent script:\n```diff\n{diff_text[:12000]}\n```\n\n"
            f"Existing card ids (do not reuse; use for composes_with / conflicts_with): {', '.join(card_ids)}\n\n"
            f"Example card (format to copy exactly):\n```card\n{example_card}\n```")

def user_librarian(ctx, example_card):
    return (f"Propose exactly n = {ctx['n']} new cards.\n"
            f"Existing card ids: {', '.join(ctx['card_ids'])}\nStill untried: {', '.join(ctx.get('untried', [])) or '(none)'}\n"
            f"Run journal: {'in the # Run journal section of your instructions' if ctx.get('has_digest') else '(no run yet)'}\n"
            + (f"Latest results (after the frozen journal):\n{ctx['extra']}\n" if ctx.get('extra') else '') + "\n"
            f"Example card (format to copy exactly):\n```card\n{example_card}\n```")
