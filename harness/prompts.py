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
    led = ledger() or {}; lrows = led.get('cards', {})
    def render(p):
        """Proven and untried cards in full; a card dead on every stack it was measured on is ONE ledger row (ADR-0018
        rule 7: id, component, family, status, record, bound) — the Selector needs its status and record to argue a
        retest, the Implementer receives the full card of the candidate it builds. Keeps the cached prefix bounded."""
        text = p.read_text()
        f = _front_fields(text)
        if not f.get('status', '').startswith('dead_under'):
            return text
        if lrows:
            e = lrows.get(f.get('id', p.stem), {})
            rec = e.get('measured_max'); b = e.get('bound')
            return (f"- DEAD `{f.get('id', p.stem)}` [{f.get('target_component')}/{e.get('family', f.get('family'))}]: {f.get('status', '')[:120]}; "
                    f"record {rec:+.4f}" + (f"; oracle bound {b:+.4f}" if b is not None and e.get('bound_kind') == 'oracle' else '')
                    + " — full card on request (a retest needs a new stack and a reason, ADR-0004)") if rec is not None else \
                   f"- DEAD `{f.get('id', p.stem)}` [{f.get('target_component')}]: {f.get('status', '')[:120]}"
        
        fm_end = text.index('\n---\n', 4) + 5
        body = text[fm_end:]
        claim = body[body.index('## Claim'):] if '## Claim' in body else body
        for sec in ('## Mechanism', '## How to implement', '## Risks', '## Measured'):
            if sec in claim:
                claim = claim[:claim.index(sec)]
        verdict = ''
        if '_Verdict:_' in body:
            v = body[body.index('_Verdict:_'):]; verdict = v.split('\n', 1)[0] + '\n'
        return text[:fm_end] + claim.rstrip() + '\n' + verdict + '(mechanism, recipe and per-node evidence in the full card, given to the Implementer on selection)\n'
    return '## Card status at a glance\n' + '\n'.join(rows) + '\n\n' + legend + '\n\n## Cards\n\n' + '\n\n'.join(render(p) for p in cards)

COMMON_PREAMBLE = """You are one role inside an autonomous ML-research harness for the KuaiRand-Pure within-user
ranking task. The harness (deterministic code) runs the loop, scores every script with the official evaluate.py and
applies these rules — """ + C.rules_text() + """
You perform exactly one role per call and answer in the required format. The facts below were measured on this
dataset; treat them as ground truth. A card's status says what was measured in SOME run on SOME stack; what the
current champion script actually contains is stated as 'Champion stack' in the context — never assume a method is
in a script because its card is proven. There is no test data anywhere you can see; never look for it.
All code must follow the script contract. """ + C.libs_text()

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
(tab=0 or dur=0 users are mostly all-negative, so their nDCG is near zero whatever the model does). Groups listed as
HARD in the state (>= HARD_GROUP_REJECTS rejected deepens; likely irreducible label noise — a long video needs only 18 s of play
to be a positive) are not "next": say so in one clause and move on. The PAIR-TYPE table is the sharper map: GAUC is
pair error, and each node's by_pair_delta says which pair types it moved (negative = fewer misordered). tab1_x_tab1 pairs
on different days carry ~69 % of the mass (facts §11); a node that moves only gap<10min pairs (2 %) or one small group
has touched little of what the metric counts — say which pair type the champion's error sits in and whether the
generation moved it. (6) If the state names a CAMPAIGN family: one clause
on whether its variants moved anything (which mechanism, which group) and whether it should stay open.""",
 'select': """Role: SELECTOR. Choose exactly k candidates for the next generation.
Rules: each candidate targets ONE component (target_component in %s); the k candidates must have DIFFERENT
target_components; fill the Consolidator's slots (merge / retest / deepen / explore) first, then the highest expected
gain per cost given the diagnosis; from generation FREE_SLOT_FROM_GENERATION on, your FIRST candidate is the FREE SLOT (ADR-0014),
filled in this priority: (i) an "untried" card whose applies_when holds against the facts, (ii) a proven card not yet
measured on the current champion stack, (iii) only if the context lists neither, a deepen — the state lists both sets;
code checks that the free slot is used and asks once more if it is not (two runs left history-same-author-run-features
untried because every later slot deepened); every OTHER candidate of yours must DEEPEN (type "deepen") unless it fills a
Consolidator merge/retest slot: a specific variant of the champion's own mechanism, or of a near-miss (|delta| <=
0.0005, unconfirmed), driven by the per-group breakdown and the diagnosis — e.g. negatives per positive, tab-stratified
pair sampling, a schedule change on the champion, a second iteration of a feature that moved one group — never a new
single-shot idea (the Explorer's slot is for those); depth beats breadth here (four runs: the confirmed gains came from
iterating two families, the wide generations returned flat single shots); never repeat an idea already measured on the
same parent unless it is a planned retest. A deepen carries "mechanism" (a short slug of the mechanism family, e.g.
long-duration-matched-pairs) and "target_group" (a breakdown group such as dur>180s or tab=4, or "all"): a mechanism
rejected this run is CLOSED for deepening for the rest of the run (the state lists them; code drops repeats) — the next
deepen must change the mechanism, not the dose, the sampling fraction or the group (live_06 spent five nodes shrinking
one rejected mechanism: 10 %% -> 5 %% -> 2.5 %%); groups marked HARD in the state are not deepen targets; a deepen of a
near-miss node edits THAT node's script, so its "parent" is that node's id.
FRONTIER AND QUEUE (ADR-0021): you are not restricted to the champion. The state lists a FRONTIER — every node worth
building on, including unconfirmed near-misses whose fresh-seed mean is within one standard error of the champion's
(three earlier runs' highest-mean node was such a node and nobody ever built on it) — and a QUEUE of proposals already
waiting. Set "parent" to whichever frontier node your candidate should edit, and spread your candidates across the
frontier where the evidence supports it rather than piling every slot on the champion. Your proposals go into the
queue: the best eligible ones run this generation and the rest wait, so propose the idea you believe in even when the
generation looks full, and never repeat something already queued.
CAMPAIGN (ADR-0016): when the state names a campaign family, EVERY candidate of yours except the Consolidator's
merge/retest slots belongs to that family (its cards are listed with their status): the family's untried or
not-yet-stacked cards take the free slot, the rest are variants of the family's mechanisms on the champion, each with
a DISTINCT "mechanism" slug ("mechanism" is required for every campaign candidate; code keeps one candidate per
mechanism, so a second dose of the same mechanism is dropped). A family closes after CAMPAIGN_FLAT_GENERATIONS generations
without an accepted node from it and the next family opens — spend the generation on the family's genuinely different
mechanisms, not on one mechanism at three doses;
prefer the cheaper implementation when expected gains tie; in generation 1 at least one candidate must be a
ranking-aligned loss (organizers' lead #1).
CALIBRATION_SENTENCE Cards state ranges for the whole family; a single first attempt lands in the LOWER THIRD of the
card's range unless the diagnosis gives specific evidence for more. Seed-to-seed SD is ~0.0003; acceptance needs a seed-mean gain of at least
MIN_EFFECT on fresh seeds at z >= Z_CRIT with the pooled seed SD, so real +0.0008 effects pass and single-seed flukes do not; the entire
remaining headroom is ~0.25.
Output exactly one fenced block:
```json
{"selections": [{"type": "improve|deepen|merge|retest|explore", "card": "<card id or method name>",
  "target_component": "<one of the list>", "hypothesis": "<one sentence: what changes and why it should help>",
  "expected_delta": 0.002, "expected_delta_basis": "<one sentence citing a card range, a numbered data fact or a journal line>",
  "cheapest_test": "<the smallest code change that tests the hypothesis>",
  "rejected_alternative": {"card": "...", "reason": "..."},
  "mechanism": "<deepen only: slug of the mechanism family>", "target_group": "<deepen only: breakdown group or all>",
  "parent": "<node whose script is edited: champion for a variant of the champion; the near-miss node's id for a variant of that node (a deepen of node_010 -> 10)>",
  "merge_parents": [<node ids, only for merge>]}]}
```""" % TARGET_COMPONENTS,
 'explore': """Role: EXPLORER (the wildcard slot). Propose exactly ONE candidate that is NOT a card on the menu as it stands:
(a) a combination of two mechanisms the cards treat separately, (b) a technique from ranking / recommendation
research that no card covers (name the paper or idea), or (c) an unconventional idea grounded in a numbered data
fact — the tab x duration structure, the repeated (user, video) pairs, the 18-second threshold, the volume collapse
after 04-12, the closed catalogue. Constraints: one target_component; implementable with the contract's libraries in under
120 changed lines starting from the champion; not a hyper-parameter tweak; not something the journal already measured;
and it must ADD INFORMATION (ADR-0014): a signal absent from the champion's input set (listed in the state as 'Champion
input set') — an exposure/session feature from earlier rows, a side-table field, a history statistic, a library model
that uses the inputs differently — named in "new_signal" with where it comes from. Capacity-only proposals (higher-order
terms, tensors, extra heads over the same id fields) are dropped by code: live_06's four such wildcards measured -0.0026,
-0.0007, -0.0005 and -0.0004 on five id fields where capacity is not the bottleneck. Prefer
mechanisms orthogonal to what the champion stack already contains (see 'Champion stack' in the context; do not
assume anything else is in it). Be bold on the idea, honest on
expected_delta (lower third of what a card in that family would promise). Output exactly one fenced block with
exactly these fields:
```json
{"selections": [{"type": "explore", "card": "<short name for the idea>", "target_component": "<one of %s>",
  "hypothesis": "<one sentence: what changes and why it should help>", "expected_delta": 0.002,
  "expected_delta_basis": "<one sentence citing a paper, a numbered data fact or a journal line>",
  "cheapest_test": "<the smallest code change that tests it>",
  "rejected_alternative": {"card": "<another idea you considered>", "reason": "..."},
  "new_signal": "<the input signal the champion does not have, and which rows/table it is computed from>", "parent": "champion"}]}
```""" % TARGET_COMPONENTS,
 'implement': """Role: IMPLEMENTER. EDIT the parent script; do not rewrite it. Return the parent script with ONLY the lines the
hypothesis requires changed, added or removed — keep every other line byte-for-byte, including the module
docstring (you may append one line to it), comments, import order, function names and the output code. The diff is
what the judges read: a one-component change is typically 5-80 changed lines; a diff above ~150 lines for an
"improve" node is a defect and will be sent back. Never restructure, rename, reformat, or "clean up".
Requirements: only the libraries the contract lists (numpy, pandas, scikit-learn, LightGBM, torch on CPU), seeded and
thread-limited exactly as the contract says; must finish in < 30 minutes on CPU (the numpy FM parent takes ~15 s; budget
trees and torch epochs accordingly); predictions.csv and
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
also applied to a second matrix) is a NOTE in reasons, not a revise. (4) INFORMATION — for a wildcard (type explore)
the diff must add the named new_signal as an input; if it only adds capacity over the champion's existing inputs, say
"revise" with the instruction to implement the named signal. (5) LIBRARIES — the contract's determinism rules: CPU only,
threads from OMP_NUM_THREADS, every library seeded from --seed, SMOKE_EPOCHS capping boosting rounds too.
REBASE: when the hypothesis deepens or edits a SPECIFIC earlier node (e.g. a variant of node_010) but the diff is against
another script, answer "revise" with "rebase_to": <that node id> — the harness hands the Implementer that node's script
for the next round; never ask for a re-implementation of the other node's mechanism on the wrong parent (live_06 lost a
slot to three such rounds).
Be terse: if the verdict is ok, give at most two short reasons; spend words only on problems. Veto only for leakage
or test access; everything else is revise (code changes only) or ok. Output exactly one fenced block:
```json {"verdict": "ok|revise|veto", "reasons": ["..."], "instructions": "<exact code changes, if revise>", "rebase_to": <node id, only with revise when the candidate must be built on another node's script, else null>}```""",
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
 {"type": "deepen", "parent": "<the node whose script is edited: champion, or the near-miss node's id (a deepen of node_010 -> 10)>", "card": "...", "mechanism": "<slug>", "target_group": "<group or all>", "hypothesis": "<a specific variant of that node's mechanism, driven by the per-group breakdown; never a closed mechanism or a hard group>"},
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
```card ... ``` block holding the complete card text (front matter + body). A node whose signal is already a card's — the same inputs and mechanism family with a different dose, gate,
threshold or cohort (live_06 minted six cards for variants of two mechanisms) — is NOT a new card: answer duplicate_of
that card and the harness files the measurement on it. Mint a card only for a signal or mechanism no card has."""

ROLE_SYSTEM['librarian'] = """Role: LIBRARIAN. Extend the menu with PUBLISHED methods that fit this problem's measured facts and the evidence in
the run journal (what is alive, what died on which stack, what nobody tried). Use web search to find concrete sources — papers,
competition write-ups (KDD Cup, RecSys Challenge, Kaggle, WSDM Cup), open-source recommender libraries (RecBole, DeepCTR, LightFM,
xLearn, TorchRec) — then check every candidate against Foundations (a user-constant term cannot move the metric) and the constraints
(the contract's libraries — numpy, pandas, scikit-learn, LightGBM, torch on CPU — one CPU, script under 30 min, no test access, no pretrained weights, edits of node_000). Propose exactly n cards, each a
DIFFERENT mechanism from every existing card and from each other; preconditions checkable against numbered facts; expected_delta
honest against the 0.0005-0.005 range the journal shows; target_component must be one of %s (an optimizer or
learning-rate change is training-schedule; a new feature is features; a new interaction model is model); How to implement = a
concrete edit of node_000 in at most 12 lines (any contract library);
source = citation with URL. Prefer methods a Selector would plausibly pick next generation over exotic ones. Same schema as the menu
cards; status: untried; evidence: []; ## Measured "(none yet)"; at most 60 lines each. Output exactly: ```json {"cards": [{"id": "<id>",
"source_url": "<url>", "why_now": "<one line tied to the journal evidence>"}]}``` followed by one ```card ... ``` block per card, in
the same order."""

ROLE_SYSTEM['select'] = (ROLE_SYSTEM['select'].replace('MIN_EFFECT', str(C.MIN_EFFECT)).replace('Z_CRIT', str(C.Z_CRIT))
                        .replace('FREE_SLOT_FROM_GENERATION', str(C.FREE_SLOT_FROM_GENERATION))
                        .replace('CAMPAIGN_FLAT_GENERATIONS', str(C.CAMPAIGN_FLAT_GENERATIONS)))
ROLE_SYSTEM['diagnose'] = ROLE_SYSTEM['diagnose'].replace('HARD_GROUP_REJECTS', str(C.HARD_GROUP_REJECTS))
_SELECT_TEMPLATE = ROLE_SYSTEM['select']

def refresh_calibration():
    """Regenerate the Selector's calibration sentence from the ledger (called with refresh_menu, ADR-0018)."""
    ROLE_SYSTEM['select'] = _SELECT_TEMPLATE.replace('CALIBRATION_SENTENCE', calibration_text())
ROLE_SYSTEM['librarian'] = ROLE_SYSTEM['librarian'] % (TARGET_COMPONENTS,)
ROLE_SYSTEM['probe'] = """Role: PROBE (ADR-0015). Before a feature hypothesis gets a node, the harness measures the proposed signal on
the valid split against the champion's own predictions. Write the PROBE SCRIPT that computes that signal: a short standalone
Python program taking --data-dir and --out-dir, reading only files under --data-dir (train.csv with its outcome columns,
valid.csv WITHOUT outcome columns — the harness strips them — and the side tables), and writing <out-dir>/features.csv with the
header row_id,<name>[,<name>...] and exactly one row per valid.csv row in file order (row_id 0..n-1), numeric finite values,
at most 8 columns. Compute the feature exactly as the hypothesis / new_signal describes it, as raw as possible (a rate, a count,
a gap in seconds, a 0/1 flag — not a bucket id unless the hypothesis is about buckets): the screen tests within-user
discrimination of the number itself and of a small tree model on top of it. Train labels may be used for statistics
(smoothed rates, out-of-fold is not needed for valid rows); history features only from rows strictly earlier in time. Aim
for under 60 seconds with numpy/pandas (vectorised; no per-row Python loops over a million rows); pandas 2.3, numpy,
scikit-learn and the standard library are available. Determinism: no randomness unless seeded with 0. The probe is not
the implementation and is never scored as one — do not train the FM, do not write predictions.csv. Output exactly one
```python block containing the whole script and nothing else. If the proposed signal is NOT something a column per valid row
can carry (a model architecture, a loss, a training schedule), answer instead with one ```json block {"not_a_column": "<why>"}
and no python block — the candidate then proceeds unscreened."""


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
    """Forget the cached prefix so cards added by the Librarian/Archivist appear in the next call; regenerate the
    calibration sentence from the ledger (ADR-0018)."""
    global _STABLE
    _STABLE = None
    refresh_calibration()

def untried_cards():
    return sorted(p.stem for p in (C.KB / 'methods').glob('*.md') if p.name != 'README.md'
                  and _front_fields(p.read_text()).get('status', '').startswith('untried'))

def proven_cards():
    return sorted(p.stem for p in (C.KB / 'methods').glob('*.md') if p.name != 'README.md'
                  and _front_fields(p.read_text()).get('status', '').startswith('proven'))

def card_index():
    """{card id: {family, target_component, status, expected_hi}} from every card's front matter (ADR-0016 campaigns)."""
    out = {}
    for p in sorted((C.KB / 'methods').glob('*.md')) if (C.KB / 'methods').exists() else []:
        if p.name == 'README.md':
            continue
        f = _front_fields(p.read_text())
        nums = re.findall(r'-?\d+(?:\.\d+)?', f.get('expected_delta', ''))
        out[f.get('id', p.stem)] = {'family': f.get('family') or 'other', 'target_component': f.get('target_component'),
                                    'status': f.get('status', 'untried'), 'expected_hi': max(float(x) for x in nums) if nums else None}
    return out

PROMISE_DISCOUNT = 0.3      # ADR-0018 rule 6b: a paper/analogy promise counts at the realised/predicted ratio measured over five runs

def ledger():
    """The family ledger (ADR-0018, kb/methods/ledger.py, research session) built from the cards NOW — never a stale file.
    None when the generator is not present (older checkouts)."""
    import importlib.util
    path = C.KB / 'methods' / 'ledger.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('kb_ledger', path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(C.KB / 'methods')

def card_value(entry):
    """What a card may still be worth, from its ledger row: the record when measured (0 if never positive), else the
    promise — discounted for paper/analogy bases and never above an ORACLE bound (ADR-0018 rules 6a/6b)."""
    rec = entry.get('measured_max_ref') if entry.get('measured_max_ref') is not None else entry.get('measured_max')
    if rec is not None:                    # the record on the reference stack when there is one, else on any stack
        return max(0.0, float(rec))
    hi = (entry.get('expected') or [None, None])[1]
    v = float(hi) if hi is not None else 0.0
    if entry.get('basis_class') in ('paper', 'analogy'):
        v *= PROMISE_DISCOUNT
    if entry.get('bound_kind') == 'oracle' and entry.get('bound') is not None:
        v = min(v, float(entry['bound']))
    return v

def calibration_text(led=None):
    """The Selector's calibration sentence, generated from the ledger like rules_text() from config (ADR-0018 rule 6c)."""
    led = led if led is not None else ledger()
    if not led:
        return ('CALIBRATION (measured in this project): predicted 0.006 / 0.004 / 0.003 realised +0.0022 / +0.0005 / -0.0003; '
                'every accepted gain so far was +0.0009 to +0.0022 on fresh seeds.')
    cards = led.get('cards', {})
    acc = sorted(float(v['measured_max']) for v in cards.values() if v.get('accepted') and v.get('measured_max') is not None)
    n_meas = sum(1 for v in cards.values() if v.get('measured_max') is not None)
    never = sum(1 for v in cards.values() if v.get('measured_max') is not None and v['measured_max'] <= 0)
    fams = led.get('families', {})
    bounded = [f for f, v in fams.items() if v.get('status') == 'bounded']; exhausted = [f for f, v in fams.items() if v.get('status') == 'exhausted']
    return (f"CALIBRATION (generated from the ledger, ADR-0018): {n_meas} cards measured, {never} never positive; every ACCEPTED gain was "
            + (f"{acc[0]:+.4f} to {acc[-1]:+.4f}" if acc else 'none yet') + " on fresh seeds; a card's expected_delta upper IS its record once measured "
            f"(0 if never positive), an unmeasured paper/analogy promise counts x{PROMISE_DISCOUNT} and never above its family's oracle bound; "
            f"bounded families (nothing in them can clear acceptance): {', '.join(bounded) or 'none'}; exhausted: {', '.join(exhausted) or 'none'}.")

def family_of(card, index=None):
    """The family of a card id, also for a deepen named '<card id> — <variant>'; None for an unknown name."""
    index = index if index is not None else card_index()
    return index.get(str(card or '').split(' — ')[0].strip(), {}).get('family')

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
            + _rules_state(ctx)
            + f"Best-so-far {ctx['best']:.4f}; non-improving generation streak {ctx['streak']} "
            f"(converged at {C.N_CONVERGE}); baseline valid primary {C.BASELINE_VALID_PRIMARY}.\n"
            f"Journal index (one line per node; the full record with diffs is the '# Run journal' section of your instructions):\n"
            + ('\n'.join(ctx['journal_lines']) or '(empty)'))

def _campaign_state(ctx):
    """ADR-0016: the campaign family of this generation and every family's status."""
    fams = ctx.get('families') or {}
    if not fams and not ctx.get('campaign'):
        return ''
    camp = ctx.get('campaign')
    rows = '; '.join(f"{f} ({v.get('status')}" + (f", best {v['best_gain']:+.4f}" if v.get('best_gain') is not None else '')
                     + (f", flat {v.get('flat_streak')}" if v.get('flat_streak') else '') + ')' for f, v in fams.items())
    return ((f"CAMPAIGN this generation: {camp} — every Selector candidate except Consolidator merge/retest slots belongs to this "
             f"family, one per DISTINCT mechanism; its cards: {', '.join(ctx.get('campaign_cards') or []) or 'none'}\n"
             if camp else "CAMPAIGN this generation: none (breadth generation, or every family is closed)\n")
            + f"Families (status, best seed-mean gain, flat generations while campaigning): {rows or 'none'}\n")

def _frontier_state(ctx):
    """ADR-0021: the nodes worth building on (not only the champion) and the proposals already waiting."""
    fr = ctx.get('frontier') or []
    if not fr:
        return ''
    rows = '; '.join(f"node_{e['n']:03d}{'*' if e['champion'] else ''} mean {e['mean']:.5f} "
                     f"({'accepted' if e['accepted'] else 'unconfirmed'}, {e['children']} children"
                     + (f", {e['barren_generations']} barren" if e['barren_generations'] else '') + ')' for e in fr)
    q = ctx.get('queue') or []
    qrows = ('; '.join(f"[{x['parent']}] {x.get('card')} ({x.get('mechanism') or x.get('target_component')}, score {x.get('score', 0):+.4f})"
                       for x in q[:8]) if q else 'empty')
    return (f"FRONTIER (nodes you may build on; * = champion; a node retires after 2 generations without an accepted child): {rows}\n"
            f"QUEUE (proposals already waiting, best first — do not repeat them; they run when a slot frees): {qrows}\n")

def _rules_state(ctx):
    """ADR-0014 planning state: what the free slot may take, what is closed, what is hard, what the champion reads."""
    hard = ctx.get('hard_groups') or {}; closed = ctx.get('closed_mechanisms') or {}
    return (_frontier_state(ctx) + _campaign_state(ctx) + f"Untried cards (free-slot priority 1): {', '.join(ctx.get('untried') or []) or 'none'}\n"
            f"Proven cards not in the champion stack (free-slot priority 2): {', '.join(ctx.get('proven_not_on_stack') or []) or 'none'}\n"
            f"Champion input set (columns / side-table fields its script references): {', '.join(ctx.get('champion_inputs') or []) or 'unknown'}\n"
            f"Closed mechanisms (deepens rejected this run; not to be deepened again): "
            + ('; '.join(f"{m} (node(s) {ns})" for m, ns in closed.items()) if closed else 'none') + '\n'
            f"HARD groups (>= {C.HARD_GROUP_REJECTS} rejected deepens; likely irreducible, move on): "
            + ('; '.join(f"{g} (node(s) {ns})" for g, ns in hard.items()) if hard else 'none') + '\n'
            + _screened_state(ctx))

def _screened_state(ctx):
    """ADR-0015: features the screen measured on valid against the champion this run; below SCREEN_MIN_GAIN they were not built."""
    sc = ctx.get('screened') or []
    if not sc:
        return ''
    return (f"Screened this run (feature measured on valid against the champion before any node; best_gain = max(additive, stack); "
            f"below {C.SCREEN_MIN_GAIN} the slot was dropped — do not re-propose a dropped signal in another wording): "
            + '; '.join(f"{e.get('card')} [{e.get('family')}] {e.get('best_gain'):+.4f} {'kept' if e.get('kept') else 'DROPPED'}" for e in sc) + '\n')

def _breakdown(ch):
    bg = (ch.get('metrics') or {}).get('by_group') or {}
    if not bg:
        return ''
    text = ('Champion by group (where it is wrong; primary / GAUC / nDCG@5, rows): '
            + '; '.join(f"{g} {v['primary']:.4f}/{v['gauc']:.4f}/{v['ndcg5']:.4f} ({v['rows']})" for g, v in bg.items()) + '\n')
    bp = (ch.get('metrics') or {}).get('by_pair') or {}
    if bp:   # GAUC is pair error: which positive-negative pair types carry it (facts §11)
        text += ('Champion by PAIR TYPE (share of the GAUC pair mass / misordered fraction): '
                 + '; '.join(f"{t} {v['share']:.2f}/{v['err']:.3f}" for t, v in bp.items() if isinstance(v, dict) and v.get('err') is not None)
                 + f" | total misordered {bp.get('total_err')}\n")
    return text

def user_diagnose(ctx):
    return _state(ctx) + '\n\nLast generation results:\n' + json.dumps(ctx.get('last_generation', []), indent=1, default=str)[:24000]

def user_select(ctx):
    plan = ctx.get('plan') or {}
    return (_state(ctx) + f"\n\nDiagnosis:\n{ctx.get('diagnosis', '(none)')}\n\n"
            f"Consolidator plan for this generation: {json.dumps(plan, default=str)}\n"
            f"Parked ideas (measured before, retest only with a reason): {json.dumps(ctx.get('parked', []), default=str)}\n"
            f"Choose exactly k = {ctx['k']} candidates, in priority order. One generation slot belongs to an Explorer role "
            f"whose target_component you cannot see: if it collides with one of yours, that one is dropped and your last "
            f"candidate takes the slot — so make the last one a genuine reserve, not a throwaway."
            + ("\n\nYour previous answer had NO FREE-SLOT candidate (ADR-0014): put exactly one candidate FIRST with type "
               "\"improve\" and a card from the untried / not-in-champion-stack lists in the state, then the rest."
               if ctx.get('free_slot_violation') else ''))

def user_probe(ctx, selection):
    return (f"Candidate to probe:\n{json.dumps({k: selection.get(k) for k in ('type', 'card', 'target_component', 'hypothesis', 'new_signal', 'cheapest_test') if selection.get(k) is not None}, indent=1, default=str)}\n\n"
            f"Champion input set (columns / side-table fields its script already reads): {', '.join(ctx.get('champion_inputs') or []) or 'unknown'}\n"
            f"Data files under --data-dir: train.csv (user_id, video_id, date, hourmin, time_ms, tab, duration_ms, is_rand, the outcome "
            f"columns long_view, is_click, is_like, is_follow, is_comment, is_forward, is_hate, play_time_ms, ...), valid.csv (row_id, user_id, "
            f"video_id, date, hourmin, time_ms, tab, duration_ms, is_rand — NO outcome columns), video_features_basic.csv (video_id, author_id, "
            f"video_type, upload_dt, upload_type, visible_status, video_duration, server_width, server_height, music_id, music_type, tag), "
            f"user_features.csv (user_id, user_active_degree, ..., onehot_feat0..17).\n"
            + (("The full method card:\n```card\n" + (C.KB / 'methods' / f"{selection['card']}.md").read_text() + "\n```\n")
               if selection.get('card') and (C.KB / 'methods' / f"{selection['card']}.md").exists() else '')
            + "Write the probe script.")

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
    card = (C.KB / 'methods' / f"{selection.get('card')}.md") if selection.get('card') else None
    if card and card.exists():
        s += f"\nThe full method card for this candidate (recipe, risks, prior measurements):\n```card\n{card.read_text()}\n```\n"
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

refresh_calibration()      # ADR-0018: the Selector's calibration sentence comes from the ledger (after every def above)
