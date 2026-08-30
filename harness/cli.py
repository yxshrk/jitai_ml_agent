"""CLI: python -m harness.cli run|submit|report ..."""
from __future__ import annotations
import argparse, json, time
from . import config as C

def main():
    ap = argparse.ArgumentParser(prog='harness')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='start or resume an autonomous run')
    r.add_argument('--run-id', default=time.strftime('run_%Y%m%d-%H%M%S'))
    r.add_argument('--brain', choices=['openai', 'anthropic', 'fake'], default='openai')
    r.add_argument('--model', default=None, help='model id for every role (default gpt-5.6-sol / claude-opus-5)')
    r.add_argument('--cheap-roles', action='store_true', help='diagnose/critique/fix/consolidate on gpt-5.6-terra')
    r.add_argument('--iteration-unit', choices=['node', 'generation'], default='node', help='what the 50-iteration cap counts (ADR-0006)')
    r.add_argument('--no-final-reseed', action='store_true', help='skip the multi-seed re-ranking of the top-3 at the end')
    r.add_argument('--k', type=int, default=5, help='branches in generation 1 (incl. the Explorer slot)')
    r.add_argument('--k-later', type=int, default=3, help='branches from generation 2 on; grows back toward --k for planned merges/retests')
    r.add_argument('--no-wildcard', action='store_true', help='all k slots from the Selector (no Explorer slot)')
    r.add_argument('--max-generations', type=int, default=None)
    r.add_argument('--max-nodes', type=int, default=C.MAX_ITERS)
    r.add_argument('--budget-usd', type=float, default=25.0)
    r.add_argument('--seed', type=int, default=C.DEFAULT_SEED)
    r.add_argument('--no-parallel', action='store_true')
    r.add_argument('--no-confirm', action='store_true', help='skip the multi-seed confirmation of positive deltas (not recommended)')
    r.add_argument('--no-librarian', action='store_true', help='never call the web-searching Librarian during the run (ADR-0013)')
    r.add_argument('--convergence', choices=['confirmed', 'official'], default='confirmed',
                   help='confirmed = stop after N generations without a seed-confirmed champion change >= RESET_MIN_GAIN (ADR-0012); official = the literal single-seed eps rule')
    r.add_argument('--no-distill', action='store_true', help='do not fold the journal into the cards when the run ends')
    r.add_argument('--no-screen', action='store_true', help='skip the feature screen (ADR-0015): build every feature candidate unmeasured')
    r.add_argument('--no-campaigns', action='store_true', help='no family campaigns: every generation is a breadth generation (pre-ADR-0016 behaviour)')
    s = sub.add_parser('submit', help='write the test submission for a node of a run')
    s.add_argument('--run-id', required=True); s.add_argument('--node', type=int, required=True)
    s.add_argument('--out', default='submission.csv')
    d = sub.add_parser('distill', help='fold a run journal back into the method cards (cross-run memory) and archive its wildcards as new cards')
    d.add_argument('--run-id', required=True); d.add_argument('--no-archive', action='store_true', help='measurements only; no Archivist calls')
    lb = sub.add_parser('librarian', help='add n web-searched cards to the menu (ADR-0013)')
    lb.add_argument('--run-id', default=None, help='run whose journal the Librarian should read'); lb.add_argument('--n', type=int, default=2)
    p = sub.add_parser('report', help='print a run summary')
    p.add_argument('--run-id', required=True)
    a = ap.parse_args()

    if a.cmd == 'run':
        from .loop import Loop
        if a.brain == 'fake':
            from tests.fake_generations import fake_generations
            from .brain import FakeBrain
            brain = FakeBrain(fake_generations())
        elif a.brain == 'openai':
            from .brain import OpenAIBrain
            models = {r: a.model for r in OpenAIBrain.DEFAULT_MODELS} if a.model else {}
            if a.cheap_roles:
                models.update({r: 'gpt-5.6-terra' for r in ('diagnose', 'critique', 'fix', 'consolidate')})
            brain = OpenAIBrain(models=models, budget_usd=a.budget_usd)
        else:
            from .brain import AnthropicBrain
            brain = AnthropicBrain(models={r: a.model for r in AnthropicBrain.DEFAULT_MODELS} if a.model else None, budget_usd=a.budget_usd)
        loop = Loop(a.run_id, brain, k=a.k, max_nodes=a.max_nodes, max_generations=a.max_generations, seed=a.seed,
                    parallel=not a.no_parallel, confirm_seeds=not a.no_confirm, final_reseed=not a.no_final_reseed,
                    iteration_unit=a.iteration_unit, wildcard=not a.no_wildcard, librarian=not a.no_librarian, auto_distill=not a.no_distill, convergence=a.convergence, k_later=a.k_later,
                    screen=not a.no_screen, campaigns=not a.no_campaigns)
        print(json.dumps(loop.run(), indent=1, default=str))
    elif a.cmd == 'submit':
        from .submit import make_submission
        print(json.dumps(make_submission(a.run_id, a.node, a.out), indent=1))
    elif a.cmd == 'distill':
        from .distill import distill, archive
        distill(a.run_id)
        if not a.no_archive:
            from .brain import OpenAIBrain
            print('archived as cards:', archive(a.run_id, OpenAIBrain()))
    elif a.cmd == 'librarian':
        from .brain import OpenAIBrain
        from .librarian import run_librarian
        print('new cards:', run_librarian(OpenAIBrain(), n=a.n, run_id=a.run_id))
    elif a.cmd == 'report':
        print((C.RUNS / a.run_id / 'journal.md').read_text())

if __name__ == '__main__':
    main()
