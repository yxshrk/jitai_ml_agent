"""CLI: python -m harness.cli run|submit|report ..."""
from __future__ import annotations
import argparse, json, time
from . import config as C

def main():
    ap = argparse.ArgumentParser(prog='harness')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='start or resume an autonomous run')
    r.add_argument('--run-id', default=time.strftime('run_%Y%m%d-%H%M%S'))
    r.add_argument('--brain', choices=['anthropic', 'fake'], default='anthropic')
    r.add_argument('--k', type=int, default=3)
    r.add_argument('--max-generations', type=int, default=None)
    r.add_argument('--max-nodes', type=int, default=C.MAX_ITERS)
    r.add_argument('--budget-usd', type=float, default=25.0)
    r.add_argument('--seed', type=int, default=C.DEFAULT_SEED)
    r.add_argument('--no-parallel', action='store_true')
    r.add_argument('--no-reseed', action='store_true', help='skip the grey-zone multi-seed confirmation')
    s = sub.add_parser('submit', help='write the test submission for a node of a run')
    s.add_argument('--run-id', required=True); s.add_argument('--node', type=int, required=True)
    s.add_argument('--out', default='submission.csv')
    p = sub.add_parser('report', help='print a run summary')
    p.add_argument('--run-id', required=True)
    a = ap.parse_args()

    if a.cmd == 'run':
        from .loop import Loop
        if a.brain == 'fake':
            from tests.fake_generations import fake_generations
            from .brain import FakeBrain
            brain = FakeBrain(fake_generations())
        else:
            from .brain import AnthropicBrain
            brain = AnthropicBrain(budget_usd=a.budget_usd)
        loop = Loop(a.run_id, brain, k=a.k, max_nodes=a.max_nodes, max_generations=a.max_generations, seed=a.seed,
                    parallel=not a.no_parallel, reseed_grey=not a.no_reseed)
        print(json.dumps(loop.run(), indent=1, default=str))
    elif a.cmd == 'submit':
        from .submit import make_submission
        print(json.dumps(make_submission(a.run_id, a.node, a.out), indent=1))
    elif a.cmd == 'report':
        print((C.RUNS / a.run_id / 'journal.md').read_text())

if __name__ == '__main__':
    main()
