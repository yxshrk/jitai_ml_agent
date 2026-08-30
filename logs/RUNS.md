# Autonomous runs — overnight 28 Aug

All runs: 0 interventions, official convergence rule, validation only.
Champion candidate: run_real_01 node_002 (0.6042). Manual 3-seed best: zoo/best.py 0.6039±0.0010.

| run | directive | stop | iters | best valid primary | delta | wall | LLM cost |
|---|---|---|---|---|---|---|---|
| 01 | default tiers | converged | 5 | **0.6042** | **+0.0026** | 23.5m | $0.90 |
| 02 | + overfit insight | converged | 5 | 0.6040 | +0.0024 | 14.2m | $0.57 |
| 03 | watch-time (ignored: policy bug, fixed) | converged | 3 | 0.6033 | +0.0015 | 11.6m | $0.66 |
| 04 | watch-time themes (FM base) | converged | 3 | 0.6026 | +0.0008 | 15.6m | $0.63 |
| 05 | compound DCN+themes | converged | 3 | 0.6033 | +0.0015 | 12.8m | $0.68 |

## Sunday 30 Aug — fan-out doctrine evolution (see git history for harness changes between runs)
| family | runs | best | note |
|---|---|---|---|
| unseeded atomic (05-11) | 7 | 0.60424 (06) | ensemble + duration-heads discoveries |
| synergy cards (12-15) | 4 | 0.60404 | first package proposals |
| effort grid (16-22,27,28) | 7 | 0.60319 | high/xhigh: measured FAILURE (truncation) |
| dial-sweep wave (23-26) | 4 | 0.60468 (25) | first in-node search + ensemble |
| chains (c1,c2,l1,25a,25b) | 5 | 0.60468 | ceiling confirmed; philosophy retired |
| bigclock wave (01-07) | 7 | **0.60558 (07) = DESIGNATED** | dial sweep cleared epsilon -> ensemble-design close |
| deep/anthropic/multidraft | 6 | 0.60462 (g1) | sonnet thinking-starvation measured |
| max wave + experiments | 12 | in flight | honest-clock depth, 1K deep runs, clean A/B |
All journals in logs/run_*/; cross-machine copies rsynced back on completion.

## Late harvest (31 Aug)
| run | machine | stop | iters | best | note |
|---|---|---|---|---|---|
| qb_b | coral | converged | 4 | 0.60466 | decayed-positive sampling; below champion, no re-designation |
| novel_r1 | ruby | max_hours | 5 | 0.60447 | pair-kernel line on ruby; below champion, no re-designation |
| final_f1 | ruby | max_hours | 4 | 0.60403 | dial-jitter member-bank close (GPU); below champion, no re-designation |
| clean_c1 | coral | KILLED n=2 | - | - | discarded: mid-run METHODS.md dedupe sync = intervention taint; relaunched as clean_c2 with fixed library |
| combo_r1 | ruby | converged | 4 | 0.60496 | SEEDED composition test: pair-kernel+frozen-stack = +0.0003 over frozen alone — mechanisms barely stack; disclosed experiment, not designation-eligible per policy |
| omega_1k | ruby | converged | 8 | 0.66892 | 1K BREAKTHROUGH: causal session features (+0.0192); independent eval exact-match; fresh-seed replication in flight; re-designation pending |
| clean_c2 | coral | converged | 3 | 0.60335 | clean cards-only run, safe path, no threat |
| final_s1 | coral | converged | 3 | 0.60184 | 3 straight rejections incl. session-time-features +0.0002 on Pure (weak-transfer confirmed); gated variant untried |
| final_s3 | coral | converged | 4 | 0.60396 | full-library lottery ticket; NOTE node_003 gated-session composite measured 0.6044 but grey-zone rejected — reconcile in morning |
| omega1k audits | ruby | complete | - | 0.6766/0.6762/0.6652 | replication + shuffle: breakthrough confirmed |
| clean_r3 | ruby | converged | 3 | 0.60381 | GPU clean run; session +0.0001 and gauge-bce flat from dial-swept base — mechanisms overlap confirmed |
| omega CSV | ruby | built+checked | - | ens-val 0.68020 | evidence/test_submission_1k_omega.csv (4.13M rows, checker OK); old max_1k_c CSV preserved pending decision |
| final_s2 | coral | converged | 4 | 0.60499 | best of final wave: gauge-bce -> hetero-ensemble close (+0.0010, first measured win for that card); gated-session +0.0006 sub-eps |
