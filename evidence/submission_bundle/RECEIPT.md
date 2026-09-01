# Submission bundle (team jit.ai, TechJam 2026 Track 2)

- kuairand_pure_submission.csv — REQUIRED benchmark. 170,588 rows, schema row_id,user_id,video_id,score.
  Built train-only from the designated run run_bigclock_07 (validation-best at convergence: primary 0.605575).
  Test labels never read. Validated with the starter-kit checker semantics (evidence/submission.py --check).
- kuairand_1k_submission.csv.gz — BONUS benchmark (gunzip before use). 4,132,081 rows, same schema.
  Faithful A-form replay of run_omega_1k's designated recipe (recorded validation primary 0.66892; replay
  caveat disclosed in SUBMISSION_RECIPE.md). Validated the same way.
- SHA256SUMS.txt — checksums of both files.
Repo: https://github.com/yxshrk/jitai_ml_agent (branch clean-agent). Full receipts: evidence/RESULTS_AND_RESOURCES.md.
