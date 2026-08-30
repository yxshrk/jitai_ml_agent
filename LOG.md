# Project log — changes, decisions, results

Chronological. Decisions are written up in `kb/adr/`; agent-run journals will live in `runs/<run_id>/`.

## 2026-08-30
- Read the problem statement (`docs`) and the starter kit; walked through the task, metrics, data, and baseline.
- **Decision:** independent attempt — the teammate's `origin/mle-agent` branch is off-limits (ADR-0001).
- Translated all starter-kit comments, docstrings, and user-facing strings to English; `README.md` → English with
  the original kept as `README.zh.md`. AST check against the pristine kit: `evaluate.py` and `data.py` logic
  identical; `baseline.py`, `ablation_features.py`, `submit.py` differ only in text strings.
- Verified the metric spec: `docs` line 43 (NDCG@10 / Recall@50 / click) is stale — see `kb/spec/corrections.md`.
  `long_view` definition confirmed from kuairand.com: `play_time_ms >= min(duration_ms, 18 s)`.
- Literature: 19 PDFs into `kb/literature/` with a reading guide. Missing: MMoE, PLE (ACM paywall).
- Environment: `.venv` (Python 3.9.6) + numpy 2.0.2. KuaiRand-Pure downloaded from Zenodo into
  `kuairand-starter-kit/KuaiRand-Pure/` (gitignored, 195 MB).
- **Baseline reproduction — Task Requirement #1 done:**

  | model | valid primary | test primary | published (valid / test) |
  |---|---|---|---|
  | random, seed 0 | 0.4827 | 0.4757 | 0.4834 / 0.4753 |
  | item popularity | 0.5807 | 0.5715 | 0.5807 / 0.5715 (exact) |
  | FM, seed 0 | 0.6015 (GAUC 0.6671, nDCG@5 0.5358) | 0.5953 | 0.6016 / 0.5946 ± 0.0008 |

  FM learning curve (valid primary): 0.5869 → 0.5956 → 0.5993 → 0.5994 → 0.6010 → 0.6006 → **0.6015 (ep 7)** →
  0.6012 → 0.6007 → 0.5996 → 0.5990, early stop at ep 11; training loss falls monotonically → overfits after
  ~7 epochs. Runtime 16 s. (The organizer script prints test scores by design; our harness will not — ADR-0005.)
- Wrote the KB spec layer (`kb/spec/`), ADR-0001…0007, this log.
- Browser: the Lark doc is reachable in Chrome (guest mode), last updated Aug 28 — the `docs` paste is current.
  Its body is not scriptable, so general sections (Key Dates, submission process) still need pasting.
  **Registration deadline 1 Sep 2026 12:00** (form + Devpost).
- **EDA** (`kb/data/eda.py` → `eda_report.md`; interpretation in `facts.md`; label diagnostics `label_check.py`).
  Headline facts: closed catalogue (0 % unseen videos/authors in valid/test); valid users have median 35 train
  rows (not hundreds); `long_view` rule verified (exceptions: `duration_ms = 0` ⇒ label 0, 1.9 % of rows);
  **short videos are the hard case** (positive rate 0.27–0.28 under 20 s vs 0.38 at 90–117 s) — an earlier guess
  overturned; `tab` dominates (tab 0 rate 0.04 vs tab 4 0.49); strong volume drift (280 K rows/day on 04-11 → 20 K/day
  from 04-18, matching valid/test) and positive-rate drift (0.337 train → 0.313 valid → 0.29 by 04-28);
  `is_click` correlates 0.76 with the label; valid cohorts 30.3 / 11.9 / 57.8 %, test composition reproduced
  exactly (27.1 / 9.2 / 63.7, oracle 0.7289) — harness matches `evaluate.py` conventions. No rows on 2022-04-08.
- **Harness foundation built and verified.** `harness/config.py`, `data_access.py` (workspace builder = the firewall:
  `workspace/data/{train,valid}.csv` + side tables, `private/test_features.csv` without labels; split sizes asserted),
  `referee.py` (subprocess + timeout, static forbidden-path check, prediction alignment validation, official
  `evaluate.py` scoring, ε-acceptance, convergence counter), `journal.py` (JSONL + diffs + markdown), the script
  contract (`workspace/CONTRACT.md`), and the baseline ported to it (`harness/seeds/node_000_fm.py`).
  Self-check: smoke run 6 s (1 epoch, primary 0.5869); full run 14 s → **valid primary 0.6015** = `baseline.py` seed 0
  exactly; a deliberately corrupted prediction row is rejected ("row 4 misaligned"). Two bugs fixed on the way
  (relative script path; workspace not on PYTHONPATH).
- Literature survey of autonomous-ML-agent architectures (AIDE, MLE-STAR, R&D-Agent, ML-Master, AIRA, AI-Scientist-v2)
  → `kb/literature/agent-design-notes.md`; harness architecture proposed as ADR-0008.
- Tooling: `anthropic` 0.125.0 and `pytest` installed in `.venv`. No API credential is configured yet
  (`ANTHROPIC_API_KEY` unset, no `ant` CLI) — needed before the first live agent run.
- **Loop built and exercised** (ADR-0008/0009): `harness/prompts.py` (cached stable prefix + six role prompts),
  `harness/brain.py` (`Brain` interface; `FakeBrain` for offline tests; `AnthropicBrain` with per-call token/cost
  metering, fenced-block parsing with one format-reminder retry, refusal/max_tokens handling), `harness/loop.py`
  (generations of k parallel branches; implementer→static firewall→critic with up to two revise rounds; smoke test
  with one fixer attempt; full runs in a thread pool; referee acceptance ≥ ε with grey-zone 3-seed confirmation;
  per-generation champion/convergence; consolidator plan; parked ideas; resumable `state.json`; a crashed
  generation counts as non-improving and never kills the run), `harness/submit.py` (final CSV + official
  `--check`, no test metric ever computed), `harness/cli.py`, tests (6 unit + 1 end-to-end).
  End-to-end fake generation on the real data: node_000 0.6015; k=8 → 0.6000 (−0.0014), lr 2e-3 → 0.6007
  (−0.0007), a deliberately broken script → smoke failure → fixer → reverted to parent → 0.6015 (Δ 0);
  all rejected, streak 1; 45 s for the generation (three full runs in parallel). Journal, diffs, summary written.
- Git: commits are attributed to Yash only (`.claude/settings.json` attribution block); repo-local identity set to
  his GitHub no-reply address; branch `yash-attempt` pushed.
- **Provider switch:** agents now run on OpenAI GPT-5.6 (`gpt-5.6-sol` for every role by default; `--cheap-roles`
  puts diagnose/critique/fix/consolidate on `gpt-5.6-terra`). `brain.py` refactored into a shared `LLMBrain` with
  `OpenAIBrain` (Responses API, reasoning effort per role, automatic prompt caching, per-call metering incl.
  reasoning tokens) and `AnthropicBrain` kept as an alternative. Request/usage shape verified with one tiny live
  call on `gpt-5.6-terra`. `.env` (OPENAI_API_KEY) is git-ignored and untracked.
- **Evaluation clarity:** `referee.score` adds `ndcg5_disc` (nDCG@5 among discriminative users, the sharper
  diagnostic); `loop.designate_final` re-ranks the top-3 valid nodes by 3-seed mean (AIRA's robust selection);
  `--iteration-unit node|generation` decides what the 50 cap counts (ADR-0006 updated). Prompts refined
  (calibration guidance for the Selector, runtime/vectorisation guidance for the Implementer, stricter contract
  checks and "veto only for leakage" for the Critic). `kb/ARCHITECTURE.md` written as the reference.
- **Method cards (KB layer 2):** schema in `kb/methods/README.md`; 13 cards written from the literature + facts
  (loss: BPR pairwise, listwise softmax, LambdaRank pairs, censored watch-time; features: duration-unknown flag,
  fine duration + tab cross; data-weighting: recency; aux-targets: is_click; history: user aggregates; model: DCN
  cross head; regularization: embedding dropout/L2; training-schedule: lr decay + half-epoch checkpoints;
  ensembling: seed average). Each has checkable `applies_when` (cites numbered facts), an honest `expected_delta`
  calibrated to the 0.002 floor, a `how_to` written against `node_000_fm.py`, and `composes_with` for merges.
  `kb/methods/validate.py` checks fields, target components, cross-references and length: 13 cards, 0 problems.
  Paper text extracted to `kb/literature/text/` (`extract_text.py`, pypdf; ignored by git). Stable prompt prefix
  is now ~11.8 K tokens (served from cache on every call).
- **First live run (`live_01`, GPT-5.6-sol, k = 3, 2 generations, $2.70, 16 min, 370 K in / 64 K out tokens):**
  node_000 baseline 0.6015 → node_001 within-user BPR **0.6036 (+0.0022, accepted)**; recency +0.0005 (grey, rejected
  on seeds), duration-unknown flag −0.0003; generation 2 on the BPR champion: recency retest +0.0003 (seed-confirmed
  neutral), LambdaRank pairs −0.0005, history aggregates −0.0014; streak 1 at the generation cap. Final designation
  (3-seed re-rank of the top-3) picked node_004 (BPR + recency, mean 0.60319) over node_001 (0.60316) — a tie within
  noise. The Selector chose the organizers' lead #1 first; the Consolidator applied the ADR-0004 retest rule unprompted.
- **Reproducibility check of +0.0022:** deterministic given the seed (node_000 reproduces exactly); across seeds
  {0, 1, 2} the BPR node scores [0.60365, 0.60312, 0.60272] vs baseline [0.60147, 0.60176, 0.60109] — every BPR seed
  beats every baseline seed, but the mean gain is **+0.0017**, not 0.0022: the winner's curse of picking the best
  single-seed branch. Every top-3 node shrank on re-seeding.
- **Acceptance v2 (ADR-0010):** every positive-delta candidate is confirmed with 2 extra seeds; accepted iff the
  seed-mean gain ≥ 0.001 and ≥ 2.5 standard errors. `p_accept` removed from the Selector prompt (an LLM's stated
  probability is not calibrated; expected-vs-realised Δ already measures calibration). Two cards added from the
  run's lessons: `loss-bpr-hard-negatives`, `training-schedule-weight-averaging` (15 cards, validator clean).
- Prompt calibration evidence: expected Δ 0.006 / 0.004 / 0.003 / 0.003 / 0.004 vs realised +0.0022 / +0.0005 /
  −0.0003 / −0.0005 / −0.0014; diffs 433 / 452 / 275 / 112 / 792 lines (the Implementer rewrote files) → prompts
  sharpened and a 200-line diff guard added (commit 3378f1b).
- **Full autonomous run `live_02`** (GPT-5.6-sol, k = 3, sharpened prompts, seed-confirmed acceptance):
  **converged by the official rule after 5 generations / 16 nodes, 29.7 min, $4.25**, 51 LLM calls, 866 K input
  tokens (682 K served from cache) / 100 K output, 0 interventions. Accepted: node_001 within-user BPR
  (+0.0016 single seed, +0.0016 over three seeds, t = 8.2). Rejected on seed confirmation despite positive single
  seeds: recency (+0.0005→+0.0004), hard negatives (+0.0006→+0.0001), L2 1e-5 (+0.0005→+0.0002), duration flag on
  BPR (+0.0000→+0.0002), fine duration + tab cross (+0.0002→−0.0001), censored watch-time aux (+0.0004→+0.0003,
  t 1.85), checkpoint averaging (+0.0002→+0.0001), 5-seed ensemble (+0.0006→**+0.0009, t 5.4** — below the 0.001
  minimum effect). Negative: history aggregates −0.0001, LambdaRank −0.0010, EMA averaging −0.0027, is_click head
  −0.0003. Diffs 4–70 lines (one 29-line EMA). **Final designation** re-ranked the top-3 by 3-seed mean and chose
  **node_015 (5-seed ensemble of the BPR champion): seeds 0.60367 / 0.60417 / 0.60395, mean 0.60393 = +0.0025 over
  the baseline's 3-seed mean 0.60144.** `submission.csv` written for node_015 and passed the organizers' `--check`
  (170,588 rows). Cards distilled (15 cards; BPR `alive`, everything else `dead_under` the FM or FM+BPR stack).
- Observation for the next iteration of the harness: the 0.001 minimum-effect floor rejected a t = 5.4 improvement
  (the ensemble); a floor of 0.0005 with the same t-test would have accepted it. The DCN cross head was planned for
  generation 6 but never ran — the only card family left untried after two runs.
