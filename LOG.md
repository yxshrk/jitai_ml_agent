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

## 2026-08-30 (later) — live_03/live_04, the code review, ADR-0012 and ADR-0013
- **Harness v3 (ADR-0011):** k = 5 with one Explorer wildcard slot; Selector/Implementer/Explorer at `xhigh`; minimum
  effect 0.0005. `live_03` stopped after one generation: the Explorer had never been shown the output schema (four
  failed calls) and the Critic vetoed a correct L2 script for "too small a gain" — the Critic is now confined to code
  review and the hypothesis is fixed for the Implementer. Cards gained an aggregated status across stacks, a
  verdict line and a status table at the top of the Selector's menu (commit 592dae9).
- **`live_04`** (k = 5, wildcard, xhigh): generation 1 accepted four nodes — the wildcard "field-aware FM embeddings"
  (+0.0016 single, +0.0012 seed-mean, t 5.2 → champion), BPR (+0.0011), history aggregates (+0.0010), L2 (+0.0009);
  generation 2 (merges on the wildcard) flat; generation 3 flat except a rank-average ensemble at +0.0006, t 2.43
  (missed the 2.5 bar; its single seed 0.6036 reset the streak under the old rule); **generation 4: node_015, a
  5-seed rank-average ensemble of the two lineages, +0.0014 single / +0.0017 seed-mean (t 7.65), valid 0.6045 →
  champion.** Selector and Explorer now run concurrently; the Selector lists k candidates with a reserve for a
  collision with the wildcard's component (each live_04 generation had lost one branch to that).
- **Code review relayed by Yash (ADR-0012)** — verified against the run artefacts and fixed: (1) convergence tracked
  the single-seed best of *any* node while acceptance used seed-means: four accepted nodes logged "no improvement",
  a rejected node's lucky seed logged "improved" → convergence now tracks the champion's seed-mean with
  early-stopping semantics (`referee.Convergence`, unit-tested); (2) the Critic never saw the parent or the diff and
  the Explorer prompt hard-coded "BPR is the champion" (true in live_02) → the correct field-aware edit was sent
  back twice until BPR was added; node_001's +0.0012 is mostly BPR (node_002 alone: +0.0011); node_006 and node_008
  produced predictions byte-identical to node_001 (md5 4ff8aa3e…) → Critic receives the unified diff + the parent's
  actual stack, Implementer is told the stack, no-op detector by prediction hash; (3) champion = accepted node with
  the best seed-mean gain, not the best single seed; (4) one seed cache never cleared (node_001's seeds had been
  recomputed — timestamps 20:33 vs 20:39); (5) prompts still said "≥ 0.002 to beat the champion" → rule text
  generated from `config.py`; (6) `stdev` not `pstdev`; (7) resume-safe wall clock, Diagnostician view uncut;
  (8) tests for `Convergence`, `pick_champion`, `confirm_stats`, the seed-cache migration, `summarize`. The
  paired-test suggestion was withdrawn by the reviewer (seeds are not paired across scripts) and not adopted.
- **Evolving KB (ADR-0013):** `Journal.digest()` puts every node with its diff into a generation-stable cached
  block (≈ 23 K tokens at 14 nodes); `kb/spec/foundations.md` (metric invariances, loss vs metric, noise, dynamics)
  joins the prefix; the **Archivist** turns wildcards into cards — first result `model-field-aware-fm-embeddings`
  with expected Δ [0, 0.0001] and the honest note that the bundled BPR explains the gain; the **Librarian** (OpenAI
  `web_search`) adds untried cards after flat generations (≤ 2× per run) or via `cli librarian`; card status
  `alive` → `proven — accepted on [stack]`, with the champion's actual stack stated in every call.

## 2026-08-30 (evening) — live_04 finished; statistics, breadth and diagnostics revised with the review session
- **live_04 final:** converged under its (pre-ADR-0012) rule after 6 generations / 28 nodes, 97 min, $14.77, 99 LLM
  calls, 1.92 M input tokens (1.47 M of them cached) / 393 K output, 0 interventions. Champion node_015 (5-seed
  rank-average ensemble, 0.6 field-aware(+BPR) / 0.4 BPR lineage): valid 0.6045, seeds 0.60447 / 0.60450 / 0.60399,
  **3-seed mean 0.6043 = +0.0029 over the baseline's mean 0.60144**; +0.0017 over its parent at t 7.65 (z ≈ 6.8 under
  the new test). Generations 5–6 were flat (LightGCN wildcard, gated rank fusion, watch-survival reranker, field-aware
  author feature, DCN-on-BPR merge, LambdaRank, user-balanced pairs, reciprocal-rank fusion: −0.0029 to +0.0001).
  The old process designated node_026 (mean 0.6043254 vs node_015's 0.6043209 — a 0.0000045 gap on a 0.00014 SE; an
  unaccepted near-no-op variant of node_015); the designation was re-run offline under the ADR-0012 code with the
  new tie-break (within one SE, prefer the accepted lineage) → **node_015 designated; `submission.csv` written for it
  and passed the organizers' `--check`.** The literal ε rule had not converged when the run ended (node_012's lucky
  seed had reset it).
- **Decisions with Yash (ADR-0012 revised):** (a) the run-journal block goes only to the planning roles, old rejected
  diffs are stubs, cached vs uncached tokens reported separately; (b) convergence: the streak resets on a ≥ 0.001
  cumulative rise of the champion's fresh-seed mean since the last reset (ε rescaled to the seed-mean's noise — every
  faithful reading of ε stops live_04 at generation 3, before node_015), the literal rule tracked and reported with the
  node it would have submitted, `--convergence official` available; (c) acceptance: pooled-variance z-test on three
  FRESH seeds (seed 0 excluded as the selected screen), z ≥ 3 and gain ≥ 0.0005, two more seeds when 2 ≤ z < 3,
  node-own SD when clearly unstable — the 3-vs-3 t-test at 2.5 had passed 3–6 % of null candidates (node_017 was
  the shape of one); (d) breadth: k = 5 in generation 1 then 3 + planned slots, Selector slots are `deepen` from
  generation 2, Explorer kept; (e) per-tab / per-duration-band GAUC+nDCG from the referee for the Diagnostician
  (live_04's champion: weakest on tab 1 GAUC 0.62 and dur > 180 s GAUC 0.63).
- **Review session (second Claude session, read-only) findings adopted:** `conv_ref` was never initialised (generation
  1's gain never counted) → seeded with the baseline; the sigma guard; designation tie-break; facts §10 (user behaviour
  around exposures: series continuation is label-conditioned and unobservable at scoring time; label-free same-author
  consecutive exposure is negative, 0.142, still 0.268 vs 0.384 within tab 1; creator affinity real but 96.6 % of valid
  rows are unseen user × author pairs; same-video repeat exposure negative) measured on train and reproduced by
  `eda.py`; cards `history-same-author-run-features` (new) and a tab-aware retest note on `history-repeat-exposure-fatigue`.
- KB after live_04: 23 cards (6 archived from wildcards/un-carded candidates, 1 from the Librarian's web search);
  `kb/spec/foundations.md` in the prefix.
- **live_05** (first run on the corrected harness; k = 5 → 3, deepen slots, z-test on fresh seeds, cumulative rule,
  per-group breakdown, Librarian, auto-distill): **converged after 4 generations / 16 nodes, 34.5 min, $9.69**, 54
  LLM calls, 1.80 M input tokens (0.95 M cached) / 165 K output, 0 interventions. Generation 1 (k = 5): BPR
  +0.0017 on fresh seeds (z 6.5) → champion; L2 +0.0010 (z 3.5, adaptive) accepted; history aggregates +0.0006 at
  z 1.9 and the seed-average ensemble +0.0008 at z 2.85 rejected — borderline cases the old t-test would have
  passed. Generations 2–4 (k = 3–4, deepen slots on the BPR champion) were flat: BPR+L2 merge +0.0003, lr decay
  +0.0003/+0.0004, two-stream BPR and field-weighted FM wildcards, duration-cohort BPR −0.0020, multiseed rank blend
  +0.0005 at z 1.8, timeSVD drift head ≈ 0. The 5-member blend deepen (node_014) was built on the wrong parent — the
  Selector wrote `"parent": "node_012"` and `_resolve_parents` only accepted integers — and the Critic rejected it
  three times (198 K tokens); fixed in `c78c1fe` with a test. The Librarian's first call lost both cards to mutual
  `composes_with` references (fixed, `5697942`); its second added `regularization-adversarial-personalized-ranking`
  and `model-first-order-exposure-transition-fm`. Designated node_012 (fresh-seed mean 0.6037; the champion node_002
  0.6031 is now always a designation candidate). The literal ε rule converged at the same generation and would have
  submitted node_002. **Submission unchanged: live_04's node_015 (fresh-seed mean 0.6043) remains the best model.**
- Token accounting corrected: `Usage.tokens_in` is total input (cached included); live_04 = 1.92 M input (1.47 M
  cached), not 3.39 M as first written. Per-role in live_05: the Diagnostician (first call of each generation) and
  the Librarian (called after ~10 min of training, when the provider cache has expired) pay the uncached prefix.

