# Wake-up brief (1 Sep, prepared ~04:45 while you slept)

DEADLINE: 12:00 noon SGT — Devpost submission AND registration form both close then.
(People's Choice voting opens 1 Sep 15:00 — post-deadline, nothing to do now.)

## State when you read this
- Check the terminal for the latest f10b / c10 status lines (they were mid-run at
  writing; watchers report every event). Standing policy applied while you slept:
  designation switches ONLY if a run converged above 0.605575 cleanly (then I
  rebuilt the CSV and updated RESULTS/DEVPOST — the terminal will say so loudly);
  redispatch authorized if a finished run showed a better-score path.
- If nothing says otherwise: Pure = bigclock_07 0.605575
  (evidence/test_submission_pure.csv, 170,588 rows, re-validated), 1K = omega_1k
  0.66892 (evidence/test_submission_1k_faithful.csv, local/gitignored).

## DONE while you slept (all committed on clean-agent)
1. Full rules/brief/webinar re-read — compliance checklist below.
2. README: team contributions FILLED from branch evidence (Rohan/Aditya/Yashraj;
   see "your 3 clicks" for Avinash). Campaign totals refreshed.
3. DEVPOST.md: final-night section added (harness hardening + ceiling finding),
   campaign aggregate updated (~10.6M tokens, ~146 runs, $122 real spend).
4. RESULTS_AND_RESOURCES.md: same totals; designated-run numbers unchanged.
5. SCRIPT_ROHAN.md: optional 20s addendum beat at the bottom (use it or don't;
   V3 body untouched — the cheatsheet still matches beat-for-beat).
6. bigclock_07 forensically dissected record-by-record (clean); Pure CSV
   re-validated; post-mortem + memory/doctrine/docs audits all complete.

## YOUR steps on waking (in order)
1. Read the terminal status; if a designation changed, the docs are already
   regenerated — just skim them.
2. RECORD: `cd site && python3 -m http.server 8642`, open
   http://localhost:8642/present.html, follow site/RECORDING_CHEATSHEET.md
   (28 keypresses, V3 narration; ~3 min video; webinar ruling: video optional
   but recommended).
3. Three clicks/edits only you can do:
   a. README: fill Avinash's one contribution line (marked {{fill:...}}).
   b. Repo public + GitHub Pages for site/ (Settings → Pages → deploy from
      branch clean-agent /site). OUTWARD — your call, but required deliverable
      is "public repo".
   c. Devpost: paste evidence/DEVPOST.md into the form, attach video link,
      submit BEFORE 12:00. Registration form too if not already done.
4. Optional sanity: `uv run python evidence/submission.py` style check already
   re-run green; CSVs are in evidence/.

## Compliance checklist (verified against PROBLEM_STATEMENTS.md deliverables)
- [x] Devpost written description → evidence/DEVPOST.md (approach, tools, APIs,
      libraries, datasets all present)
- [x] Public repo w/ README: overview, setup, reproduce steps, limitations,
      team contributions (Avinash line pending) — public flip is step 3b
- [x] Run & iteration logs: per-iteration hypothesis/diff/metrics/recovery in
      every logs/run_*/journal.jsonl + RUNS.md + intervention counts (0)
- [x] Final submission CSVs in starter schema (Pure 170,588 rows validated;
      1K faithful A-form per SUBMISSION_RECIPE policy)
- [x] Results table + absolute deltas → RESULTS_AND_RESOURCES.md
- [x] Resource usage: designated runs AND full-campaign totals (tokens,
      wall-clock, iterations, GPU-hours) — both disclosed per the fairness call
- [x] Multiple-runs ruling honored: all runs disclosed, best designated
      (webinar: "submit best, show all logs")
- [x] Zero mid-run interventions under the official definition (behavior-
      changing actions only) — journaled booleans agree
- [ ] Video — YOUR recording (optional per webinar; report is already long)
- [ ] Repo public + Pages — step 3b
- [ ] Devpost form submitted — step 3c

## Post-deadline cleanup (do NOT do before 12:00)
Rotate OpenAI + Anthropic keys; delete KuaiRand data copies everywhere
(rules require); terminate cpupod from RunPod console (it's unreachable via
ssh — likely already stopped, verify in console); coral/ruby: delete
~/mle-agent data/real_ws copies.

## 09:20 update (you're awake and recording)
- Designation unchanged: bigclock_07. Tickets f16 (ruby) / f17 (coral) running with the
  fixed opener card; switch only if one converges >= 0.6058 by ~10:00.
- Record SCRIPT V5 (not V3/V4) against the refreshed present.html; cheatsheet matches V5.
- Public branch = clean-agent. Make it the default branch (or ask Claude to push it
  over main). Then Pages, then Devpost paste (evidence/DEVPOST.md) before 12:00.
- Avinash's README line still needs your words.
