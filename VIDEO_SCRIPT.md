# Demo video script (~2 min) — record Sunday

Format: screen capture + voiceover. Rehearse once; speak plainly; no music needed.

## Shot 1 — the problem (0:00–0:15)
Screen: PROBLEM_STATEMENTS.md Track 2 heading, or the field-guide artifact.
Say: "Track 2 asks for an autonomous ML research agent: given the KuaiRand-Pure
benchmark, it must reproduce the official baseline, then improve it — on its own.
We built that agent, and ran it with zero human interventions."

## Shot 2 — one live iteration (0:15–0:55) ← the core shot
Screen: terminal, launch a run:
  uv run python -m harness.cli run --data-dir data/real_ws \
    --baseline-script zoo/baseline_ws.py --seed-scripts zoo/frozen_stack.py \
    --max-iters 6 --sigma 0.0008 --context-mode compact --run-dir logs/run_demo
Then open logs/run_demo/journal.jsonl (or tail it live) and point at one record.
Say: "Each iteration is a full research step: the agent reads its own learning
curve, diagnoses — here, overfitting after epoch three — selects a method from a
cited library, implements it as runnable code, trains, and judges the result
against a noise floor. Every decision is logged: hypothesis, diff, metrics,
even its predicted gain versus what actually happened."

## Shot 3 — the evidence (0:55–1:30)
Screen: logs/run_official_01/report/trajectory.png, then RUNLOG.md, then PRACTICES.md.
Say: "Across all official runs the agent needed zero interventions — the count is
machine-logged. Behind it sits our decision table: roughly 250 curated experiments
plus 3,600 automated trials. Every configuration choice in the final system has a
measured A/B behind it, including the negatives — we refuted sequence models and
watch-time losses for this task, with seeds."

## Shot 4 — results + close (1:30–2:00)
Screen: results.md table, then SUBMISSION_RECIPE.md.
Say: "Final result: the official baseline scores 0.6016 on validation; our
agent-designated five-seed ensemble reaches 0.60513 — and on KuaiRand-1K the
agent scaled its own ensemble to ten seeds, finishing at 0.63874. Test data was touched exactly once, for the final
submission — the leakage guard makes anything else structurally impossible.
The agent is the product; the score is its evidence. Thanks."

## Recording notes
- Run the Shot-2 demo command BEFORE recording so a finished run_demo exists as
  backup footage if the live one is slow.
- Terminal font large; hide personal notifications; 1080p or higher.
- Keep total under 2:30; judges may watch nothing else — the video must stand alone.
- No third-party trademarks/music (rules requirement).
