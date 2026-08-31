# Recording cheat sheet: present.html vs Script Version 3 (docx, 31 Aug 22:59)

Open http://localhost:8642/present.html, window 1920x1080, browser UI hidden (full screen).
Keys: `→` / `space` = next beat, `←` = back one beat, `1`..`9` and `0` = jump to scene,
`Home` / `End` = first / last, `Esc` = overview (do not press while recording).
Every `→` below is one beat. Total: 27 presses from title to receipts.
Where a scene has beats, `←` steps back within the scene, so a fluff is recoverable.

| # | Scene | Key(s) | What appears | Narration (V3 docx) | Time |
|---|---|---|---|---|---|
| 1 | `#s-title` | `Home` (or `1`) | Kicker, jit.ai, headline "An agent that improves its own ranking model. 0.6016 -> 0.6056, zero mid-run interventions.", the four names | "Hi, we're team jit.ai, and this is our submission for TikTok TechJam 2026, Track 2." | 0:00 |
| 2 | `#s-run` | `→` | Run chart draws itself: 00 to 06, green accepted, red dead ends, amber X at 04, ends at 0.6056 over the 0.6016 baseline line | "We built an autonomous research agent that trains models to rank videos by how likely a user is to long-watch them. In one clean run, with zero mid-run interventions, it took the official baseline from 0.6016 to 0.6056. Let's get into it." | 0:08 |
| 3 | `#s-loop` | `→` then `→` x3 | Four stage cards; the lit card advances DIAGNOSE -> TREAT -> RETRAIN -> MEASURE, one key per word | "Every iteration has four main steps: diagnose, [→] treat, [→] retrain, [→] measure. The agent reads the last model's validation curve and names the failure mode, then picks a treatment aimed at it. The harness rewrites the script, retrains and measures, and that measurement feeds the next diagnosis." | 0:20 |
| 4 | `#s-arch` | `→` then `→` x3 | Diagram builds in four parts: (1) agent row only, (2) harness row with the workspace boundary, (3) journal and the accept / dead end arrow, (4) convergence rule and hidden test window | "Let's talk about the full system, though. The agent proposes; [→] the harness executes, scores [→] and journals. Every iteration is journaled: what it saw, hypothesised, changed, and whether it worked, so later iterations don't relearn it. [→ full diagram, hold]" | 0:33 |
| 5 | `#s-log` | `→` then `→` x7 | Replay: chart on the left grows one point per key, card on the right shows the journal entry. Order: ITER 00 baseline 0.6018 -> 01 dead end 0.6020 -> 02 dead end 0.6017 -> 03 accepted 0.6042 -> 04 VOID (amber card) -> 05 dead end 0.6030 -> 06 accepted 0.6056 -> STOP, CONVERGED (green band) | "Now the run we submitted. Six iterations, seventeen minutes, on a single CPU. It first reproduces the baseline at 0.6018. [→] Iteration one: ... 0.6020. Dead end. [→] Iteration two: ... 0.6017. Dead end. That's two strikes. [→] Iteration three: ... 0.6042. Accepted, strikes reset. [→] Iteration four: the proposal crashes. ... void and the loop carries on. [→] Iteration five: ... 0.6030. Dead end. [→] Iteration six: ... 0.6056. Accepted, but under 0.002, so third strike. [→] The convergence rule fires and the run stops itself. No one chose an experiment, edited a script, or restarted anything." | 0:55 |
| 6 | `#s-harness` | `→` | Six numbered cards in the script's order: 01 leakage guard, 02 acceptance gate, 03 method cards, 04 decision benches, 05 deterministic ensemble planning, 06 telemetry-gated diagnosis. Point with the mouse as each is named. | "The agent's judgment matters, but the harness matters just as much. Six things. One, a structural leakage guard ... Two, an acceptance gate ... Three, method cards. Exactly 42 ... about 250 cells across 139 completed runs. One card was invented by a run itself. Four, a decision bench ... Five, deterministic ensemble planning ... No LLM in the measurement path. Six, telemetry-gated diagnosis ..." | 1:47 |
| 7 | `#s-methods` | `→` | Card library pre-filtered to "ensemble": ensemble cards with citations and measured status (0.6047, 0.605575, untried) | No dedicated line. Show it under "Our first principle is the clean run." or skip with a second `→` if the take is tight. | 2:17 |
| 8 | `#s-principles` | `→` | Five principle cards: Clean runs, Grounded action, Noise awareness, Learning from failure, Bounded authority | "Our first principle is the clean run. We never learn the recipe ourselves and hand it to the agent; that's giving someone the map and watching them walk out of the maze. Every turn you just saw was the agent's own." | 2:20 |
| 9 | `#s-model` | `→` | Bench table, gpt-5.6-sol medium highlighted as chosen; sol low failed endgame_eps_math; strip 139 / ~250 / 42 / 4/10 -> 10/10 | "The model is gpt-5.6-sol at medium effort. High and extra-high made proposals truncate and fail in earlier runs. Low passed routine decisions on our bench but failed the endgame epsilon arithmetic. Medium was the measured sweet spot." | 2:30 |
| 10 | `#s-receipts` | `→` (or `0`) | Nine cells: 0.6056 primary, 0.6728 GAUC, 0.5383 nDCG@5, +0.0040, 6 of 50, 17.0 min, 115,315 tokens, epsilon rule, 0 mid-run interventions; note with 1K 0.66892 and the team line | "That run: six of a possible fifty iterations, seventeen minutes wall-clock, 115 thousand LLM tokens, no GPU, zero mid-run interventions. From 0.6016 to 0.6056, plus 0.0040 on validation primary. We're team jit.ai. Thank you." | 2:42 |

## Before you hit record

- Reload the page once (`Cmd+R`) then `Home`: the run chart and the replay reset only on reload.
- Scene 3 and scene 4 each swallow three extra keys; if you press once too many you land on the next scene, press `←` to come back (it returns to the last beat of the previous scene).
- Scene 5 needs exactly seven `→` after arrival. The eighth moves to the harness.
- Scene 7 has a search box; do not click inside it, the keys go to the box instead of the deck.
- Mouse pointer: keep it parked at the bottom edge except on scene 6 where you point at cards.

## Script notes (not changed by me, SCRIPT_ROHAN is read-only)

- The docx shot list still names `#s-decisions`; that scene does not exist. The real pair is `#s-principles` then `#s-model` (rows 8 and 9 above).
- The stopping rule (three consecutive iterations without +0.002 on the best) was cut from the architecture beat in the docx. The run walkthrough still says "under 0.002, so third strike. The convergence rule fires". If you keep that, one sentence in scene 4 beat 4 sets it up: "A rule ends the run: three iterations in a row that fail to beat the best by 0.002, and it stops." The diagram's fourth beat shows exactly that box.
- Timestamps for rows 2, 8 and 9 are my split of the docx section times (0:00 to 0:20 and 2:17 to 2:42); the others are the docx section starts.
