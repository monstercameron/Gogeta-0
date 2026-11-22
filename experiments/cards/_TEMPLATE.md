---
id: E<n>            # E<n> experiment, K<n> check, SQ<n> side quest
type: experiment    # experiment | check | side-quest
title: <short title>
status: planned     # planned | running | done | failed | abandoned | inconclusive
headline: <one line: the key result with its number, or "not run yet">
reuse: <one line: what later work can reuse from this (checkpoint, script, data, finding)>
---

# <ID>: <title>

## Question and prediction
- **Question:** …
- **Prediction (committed before running):** where it lives (registry field or `experiments/checks/PREDICTIONS.md` section) and the commit.
- **Pass criteria:** …

## How to run it
- Config: `configs/…` · Launch: `bash experiments/launch/<ID>.sh [--seed N]` (or the check script path)
- Code version: commit(s) the runs used
- Compute: GPU/CPU, wall time, tokens/FLOPs

## Runs
| Run id | Seed | Commit | Status | Key metrics |
|---|---|---|---|---|

## Results
Numbers, tables and the verdict per criterion. Link the result files (run `summary.json`, `experiments/checks/results/*.json`, `experiments/curves/*.json`).

## Artifacts (where things live)
- Checkpoint(s): `~/fusiondance-data/checkpoints/<run>/model.pt` (WSL, not in git)
- Logs: `~/fusiondance-data/logs/…` · Tracker: Trackio project `gogeta-0`, run `<run>`
- Samples / curves / result JSON: paths

## How to reuse this
Concrete uses for later agents: e.g. "the E2 checkpoint is the reference looped model for loop diagnostics (K6, SQ14)", "use this config as the matched baseline", "this script is a template for X". Also list what **not** to reuse it for (known flaws).

## Caveats
Single seed? Untuned? Known bugs? Superseded by?

## Links
Ledger claims (C…), side quests (SQ…), write-up (`experiments/writeups/<ID>.md`), notebook section, devlog version.
