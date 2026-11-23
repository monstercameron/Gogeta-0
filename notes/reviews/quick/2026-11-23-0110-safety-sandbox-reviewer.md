# Quick review 2026-11-23-0110 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- The queue identifies a concrete blocker: reward code can write to the repo and `/mnt/c`, and SQ61 must harden it before GRPO runs (`experiments/QUEUE.md`, C105).
- The claims distinguish two limits: the eval sandbox passed 11 probes but still exposes world-readable files (C106); the reward sandbox may silently run without network isolation when `unshare -rn` fails (C83).
- The latest devlog records that labeled F2 stage-1 scores reached the session through the fleet monitor. The estimator was committed before use, but this exposure weakens the blinding boundary (`DEVLOG.md`, v0.4.89).

## Ideas

- **Fail-closed isolation check [speculation]:** Hypothesis: when network isolation setup fails, reward code can still execute. Cheapest test: under 1 CPU-minute at 13M scale, force `unshare -rn` to fail and assert the child never starts. If it does start, keep reward execution gated until the wrapper fails closed.
- **F2 redaction regression check [speculation]:** Hypothesis: score redaction may cover the fleet monitor but miss another reporting path. Cheapest test: under 1 CPU-minute, feed synthetic labeled F2 status records through the reporting paths and assert outputs contain only permitted counts. If any labeled score appears, suppress stage-1 score fields across that path before further blinded analysis.

## Question

- Which paths besides the fleet monitor can expose stage-1 F2 scores, and are they covered by the redaction change?

## Triage
- Fail-closed isolation check: merged into SQ61 as a required test (C83: the reward child must never start if `unshare -rn` fails).
- F2 redaction regression check: accepted as SQ313. Partial fix now in place: F2 stage-1 score files are no longer staged in version commits until the blinded sigma is recorded.
- Question (other exposure paths): (1) the fleet monitor is redacted since ~01:00 EST; (2) **67 F2 stage-1 score JSONs were committed and pushed in v0.4.88** (public repo), so they are readable; review agents are now told not to compute per-cell F2 results; (3) the cockpit TUI Fleet/Runs tabs and run-form drafts can show val loss and are not yet audited (SQ313); (4) `pull.sh` prints only counts. To be disclosed in the next devlog entry.
