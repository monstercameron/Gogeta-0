# Quick review 2026-11-23-0124 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- `C83` says `rewards.py` can silently run generated code with network access when `unshare` is unavailable; its filesystem is also not private. (`experiments/claims-status.md`, `post_training/rewards.py`)
- `C106` reports 11/11 probes for the separate eval sandbox, while explicitly limiting its claim: world-readable files remain visible and it is not a security boundary. (`experiments/claims-status.md`)
- `SQ61` is the next CPU item and targets moving eval sandbox hardening into the reward harness before GRPO. (`experiments/QUEUE.md`) The latest devlog also discloses that F2 stage-1 per-run scores reached the session monitor before redaction. (`DEVLOG.md`, v0.4.89)

## Idea

- **Fail closed at the reward integration:** Hypothesis: requiring both network and filesystem isolation at the GRPO reward call boundary prevents a fallback from being mistaken for a sandboxed run. Cheapest test: under 1 CPU minute at 13M scale, simulate unavailable `unshare` and confirm the reward path refuses to execute generated code; then confirm it runs when isolation is available. If code executes in the simulated fallback, make this a hard GRPO gate; if it refuses, retain the guard as a regression check.

## Question

- Can SQ61’s acceptance criterion include a test that the GRPO reward integration refuses to score code when either isolation guarantee is absent?
## Triage
- Fail-closed at the reward integration: duplicate of the fail-closed test merged into SQ61 at 01:10 (C83). Extended there to require **both** network and filesystem isolation at the GRPO reward call boundary.
- Question: yes. SQ61's acceptance now includes "refuses to score when either isolation guarantee is absent".
