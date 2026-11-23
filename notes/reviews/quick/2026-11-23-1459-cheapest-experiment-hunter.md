# Quick review 2026-11-23-1459 - cheapest-experiment hunter (Codex GPT-6 Luna high)

- **Observation:** 71 of 140 claims are UNVERIFIED, so cheap checks that unblock later work have high value. (`experiments/claims-status.md`)
- **Observation:** The C80 recheck found the Hugging Face wrapper broken from v0.5.0 through v0.5.23; v0.5.24 records the fix and saved passing output. (`DEVLOG.md`, v0.5.24)
- **Observation:** E8 is using the GPU; Q15 is next on CPU, and Q16’s 15-minute GPU check depends on it. Q15 currently has no launcher command. (`experiments/QUEUE.md`)

- **Idea — Add a synthetic scorer check to Q15. Hypothesis:** A tiny known-answer fixture can catch MQAR recall metric errors before Q16 spends GPU time. **Cheapest test:** About 2 CPU minutes; feed the scorer synthetic 13M-scale-run outputs with known 100% recall, then a deliberately corrupted key/value mapping. **Change the plan if:** either score misses its expected result—fix the scorer before running Q16; otherwise proceed with the redesigned check.

- **Question for the main agent:** Can Q15 include this scorer fixture so Q16 starts only after its metric passes?
## Triage
- Synthetic scorer check for Q15: accepted and written into Q15's queue note (via `tools/queue.py`). Known-answer outputs must score 100% recall, and a corrupted key/value mapping must score at chance. Q16's GPU check starts only after both pass. K3 failed its validity gate once (attention control 8.6%), so checking the metric before spending GPU time is the right order.
- Question (can Q15 include the fixture so Q16 waits on it?): yes, done. It is part of Q15's definition now, and Q16 already depends on Q15.
