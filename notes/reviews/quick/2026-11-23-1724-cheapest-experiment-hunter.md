# Quick review 2026-11-23-1724 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- **75 of 142 claims are UNVERIFIED**, and several claim scopes are still TBD; the queue’s validity-first ordering is well motivated. [experiments/claims-status.md; experiments/QUEUE.md]
- **C214 is limited to one 330-file out-of-corpus set**, where unseen files and library style are confounded. [C214; DEVLOG.md, v0.5.39]
- **SQ75 is the next GPU pick at 8 minutes** and profiles E0/E1/E2 training steps to check the roofline verdict. [experiments/QUEUE.md; C119]

## Idea

- **[speculation] Check whether the training bottleneck shifts with sequence length.** Hypothesis: the kernel-time mix at the current profiler length may not predict the bottleneck at longer contexts. **Cheapest test:** GPU, about 5 minutes at 13M scale; extend SQ75’s warmed-up profile to lengths 256, 512, and 1024 for one fixed config. **Change the plan if:** attention or mask work becomes the largest component by 512; prioritize that path before broader kernel optimization.

## Question

- At what sequence length was the planned SQ75 profile intended to run?
## Triage
- Bottleneck vs sequence length: accepted into SQ75's queue note (via `tools/queue.py`). The profile also runs at seq_len 256 and 512 next to 1024. SWA runs as a dense T x T mask (C12), so attention and mask work may become the top bucket at long context, which would rank a windowed SWA kernel ahead of other kernel work.
- Question (at what sequence length was SQ75 planned?): T = 1024 with B = 32, r fixed at 4 for E2, `torch.compile` default mode (`experiments/checks/sq_hw_step_profile.py` lines 8 and 48). That matches the training setup. The added lengths make it a length sweep.
