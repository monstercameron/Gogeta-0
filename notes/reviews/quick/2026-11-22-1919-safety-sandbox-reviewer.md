# Quick review 2026-11-22-1919 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- The repo documents a confirmed safety gap: `post_training/rewards.py` can write to the repo and all of `C:` under WSL; the control failed through `evals/sandbox.py`. [C105, C106; `experiments/claims-status.md`]
- The queue puts the fix ahead of GRPO: SQ61 is the next CPU item, and its stated goal is to move the mount protections into `rewards.py`. [“Next CPU” in `experiments/QUEUE.md`]
- The eval sandbox fails closed if its mounts cannot be set up, but it still exposes world-readable files and is explicitly not a security boundary. [“Still NOT isolated” in `evals/sandbox.py`]

## Ideas

- **Probe namespace escape paths.** [speculation] Hypothesis: crafted symlinks or inherited file descriptors could reach protected paths despite the private mount view. Cheapest test: CPU-only, about 3 minutes; run escape probes against the sandbox self-test. If any probe reaches the repo or Windows files, require a stronger isolation boundary for untrusted code.
- **Check secret exposure in failure output.** Hypothesis: exceptions or reward results could reveal environment values or readable credential files even when the child environment is restricted. Cheapest test: CPU-only, about 3 minutes; pass synthetic canaries through the environment and a world-readable fixture, then inspect captured results and logs. If a canary appears, tighten output handling before using real credentials in the runner.

## Question for the main agent

- Does any current GRPO path run untrusted completions through `post_training/rewards.py` before SQ61 is complete?