# Quick review 2026-11-23-1834 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- C106’s 16/16 self-test and 6/6 gate tests support specific process-sandbox behaviors; its scope explicitly stops short of a general security boundary. [claims-status.md, C106; DEVLOG.md, v0.5.43]
- The latest devlog records that the GRPO reward path fails closed when isolation checks fail. The remaining sandbox hardening work includes resource-budget and descendant-cleanup checks. [DEVLOG.md, v0.5.43; side-quests.md, SQ61]
- The queue’s next GPU and CPU picks concern profiling and MQAR; neither addresses the sandbox limitations. [QUEUE.md, Next picks]

## Ideas

- **[speculation] Forced-kill mount cleanup.** **Hypothesis:** killing an evaluation worker during setup or execution could leave mounts behind in its namespace or affect a later evaluation. **Cheapest test:** CPU, about 2 minutes; run a synthetic worker, force-kill it, then check mount state before and after, without loading a model. **Change the plan if:** any mount persists or crosses into the host namespace; block repeated sandbox runs until cleanup is fixed.
- **[speculation] Concurrent fail-closed gate.** **Hypothesis:** simultaneous evaluation requests could start children before a failing isolation self-test has settled. **Cheapest test:** CPU, under 5 minutes; use a stubbed failing self-test and concurrent 13M-scale evaluation workers, then assert zero children start. **Change the plan if:** any child starts; serialize or synchronize the gate before relying on parallel evaluation.

## Question

- Should GRPO remain gated until the open resource-budget and descendant-cleanup checks in SQ61 pass?

## Triage
- **Idea 1 (forced-kill mount cleanup): tested now. Mounts are clean, and it surfaced a real descendant escape.** `experiments/checks/sandbox_kill_cleanup.py` (saved output `results/test-runs/2026-11-23-sandbox-kill-cleanup-1840.txt`).
  - The host mount table is identical before and after (32 entries). The tmpfs mounts live in the child's private mount namespace and die with it.
  - A busy loop and a same-group child are killed within the 1 s timeout.
  - **A child that calls setsid survives the process-group SIGKILL** (as nobody). The harness then blocks in `proc.communicate()`, which has no timeout after killpg (`post_training/rewards.py:495`), until that child exits. One generated sample can therefore stall a reward or eval worker indefinitely.
  - Recorded as limitations on C83 and C106. SQ61 now gates GRPO; the fix is a PID namespace with --kill-child plus a bounded communicate.
- **Idea 2 (concurrent fail-closed gate): rejected by code read.** `_GATE` is assigned once, only after `self_test` has built the full report (`evals/sandbox.py`, end of `self_test`), and every `run_tests()` checks it before any child starts. Two racing threads at most run the self-test twice, and neither can start a child on a partial or failing report. Each process has its own `_GATE`.
- **Question (should GRPO stay gated until SQ61's resource-budget and descendant-cleanup checks pass?): yes, now for a demonstrated reason.** The setsid escape above is a stall-and-survive bug in the reward path. GRPO and code-execution evals wait for SQ61's descendant cleanup to pass this probe. The `__eq__` lift (SQ411) stays, but it no longer means the harness is GRPO-ready.
