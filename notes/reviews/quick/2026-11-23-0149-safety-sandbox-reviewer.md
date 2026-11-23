# Quick review 2026-11-23-0149 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- `C106` records 11/11 sandbox probes passing, but says world-readable files remain readable. `C105` documents that reward code can write to the repo, a gap also called out in the queue. (`experiments/claims-status.md`; `experiments/QUEUE.md`)
- The queue places sandbox hardening (`SQ61`) before GRPO runs, while its next GPU pick is E9. The security fix is visible as queued work, but the queue excerpt does not state a launch gate. (`experiments/QUEUE.md`)
- The latest devlog says F2 stage 2 is running with scores blinded. The newest quick review also flags that the queue is not a full view of active fleet work. (`DEVLOG.md`, v0.4.93; `notes/reviews/quick/2026-11-23-0144-reproducibility-auditor.md`)

## Ideas

- **Timeout descendant cleanup.** [speculation] Hypothesis: a timed-out evaluator can leave a child process running after its parent exits. **Cheapest test:** 1 CPU-min; have a synthetic snippet spawn a looping child, trigger timeout, then check whether the child remains. **Change the plan if:** any descendant survives; require process-tree cleanup before code-eval or GRPO runs.
- **Cockpit database access.** [speculation] Hypothesis: redaction can protect displayed values while SQLite files or sidecars remain readable to broader local users. **Cheapest test:** 2 CPU-min; create a synthetic database/WAL and inspect their Windows ACLs and WSL permissions. **Change the plan if:** unrelated users can read them; restrict permissions and verify newly created sidecars inherit them.

## Question

- Can code-execution launches be gated on a passing descendant-cleanup check as well as filesystem and network isolation?

## Triage
- Timeout descendant cleanup: accepted and merged into SQ61's acceptance tests. `evals/sandbox.py` already kills the whole process group (`start_new_session=True` + `os.killpg`), but its `timeout_kills` probe only covers a busy loop, and a grandchild that calls `setsid` escapes the group. `post_training/rewards.py` has no process-group kill. SQ61 now requires no survivor after a timeout, via a PID namespace or cgroup kill.
- Cockpit database ACLs: rejected for now. The cockpit runs on a single-user laptop, its SQLite logs hold run metadata and no secrets (AGENTS: secrets stay out of the repo, logs and Codex), and blinded F2 values are already in the public repo (v0.4.88), so file ACLs would not protect the blinding.
- Question (gate code-execution launches on descendant cleanup as well as isolation): yes. SQ61's acceptance now has three fail-closed gates: network isolation, filesystem isolation and process-tree cleanup. GRPO stays blocked until all three pass.
