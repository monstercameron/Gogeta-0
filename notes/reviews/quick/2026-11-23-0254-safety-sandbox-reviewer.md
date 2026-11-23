# Quick review 2026-11-23-0254 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- **C106:** The eval sandbox passed 11/11 self-tests, but its recorded limitation says world-readable files remain accessible; this is a process sandbox, not a security boundary. (experiments/claims-status.md, C106)
- **Shared filesystem surface:** `evals/sandbox.py` puts `/mnt`, `/tmp`, `/dev/shm`, and `/run` behind private tmpfs mounts. **[speculation]** Other shared paths, such as `/var/tmp`, may still be writable by the sandbox child. (evals/sandbox.py)
- **GRPO remains a separate risk:** the queue marks SQ61 as needed before any GRPO run because `post_training/rewards.py` can write to the repo and `C:` under WSL. (experiments/QUEUE.md, C105)

## Idea

- **Probe writable paths outside the private mounts.** **Hypothesis [speculation]:** a world-writable path outside the four mounted directories could permit cross-run writes. **Cheapest test:** ~1 CPU minute at 13M scale; add a self-test probe for representative shared paths such as `/var/tmp`, using a unique canary and checking cleanup. **Change the plan if:** any write succeeds; mount that path privately or fail closed before code evaluation.

## Question

- Can the eval sandbox’s self-test verify that no shared writable paths remain outside its private mounts?

## Triage
- Probe writable paths outside the private mounts: **confirmed and fixed now.** A canary written to `/var/tmp` from inside `evals.sandbox.run_tests` persisted on the host (`/tmp` was already private). The fix:
  - `/var/tmp` added to the private tmpfs mounts (`HIDE`);
  - a new self-test probe, `shared_tmp_private`;
  - re-probe: the canary is absent from the host;
  - `python -m evals.sandbox`: 12/12 checks ok (`experiments/checks/results/test-runs/2026-11-23-sandbox-selftest.json`).

  C106 gets the evidence.
- Question (can the self-test verify no shared writable paths remain?): partly, as of now. `shared_tmp_private` covers `/var/tmp`, and the existing probes cover the repo, home, `/mnt/c` and `/tmp`. A full sweep (every world-writable dir from `find / -xdev -perm -o+w -type d`) is a follow-up inside SQ61.
