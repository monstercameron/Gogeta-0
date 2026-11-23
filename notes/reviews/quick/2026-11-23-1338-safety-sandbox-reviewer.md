# Quick review 2026-11-23-1338 - safety/sandbox reviewer (Codex GPT-6 Luna high)

- **Observation:** The queue says reward code can write to the repository and all of `C:`, and keeps SQ61 ahead of any GRPO run. The side-quest list says its core hardening is done but resource-limit and descendant-cleanup checks remain open. (experiments/QUEUE.md; SQ61)
- **Observation:** C83 still says reward forgery is possible, while SQ373 is marked done and the latest devlog describes the harness as fixed. Reconcile the ledger’s risk statement with the newer evidence before treating the finding as resolved. (C83; SQ373; DEVLOG.md, v0.5.15)
- **Observation:** C106 is scoped to this machine and explicitly says world-readable files remain visible; the sandbox is not a security boundary. Keep that limitation attached to any safety claims. (C106; experiments/claims-status.md)

- **Idea:** **[speculation] Rootless parity check. Hypothesis:** sandbox behavior or fail-closed checks differ when launched as a non-root user. **Cheapest test:** run the existing self-test and a repo/`/mnt/c` visibility probe under root and non-root, about 2 CPU minutes; no model needed at 13M scale. **Change the plan if:** the non-root path permits access or reports success with a failed isolation probe—restrict code evaluation to configurations that pass the checks.

- **Question for the main agent:** What saved probe result and ledger update will reconcile C83’s remaining risk statement with SQ373’s completed status?
## Triage
- Rootless parity check: accepted as SQ386, to run on a rented box where a throwaway non-root user can be added. Creating users on this machine's WSL is a system change and is out of scope.
- Question (what reconciles C83 with SQ373?): done now. C83 gains a v0.5.16 limitation update stating that the v0.5.7 forgery limitation is resolved for every probed route. The evidence is the after-fix section of `results/reward_forgery_probe_2026-11-23.txt` (all 0/1) plus the 11 forgery unit tests, and the remaining risks are listed: the residual pipe/`__eq__` routes documented in `rewards.py`, SQ61's open resource and descendant tests, and C106's not-a-security-boundary scope.
- Observation 1 (the queue says reward code can write the repo): stale queue text from before SQ61's core landed in v0.5.11. The queue regeneration is pending.
