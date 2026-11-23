# Quick review 2026-11-23-1644 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- `C106` is verified on this machine, and its own limitation says world-readable files remain readable. The sandbox documentation also calls it a process sandbox, not a security boundary. ([experiments/claims-status.md](experiments/claims-status.md), [post_training/rewards.py](post_training/rewards.py))
- The evaluator gates execution on a passing sandbox self-test, including checks for network isolation, hidden paths, memory limits, and timeout handling. That evidence is machine-specific. ([evals/sandbox.py](evals/sandbox.py), `C106`)
- B1 is listed as running on a rented RTX 5090, but the queue excerpt gives no current spend or idle-time reading. The latest devlog progress is a snapshot, so it does not establish live fleet status. ([experiments/QUEUE.md](experiments/QUEUE.md), [DEVLOG.md](DEVLOG.md), v0.5.34)

## Ideas

- **[speculation] Test whether a solution can fool an assertion through Python object behavior.** Hypothesis: a custom object with an always-true `__eq__` may pass an assertion without producing the expected value; the harness documentation identifies this as a residual risk. **Cheapest test:** CPU, under 1 minute, harness-only at 13M model scale. **Change the plan if:** a false result scores as passing; isolate test evaluation or constrain accepted result types before trusting execution scores. ([post_training/rewards.py](post_training/rewards.py))
- **[speculation] Add a live cloud-spend and idle-age reading to the cockpit status used during active batches.** Hypothesis: a queue marked “running” can hide a stalled or over-budget instance when progress snapshots age. **Cheapest test:** CPU, under 1 minute, inspect one status refresh during B1; no model run needed. **Change the plan if:** spend or idle age is unavailable or stale; make those fields part of the required status view before relying on kill-on-empty. ([experiments/QUEUE.md](experiments/QUEUE.md), [DEVLOG.md](DEVLOG.md))

## Question

Can the cockpit report B1’s current provider spend and last-job activity time alongside its running state?
## Triage
- `__eq__` forgery: accepted as SQ411, marked as blocking any GRPO run (as SQ373 was). It is a real open route, documented as residual in `post_training/rewards.py` (lines 81-83): a returned object with an always-true `__eq__` passes `assert candidate(x) == expected`. The fix rewrites the tests' comparisons into a harness-bound strict equality over builtin value types, with a probe arm in `reward_forgery_probe.py` and unit tests. No GRPO run is planned before it lands.
- Live spend and idle-age reading: accepted as SQ412, a cockpit function exposed via MCP (AGENTS 0g: extend the cockpit rather than hand-check).
- Question (can the cockpit report B1's live spend and last-job activity?): not yet; SQ412 adds it. A manual reading now (16:46, `vastai show user` / `show instances`): one instance, gogeta-cockpit-b1-1, running at $0.658/h with 0.83 h uptime (about $0.55 so far); provider total spend $43.88, credit $56.12. Last-job activity: 20 of 23 jobs had RUN lines by 16:40, with the latest finished at 16:39 EST. The single controller polls every 60 s and destroys the box on an empty queue.
