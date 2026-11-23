# Quick review 2026-11-23-0239 - safety/sandbox reviewer (Codex GPT-6 Luna high)

## Observations

- C106 records 11/11 sandbox probes passing, but also states that world-readable files remain accessible. Treat the result as a narrow check, not a general security boundary. (experiments/claims-status.md, C106)
- The queue puts SQ61 ahead of GRPO work because the current reward harness can write to the repo and all of `C:`. That is a concrete execution boundary failure still awaiting a fix. (experiments/QUEUE.md, SQ61)
- The latest devlog reports a prior display-blinding breach and committed per-run scores. The follow-up analysis is documented, but the incident shows that restricted results can escape through ordinary project outputs. (DEVLOG.md, v0.4.97)

## Ideas

- **Resource exhaustion probe.** **Hypothesis:** filesystem and network isolation still leave the host exposed to runaway CPU, memory, or process use. **Cheapest test:** about 3 CPU minutes with a 13M-scale local harness; run bounded allocation, fork, and CPU-loop probes inside the sandbox and record enforced limits and cleanup. **Change the plan if:** any probe can exceed its intended resource budget; add enforceable limits before running untrusted code.
- **Safety-claim wording audit.** **Hypothesis:** docs or generated site text may describe sandbox protections more broadly than C106 supports. **Cheapest test:** about 2 CPU minutes; scan the README, site data, and benchmark docs for sandbox/security claims and compare each with its ledger scope and limitations. **Change the plan if:** any unqualified security-boundary claim appears; correct it before the next version.

## Question

- Which specific untrusted-code workloads are expected to run before SQ61 is completed?

## Triage
- Resource exhaustion probe: merged into SQ61's acceptance tests as a resource-budget test (memory, CPU-time and process-count limits enforced and recorded per probe), alongside the network, filesystem and descendant-cleanup gates.
- Safety-claim wording audit: done now (2 CPU-min). README.md and docs/*.html make no sandbox or security-boundary claims. `notes/benchmark-suite.md` uses "sandboxed execution" descriptively (lines 56, 60, 130, 220) for eval tasks that floor at 0 pass. Nothing to correct; C106's own scope ("not a security boundary") stands.
- Question (which untrusted-code workloads run before SQ61): none. GRPO (reward code execution) is blocked on SQ61. The execution evals (Q14 tier 1) are human-gated on dataset downloads and installs. The only execution so far was the `--limit 1` smoke test of the eval sandbox (notes/benchmark-suite.md:220), which ran the benchmarks' own reference tests on model output.
