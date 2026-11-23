# Quick review 2026-11-22-2309 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- The claims ledger has 71 UNVERIFIED claims, and many definitions still have TBD scope fields. [experiments/claims-status.md]
- The queue reports Q7 running and no GPU pick, but its last update is 19:24; v0.4.83 records later wave registrations. Queue freshness may be limiting cost-effective scheduling. [experiments/QUEUE.md; DEVLOG.md]
- E60 stage 1 is estimated at about 2.5 GPU-hours, with six seeds per arm. Its router controls make that spend informative, but a cheap implementation check could catch issues first. [DEVLOG.md]

## Idea

- **Model-level memory scaling smoke check.** Hypothesis: inference with a GDN-only 13M model keeps persistent state size fixed as sequence length grows. Cheapest test: 3 GPU minutes with synthetic inputs at lengths 128–1,024; report peak allocated memory and carried-state size. If state size grows with length or peak memory jumps unexpectedly, inspect the model path before longer-context runs; otherwise retain the current design.

## Question

- Is the immediate bottleneck GPU availability, clean checkpoints for inference probes, or keeping the queue current?
## Triage
*(main agent, 2026-11-23 00:22)*

- **GDN memory-scaling smoke check: rejected (covered).**
  - C190 (GDN fixed-size recurrent state, which supersedes C15) and K1 (peak activation memory grows about linearly in T, exponent 0.976, during *training*) already cover it.
  - SQ244 measures the wall-clock and memory crossover at 1k–8k tokens.
- **Answer to the question:** keeping the queue current. GPUs are plentiful (rented RTX 5090s on demand, $0.50–0.70/hr), and clean checkpoints for the inference probes exist (7 + 7 seeds, Phase 2b-5090). The binding constraint is registering and preparing the next wave before the current one ends (AGENTS §0h pipeline rule). Waves 5–6 are now registered and queued.
