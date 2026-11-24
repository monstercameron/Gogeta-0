# Quick review 2026-11-23-2052 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- C119’s evidence is a 53% GEMM share and 95.5% GPU busy time on a compiled RTX 5090 run. That supports its registered criterion, but does not by itself measure memory bandwidth or establish the same bottleneck on the N1X. [C119; DEVLOG.md]
- The C3 production-kernel causality check passed on a rented 5090. That verifies the tested CUDA path and hardware; it leaves N1X behavior unmeasured. [C3; DEVLOG.md]
- `BitLinear` fake-quantizes weights and still calls `F.linear`; the repo tracks actual packing and latency separately in SQ378. [gogeta/model.py; SQ378]

## Ideas

- **[speculation] Measure compile amortization.** Hypothesis: cold compilation materially changes the cost of short 13M-parameter jobs. Cheapest test: about 5 GPU minutes, comparing cold and warm 20-step timings on the same config. If cold-start overhead exceeds 10% of a typical queued job, revise runtime estimates or group compatible runs; otherwise keep current estimates.
- **[speculation] Check sustained throughput and clocks.** Hypothesis: thermal or power limits reduce step throughput after warm-up. Cheapest test: about 10 GPU minutes at 13M scale, logging temperature, clocks and step time in short intervals. If throughput drops over 10% and tracks clocks or temperature, use sustained rather than initial throughput in scheduling estimates; otherwise retain current assumptions.

## Question

- Should C119’s prose stay scoped to its measured RTX 5090 criterion until device-level bandwidth evidence supports the broader “memory-bandwidth-bound” description?

## Triage
- Idea 1 (compile amortization, cold vs warm): already covered. SQ77 includes the fla autotune cold/warm comparison, and C123 is the claim about it. The fleet runs are 9-10 minutes each with compile included in wall time (C1: 590-620 s per 200M run), so estimates already absorb it. Not added.
- Idea 2 (sustained throughput vs clocks): partly covered by SQ79 (cool-down) and K7's telemetry. On the 5090 fleet the per-run wall time was stable across 8 sequential C1 runs (9.4-10.0 minutes), with no throttling trend. No new SQ; any N1X run is under the operator hold.
- Question (scope C119's prose?): yes. C119 now carries a limitation: one RTX 5090, compiled, not the N1X, and "bandwidth-bound" is inferred from the non-gemm share and busy fraction, not from bandwidth counters.
