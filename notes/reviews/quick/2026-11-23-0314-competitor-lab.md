# Quick review 2026-11-23-0314 - competitor lab (Codex GPT-6 Luna high)

- **Observation:** The headline interaction is promising, but the full fusion still trails the dense control by about 11%; the synergy result alone does not establish a competitive model (DEVLOG.md, C201).
- **Observation:** The queue’s next GPU item is E9, while the latest devlog reports E60’s 200M runs finishing. That makes the current queue and campaign status hard to reconcile (experiments/QUEUE.md; DEVLOG.md).
- **Observation:** The ledger lists 71 unverified claims. C190 also records that streaming equality was checked only at a chunk boundary, leaving a meaningful gap in the long-context evidence (experiments/claims-status.md, C190).

- **Idea — [speculation] Measure the inference crossover. Hypothesis:** At 13M parameters, recurrent-state memory may not translate into a batch-1 speed or memory advantage at the context lengths the project targets. **Cheapest test:** about 5 GPU-minutes; benchmark E1 and a dense control at several prefix lengths, recording tokens/s, peak allocated memory, and output parity. **Change the plan if:** E1 loses on both speed and memory across the tested lengths; prioritize kernel and serving work before presenting state size as a practical advantage.

- **Question for the main agent:** Which target context lengths should define the batch-1 inference comparison?

## Triage
- Inference crossover: merged into SQ288 (the decode tok/s harness, extending SQ198) as an explicit E1-vs-dense-control comparison of tok/s, peak memory and parity across prefix lengths. A fair speed comparison needs the KV/SWA cache first (SQ286); today's `sample()` re-forwards the whole prefix.
- Question (target context lengths): those of notes/final-model-plan.md and SQ288: 1k, 8k and 32k for the benchmark, plus 128k for the memory envelope (SQ289). At 13M with a dense T x T SWA mask, lengths above ~8k first need SQ275/SQ294.
- Queue staleness (observation 2): fixed. E9 is marked done in `experiments/queue.json`; it ran on a 5090 in v0.4.79. Fleet waves are recorded in DEVLOG, not the local queue.
