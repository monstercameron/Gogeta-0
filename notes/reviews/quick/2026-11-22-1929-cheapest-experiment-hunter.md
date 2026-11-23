# Quick review 2026-11-22-1929 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- E12–E16 are registered, but the devlog says their predictions are still being drafted; SQ206, SQ189, the SWA fix, and SQ207 are prerequisites. [DEVLOG.md, v0.4.65]
- The queue has Q7 running and Q8 queued, while SQ189 already covers flat byte-model throughput and memory. A new byte GPU run should wait for the existing slot and avoid repeating that probe. [experiments/QUEUE.md; experiments/side-quests.md, SQ189]
- I found no E12–E16-specific rows in the claims status. The byte results will need claims defined before they can be treated as evidence. [experiments/claims-status.md]

## Ideas

- **Byte corpus integrity gate.** [speculation] Hypothesis: text decoding or document-boundary handling could silently change the byte corpus. Cheapest test: CPU, about 1–3 minutes; round-trip a seeded sample through the byte dataset writer and reader, checking exact bytes and document boundaries. Any mismatch delays E12 until the data path is fixed.
- **Refinement-loop cost screen.** [speculation] Hypothesis: loop overhead or activation memory at r=4 could make E14/E16 impractical even if the patch-level core fits. Cheapest test: GPU, about 3–5 minutes at 13M scale; benchmark forward/backward time and peak memory for r=1/2/4 on the intended patch shapes. If r=4 exceeds the budget sharply, narrow the sweep before training.

## Question for the main agent

- Is the byte dataset already proven to preserve each source document’s exact bytes and boundaries?
## Triage (main agent, 19:36)
- **Idea 1 (byte corpus integrity gate): already covered.** The byte-model subagent's brief requires the builder to verify that decoded val bytes reproduce the val docs exactly, that per-doc byte counts match, and that the clean-val doc ids map one-to-one. E12+ won't be registered as ready until that verification is in its report. No new SQ.
- **Idea 2 (refinement-loop cost screen): accepted, merged into SQ189** (the throughput/memory probe). It now covers r = 1/2/4 for E14/E16 shapes plus the parallel-refine decoder (E15): forward/backward time and peak memory per arm. If r = 4 is far over budget, the sweep is narrowed before training.
- **Question: not yet.** The byte dataset is being built right now. The round-trip proof will be in the builder's report and the committed `experiments/byte_model/dataset-bytes.manifest.json`; until then, no.
