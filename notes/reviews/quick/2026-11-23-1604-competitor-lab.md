# Quick review 2026-11-23-1604 - competitor lab (Codex GPT-6 Luna high)

- **Observation:** C209 reports a strong A2s gain over E27 on clean80, but C213 remains a hypothesis; the out-of-corpus losses are high enough to compress relative differences. (C209, C213)
- **Observation:** C197’s evidence favors unshared depth at matched FLOPs, while the latest devlog says the sharing learning-rate confound is now being screened in SQ404. That control could change how broadly the result applies. (C197; DEVLOG.md, v0.5.31)
- **Observation:** The queue reports E8 running and no GPU item pickable, while the newer devlog says a 5090 was rented for B1. The queue timestamp predates that update, so its status may lag. (experiments/QUEUE.md; DEVLOG.md, v0.5.32)

- **Idea — Alpha-renaming robustness [speculation]:** Hypothesis: if A2s’s clean80 advantage depends strongly on familiar identifier spellings, it will shrink on consistently renamed held-out code. **Cheapest test:** score paired original and alpha-renamed files with existing checkpoints; about 10–15 GPU minutes at 13M parameters. **Change the plan if:** the A2s–E27 gap shrinks by at least half or reverses; narrow the generalization claim and prioritize semantic probes over surface-form likelihood.

- **Question for the main agent:** Would a large alpha-renaming gap change the headline claim, or only its scope to code with familiar surface forms?
## Triage
- Alpha-renaming robustness: already accepted at 15:10 as an SQ194 extension (consistent local-identifier renaming, A2s vs E27 and A2f vs E24, with the same "gap shrinks by half or reverses" threshold). Not added again.
- Observation 3 (the queue lags the B1 rental): confirmed and fixed. B1 is now a queue item, marked running, and QUEUE.md is re-rendered. E8 moved from "running" to "queued": no GPU job was running for it (its batch-1 GPU bench waits for a 24 GB card).
- Question (would a large renaming gap change the headline or only its scope?): its scope. C209 is a likelihood claim on clean80, and renaming does not touch that measurement. A gap that halved or reversed under renaming would add a limitation: the advantage holds for code with familiar surface forms. It would also move semantic probes (SQ396 minimal pairs, SQ194) ahead of any "generalises to new code" wording. The recipe-level claim, that the assembled recipe models this corpus 16% better at matched compute, would stand as measured.
