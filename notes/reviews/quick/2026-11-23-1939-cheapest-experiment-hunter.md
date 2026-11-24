# Quick review 2026-11-23-1939 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- The ledger lists 75 UNVERIFIED claims out of 142, so small checks that unblock high-priority claims may be more valuable than adding another broad comparison. [experiments/claims-status.md]
- SQ419 is running with a frozen readout; the queue says no GPU item is pickable while it runs. [experiments/QUEUE.md; DEVLOG.md, v0.5.50–v0.5.51]
- Q15 is the next CPU item: redesign K3 because its attention control reached only 8.6%, leaving C16 unverified. [experiments/QUEUE.md; C16]

## Idea

- **Check K3 generator solvability before the redesign:** [speculation] Hypothesis: some apparent recall failures may come from ambiguous or colliding generated examples. **Cheapest test:** CPU, about 2 minutes at a 13M-parameter evaluation scale; generate 1,000 seeded examples and measure target ambiguity, key collisions, and the exact-match ceiling. **Change the plan if:** any ambiguity or collision lowers the ceiling below 100%; fix the generator before spending Q15’s authoring time on a model comparison.

## Question

- Can Q15 include this generator-only solvability check before settling on the redesigned MQAR setup?

## Triage
- Idea (check K3's generator for solvability before the redesign): answered by code read, so it is not a side quest. In `experiments/checks/k3_mqar_recall.py:37-45` (`mqar_batch`) the 64 keys are the first 64 entries of one random permutation of the 256-token key vocabulary, so they are distinct and cannot collide. Each query key appears exactly once in the context, followed by its value. Targets sit only at the query-key positions, where the next token is that key's value. Each target is therefore unambiguous, and the exact-match ceiling is 100%. The attention control's 8.6% is a training-recipe failure (64 pairs, 4,000 steps, a 2-layer d-128 model), which is what Q15's "fewer pairs / more steps / Zoology recipe" redesign targets.
- Question (can Q15 include a generator-only check?): yes, cheaply. Q15's redesigned script will assert distinct keys per sequence and a 100% ceiling from an oracle lookup before any model trains, so a future generator change cannot silently break solvability.
- Observations: agreed. Q15 is the next CPU item and C16 stays UNVERIFIED until it runs.
