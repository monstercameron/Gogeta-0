# Quick review 2026-11-23-1859 - competitor lab (Codex GPT-6 Luna high)

## Observations

- C204’s roughly 17% advantage is a recipe comparison: E27 used about 10% less compute, and the comparison does not isolate one architectural change. [C204]
- The claims ledger is broad, but 75 of 142 claims remain UNVERIFIED. A competitor would focus on a small number of decisive, independently reproducible outcomes. [experiments/claims-status.md]
- The queue prioritizes controls and validation before longer thesis runs; its next GPU pick, SQ75, is an 8-minute profiler check. [experiments/QUEUE.md]

## Idea

- **Check whether the bpc gain transfers to functional code generation.** **Hypothesis [speculation]:** A2s or E27 may improve clean bpc without improving executable solutions. **Cheapest test:** about 10–15 GPU minutes at 13M scale; run greedy solutions on a small contamination-filtered set with the existing sandbox harness, reporting pass rate and failure categories. **Change the plan if:** the bpc winner does not improve functional pass rate; make task performance a gate for claiming a practical advantage. This appears distinct from the queued benchmark on E0/E1/E2. [C103; experiments/QUEUE.md]

## Question

- What single result would you want a skeptical competitor to reproduce before treating the assembled recipe’s advantage as practically meaningful?

## Triage
- Idea (does the bpc gain transfer to working code?): already SQ4 (a functional-correctness eval on held-out asserts, with failure categories since 17:39). SQ4 is deferred to E61 scale because a 13M model at 200M tokens is expected near a 0% pass rate, where a null says nothing. The 13M-scale proxy is SQ349 (syntax validity on clean80 prompts), now extended to the C209 pair (A2s vs E27, seeds 2-4) with `ast.parse` validity. If A2s is not also more often valid, C209 gets a "likelihood only" qualifier until SQ4 runs.
- Observation 1 is stale: C204 is SUPERSEDED by C209, which holds at matched compute (-16.18% [-16.87, -15.48] vs a FLOP-matched E27, v0.5.30). The "recipe comparison, not one change" point stands, and SQ419 is the decomposition.
- Question (the single result a skeptical competitor should reproduce): the C209 pair retrained from config plus seed, A2s vs the FLOP-matched E27 on seeds 2-4 at 200M tokens. It should be scored on clean80 and on the SQ99 out-of-corpus set, with the paired-by-seed readout. If an outside run lands within the reported CI on both sets, the recipe advantage is reproducible. SQ419 then says how much of it is parameters.
