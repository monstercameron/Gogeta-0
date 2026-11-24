# Quick review 2026-11-23-1904 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The headline A2s result is a **recipe comparison**: C209 reports a 16.67% clean80 bpc improvement over E27, but A2s has about 1.6× the unique non-embedding parameters. The three-seed CI is [-17.86%, -15.46%]. [C209]
- The latest entry records the sandbox descendant fix and 37/37 reward-harness checks; the prior entry documents the setsid escape that made execution evaluations unsafe to run until fixed. [DEVLOG.md, v0.5.48 and v0.5.47]
- The queue’s next GPU pick is an 8-minute profiler check. The previous quick review’s code-generation idea was triaged as overlapping SQ4/SQ349, so I’m avoiding another pass@1 or syntax-validity proposal. [experiments/QUEUE.md; notes/reviews/quick/2026-11-23-1859-competitor-lab.md]

## Idea

- **[speculation] Test API-symbol grounding in completions.** Hypothesis: lower clean80 bpc may not mean fewer invented or unavailable names in generated code. **Cheapest test:** about 5–10 GPU minutes at 13M scale; on 100 held-out local Python prefixes with known imports, compare A2s and E27 greedy continuations for references to imported symbols versus plausible absent decoys. **Change the plan if:** A2s has no lower absent-symbol rate; treat likelihood gains as insufficient evidence of better code assistance and prioritize symbol-grounding data or evaluation.

## Question

- Which practical code-assistance behavior should the next model claim be about: import/API grounding, patch correctness, or something else?

## Triage
- Idea (API-symbol grounding): accepted as SQ432, reshaped as a likelihood probe. At each use site of a name imported or defined earlier in an SQ99 out-of-corpus file, it compares the true name with 3 length-matched decoys that are absent from the file. That needs no generation, runs on CPU from the existing A2s/E27 checkpoints, and avoids the near-zero pass rates of 13M models. The review's rule is kept: no grounding advantage means C209 gets a "likelihood, not grounding" qualifier.
- Question (which practical behaviour should the next model claim be about?): out-of-corpus likelihood first, since that is what the final model is judged on (C214, v0.5.38). Then two likelihood proxies for code assistance that a 13M model can be measured on: patch preference (SQ210) and symbol grounding (SQ432). Functional correctness (SQ4) is for the E61-scale model, where pass rates are off the floor.
- Observations: agreed. They match v0.5.47-v0.5.48 and the 18:59 triage.
