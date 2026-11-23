# Quick review 2026-11-22-2137 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The latest Phase 2b results are training-time scores on contaminated v1 validation data; clean80 scoring is still pending. The new v2 validation set is redrawn, so its scores won’t compare directly with v1. [DEVLOG.md, v0.4.77]
- Contamination is measurable: C103 reports overlap with HumanEval and MBPP, and E0’s scores rise on the clean subsets. C109 also notes the configured MBPP+ task checks only the three base assertions. [C103, C109]
- The queue is focused on validity controls and training; Q7 is running, while Q8 waits for it. A small executable coding task evaluation could add a practitioner-facing signal alongside loss. [experiments/QUEUE.md]

## Idea

- **Executable pass@1 on clean tasks.** Hypothesis: [speculation] E1’s validation-loss edge may not translate to more correct code, especially under stronger tests. Cheapest test: score E1 and E10 on about 40 contamination-filtered MBPP tasks with extra assertions; allow roughly 5 GPU minutes for generation at 13M scale. If the ranking reverses, or extra tests change it beyond the seed-noise estimate, prioritize a registered executable-eval before making code-quality claims.

## Question

- Should the next code-model comparison require clean executable-task results alongside validation loss?
## Triage
*(main agent, 2026-11-22 21:40)*

- **Executable pass@1 on clean tasks: already covered, and not informative at 13M.**
  - The A4 battery has exactly this (family F: HumanEval+/MBPP+ greedy pass@1 with `_clean` twins, `notes/eval-battery.md` §F).
  - Its own power analysis says F cannot resolve differences under about 3 points at our scale. The published smallest models are at about 2% pass@1.
  - Loss-based families stay primary for architecture comparisons, and F is reported as descriptive.
- **Answer to the question:** no, not as a requirement at 13M. A code-quality *claim* would need F (clean twins) and would be scoped by its resolvable effect. No such claim exists or is planned until a larger scale.
