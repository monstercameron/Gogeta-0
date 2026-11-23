# Quick review 2026-11-23-1739 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The ledger has 75 unverified claims out of 142, and C103 records HumanEval/MBPP overlap in training data. Code-quality conclusions need clean, task-level evidence. [experiments/claims-status.md; C103]
- The latest devlog documents careful fixes to evaluation and reward-harness issues, but those checks do not establish that generated code solves held-out tasks. [DEVLOG.md, v0.5.40]
- The immediate GPU pick profiles training kernels; functional code evaluation is already proposed as SQ4. Practitioner-facing evidence is still behind systems diagnostics. [experiments/QUEUE.md; SQ4]

## Ideas

- **[speculation] Measure distractor sensitivity in repository context.** Hypothesis: irrelevant files or snippets reduce functional pass rate at a fixed context budget. **Cheapest test:** 5 GPU minutes, 100 uncontaminated held-out tasks, compare each prompt with no distractors versus one or four unrelated snippets. **Change the plan if:** distractors reduce pass rate by more than 5 percentage points; prioritize context selection before scaling.
- **[speculation] Track failure types alongside functional pass rate.** Hypothesis: syntax, import, and assertion failures point to different bottlenecks than aggregate clean80 loss. **Cheapest test:** 5 CPU minutes to classify existing SQ4 execution results by failure type. **Change the plan if:** one category dominates; target that failure mode in the next data or training change.

## Question

- Should the practitioner bar be functional pass@1 with repository context before scaling the model?
## Triage
- Distractor sensitivity with pass@1: already merged at 16:39 into SQ385 as a likelihood version (target-token NLL with and without inserted unrelated code), which works at 13M. The pass@1 version waits for Q14 at final scale, where pass@1 leaves the ~2% floor. Not added again.
- Failure-type breakdown: accepted as an SQ4 extension (syntax, import/name, runtime, assertion, timeout next to pass@1). It costs nothing once execution results exist.
- Question (should functional pass@1 with repository context be the bar before scaling?): no, not before scaling. At 13-35M params pass@1 sits at the ~2% floor (`notes/benchmark-suite.md`), so it cannot rank architectures; scaling is what lifts it off the floor. The bar for choosing the architecture stays clean80 plus out-of-corpus bits per char (v0.5.38). Functional pass@1 (Q14, with the forgery-proof harness from SQ373/SQ411) becomes the bar for the final model (E61), together with repository-context completion.
