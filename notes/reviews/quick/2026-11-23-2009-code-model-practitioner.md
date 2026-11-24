# Quick review 2026-11-23-2009 - code-model practitioner (Codex GPT-6 Luna high)

## Observations

- The assembled model’s reported 16.7% clean80 gain over E27 comes with about 1.6× the unique parameters; SQ419 is running to test a parameter-matched control. [C209; DEVLOG.md, v0.5.54]
- The project already recognizes that code likelihood can diverge from practical usefulness: the planned functional eval scores generated code against held-out asserts. [experiments/side-quests.md, SQ4]
- Q15’s redesign must confirm the attention control on held-out generator seeds before Q16 compares mixers; that helps keep control tuning separate from the mixer result. [experiments/QUEUE.md; C16]

## Idea

- **[speculation] Check whether model confidence can flag risky completions.** **Hypothesis:** average completion-token likelihood ranks executable completions above failures well enough to support selective review. **Cheapest test:** on 50 held-out tasks at 13M scale, generate a few samples, record likelihood and execution outcome, and compute AUROC; about 5–10 GPU minutes. **Change the plan if:** AUROC exceeds 0.65—add confidence-based deferral to the code eval; otherwise, don’t use likelihood as a correctness signal.

## Question

- Can Q14’s saved completion-level scores include token likelihood so confidence can be compared with execution outcomes?

## Triage
- Idea (does likelihood flag risky completions?): accepted as an SQ4 extension. Each completion's mean token log-likelihood is stored next to its execution outcome, with the AUROC reported and the review's 0.65 rule. It runs at E61 scale with SQ4, because a 13M model's pass rate is near 0, which leaves too few passes for an AUROC.
- Question (can Q14's completion scores include token likelihood?): yes. The SQ4 extension makes it a stored field of the code-eval output, so Q14 inherits it when it runs.
