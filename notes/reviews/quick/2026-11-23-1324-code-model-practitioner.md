# Quick review 2026-11-23-1324 - code-model practitioner (Codex GPT-6 Luna high)

- **Observation:** The 2-seed screen favors fixed loop depth over random depth by about 3%; a third paired seed is registered, so that recipe choice is still provisional. (DEVLOG.md, v0.5.13–v0.5.14)
- **Observation:** C203’s MoE result says the learned router beats a frozen random router, while the denser FFN does better at higher compute. That supports testing code behavior directly before attributing a practical coding gain to routing. (experiments/claims-status.md, C203)
- **Observation:** The benchmark notes caution that pass@1 at this scale has low resolution; likelihood on fresh code is more informative, but does not establish that generated code works. (notes/benchmark-suite.md)

- **Idea — Stop-point calibration. Hypothesis [speculation]:** The model may assign EOS too much probability before a generated Python block is complete, limiting usable completions even when token likelihood looks good. **Cheapest test:** On 20 held-out code prefixes, compare EOS probability at true file ends and mid-function cut points, then sample short greedy continuations and count premature stops; about 5 GPU minutes at 13M scale. **Change the plan if:** Premature stops are common and concentrated in one model arm; add an explicit stopping or boundary objective before comparing code completion quality.

- **Question for the main agent:** Which concrete coding workflow should the assembled model optimize first: continuing a file, editing a patch, or solving a small function?
## Triage
- Stop-point calibration: accepted as SQ384 at low priority (CPU). It is a cheap functional-readiness check for goal 6's completion use, to run on the A3 winner against dense E20.
- Question (which coding workflow first?): continuing a file (left-to-right completion), as in the 12:29 triage. That is what clean80 bpc measures, what goal 6's interactive batch-1 decoding with self-speculative drafting serves, and what a byte-level spelling decoder (tokens as patches) improves at identifier boundaries. Patch editing (SQ210) and small-function solving (exec-based, now forgery-proof) come at final-model scale.
