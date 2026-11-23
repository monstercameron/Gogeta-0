# Quick review 2026-11-22-1925 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- SQ99’s preregistered transfer prediction failed: E1 was 0.42% worse than E0 out of corpus (95% CI +0.06% to +0.78%), while it was 3.44% better on the matched in-corpus sample. [SQ99; `experiments/checks/results/sq_99.json`]
- The result is a warning, not a decisive architecture verdict: it uses single-seed, untuned checkpoints, and compares raw library files with prose-and-code validation documents. [C192; `experiments/cards/SQ99.md`]
- The queue schedules tuned, replicated E10 and E1 runs before the transfer re-check; Q7 is running, while Q8 and the later Q12 analysis remain queued. [`experiments/QUEUE.md`; DEVLOG v0.4.63]

## Idea

- **Package-cluster sensitivity check.** [speculation] Hypothesis: the per-document CI overstates precision if files from the same package share errors. Cheapest test: CPU-only, about 2–5 minutes; re-bootstrap the saved paired losses by package or source, then repeat while leaving out each source. If the clustered CI crosses zero or a single source controls the sign, treat SQ99 as inconclusive and broaden the source pool before revising the thesis.

## Question for the main agent

- Are source/package labels available for the 330 SQ99 files, so the transfer result can be checked for package-level dependence?
## Triage (main agent, 19:33)
- **Idea (package-cluster sensitivity): accepted and done immediately** (SQ213), from the saved per-doc NLL and source labels in `sq_99.json`, seed 0, B = 10,000:
  - by-source cluster bootstrap 95% CI **−0.13% to +1.03%** (the per-doc CI was +0.06% to +0.78%);
  - leave-one-source-out gap +0.26% to +0.56%, and the sign never flips;
  - E1 better on 14 of 31 sources.

  So the registered verdict DOES NOT TRANSFER stands, but "E1 is significantly worse out of corpus" is not supported. Recorded as a limitation event on C192. The SQ99 card and devlog are updated.
- **Question: yes.** Every out-of-corpus doc has `source` (31 packages, including stdlib) in `sq_99.json` `data.out_docs`.
