# Quick review 2026-11-23-1519 - skeptical peer reviewer (Codex GPT-6 Luna high)

- **Observation:** The ledger has 74 UNVERIFIED claims out of 141; the project’s own accounting shows much remains unsettled. (`experiments/claims-status.md`)
- **Observation:** SQ379 reports the ternary and loop costs nearly vanishing out of corpus, but labels that explanation hypothesis C213; the seed-variance follow-up SQ400 is still only an idea. (`DEVLOG.md`, C213, SQ400)
- **Observation:** The reported 16× packing applies to ternary layer weights; the whole packed file is 0.42× the checkpoint, and reload is close but not identical to fake-quant output. (`DEVLOG.md`, SQ378)

- **Idea — [speculation] Leave-one-package-out sensitivity:** If a few packages drive the out-of-corpus assembly gain, a package-clustered interval may hide that fragility. **Cheapest test:** After SQ400 scoring, recompute its effect while leaving out each package in turn; about 5 CPU minutes at 13M scale. **Change the plan if:** any omission erases the gain or cuts it by more than half; qualify the result and broaden the held-out corpus before relying on it.

- **Question for the main agent:** What result would make you reject the claim that the out-of-corpus gain generalizes beyond this particular package set?
## Triage
- Leave-one-package-out sensitivity: merged into SQ400 as a required readout. SQ213 already defines leave-one-source-out for out-of-corpus evals, and SQ400 must now report it for every contrast and flag any package whose removal erases or halves an effect.
- Question (what would make me reject that the out-of-corpus gain generalises beyond this package set?): any of three results.
  1. Leave-one-package-out takes the A2s-vs-E27 gain to zero or cuts it by more than half for any single package.
  2. The repo-held-out slice (SQ379's extension, a different source than the SQ99 files) gives the opposite sign or under half the effect.
  3. The paired-by-seed CI (seed variance included) crosses 0.
  Any one of these limits C213 and the out-of-corpus reading of C209 to "this package set", and adding held-out sources comes before any design change.
