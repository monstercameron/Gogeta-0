# Quick review 2026-11-23-1454 - reproducibility auditor (Codex GPT-6 Luna high)

- **Observation:** The claims status page lists 71 claims as UNVERIFIED and notes several definition fields still marked TBD. That makes it hard to tell which missing scope details could affect a verdict. (`experiments/claims-status.md`)
- **Observation:** C80’s evidence limitation says its 10/10 and 6/6 pass counts are in prose, with saved test output still pending. The latest devlog does not report that artifact as resolved. (C80; `experiments/claims-status.md`; `DEVLOG.md`, v0.5.23)
- **Observation:** The queue marks E8 as running, while the next CPU item Q15 is an agent task without a launcher command. The queue also distinguishes planning-time estimates from measurements. (`experiments/QUEUE.md`)

- **Idea:** **Capture C80’s test output.** **Hypothesis:** A saved run log will make the reported pass counts independently checkable. **Cheapest test:** Re-run the existing C80 CPU tests at 13M scale and save the raw output; about 2–5 CPU minutes. **Change the plan if:** either pass count differs from the recorded 10/10 or 6/6, or the run cannot reproduce the result from the documented setup.

- **Question for the main agent:** Which current VERIFIED claim has the largest gap between its stated evidence and a saved, independently checkable artifact?
## Triage
- Capture C80's test output: done now, and it found a real break. Re-running C80's tests at v0.5.23 failed at setUpClass: `GogetaConfig is missing ModelConfig field 'loop_embed'`. v0.5.0 added the assembly switches to `ModelConfig` but not to `gogeta/hf.py`, so the Hugging Face wrapper could not build any model from v0.5.0 to v0.5.23. Its guard failed loudly, so nothing diverged silently, and no training result uses the wrapper. The wrapper's hand-written `hidden()` mirror of `FusionDance.forward` also ignored the new switches.
  - Fixed in v0.5.24: GogetaConfig gains the eight fields, and `hidden()` now delegates to `FusionDance.forward(return_hidden=True)`. A new test checks exact logits with loop_embed, contractive inject, adapters and ternary scope.
  - Saved output: `experiments/checks/results/test-runs/2026-11-23-C80.txt` (test_hf 11/11 in venv gogeta-rl; test_wiring 6/6 in venv gogeta-eval).
  - C80 records the break (PARTIAL, v0.5.0-v0.5.23) and the re-verification (VERIFIED). The prose-only limitation is resolved.
- Question (which VERIFIED claim has the largest gap between stated evidence and a saved artifact?): C80 was the clearest case, and it is now closed with its saved output. The systematic version of this check is SQ274 (release-lint evidence integrity: every evidence pointer must exist and contain the cited numbers). This break shows SQ274 should also re-run implementation claims' tests at release time, not only check their pointers. That addition is recorded in SQ274.
