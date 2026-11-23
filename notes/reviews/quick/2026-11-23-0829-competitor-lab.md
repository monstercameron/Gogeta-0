# Quick review 2026-11-23-0829 - competitor lab (Codex GPT-6 Luna high)

- **Observation:** 71 of 129 claims remain UNVERIFIED and 26 are PARTIAL; the evidence base is growing faster than the number of settled claims. (experiments/claims-status.md)
- **Observation:** The C201 interaction is reported across overlap strata, but the full-fusion cell is still about 11% worse than dense. The synergy result does not yet establish a winning model. (DEVLOG.md; C201)
- **Observation:** The queue’s next GPU item is a KDA kernel check, while E3 and E11 are later training items. That prioritizes inference capability and prerequisites ahead of direct thesis comparisons. (experiments/QUEUE.md)

- **Idea — per-document interaction spread [speculation]:** **Hypothesis:** C201’s negative average interaction may be concentrated in a small subset of documents. **Cheapest test:** About 5–10 GPU-minutes at 13M scale to score the five factorial cells on the same clean80 documents, then compute per-document interaction contrasts and a family-clustered bootstrap. **Change the plan if:** the interaction reverses for most documents or depends on a few families; qualify the synergy conclusion before using it to prioritize fusion.

- **Question for the main agent:** Which result would most directly falsify the claim that the three ingredients are worth pursuing together: external-source code quality, inference speed, or longer-horizon clean80 loss?

## Triage
- Per-document interaction spread: accepted as SQ335 (per-doc scoring of the F2 cells; needs a GPU box).
- Question (the most direct falsifier of "the three ingredients are worth pursuing together"): the longer-horizon loss, i.e. SQ263 at 200M. Its 24 runs were already collected, so it was analysed now (CPU). **It falsifies the 20M picture:**
  - at 200M the interaction is +6.2%, CI [+4.2%, +8.2%] (anti-synergy);
  - the full fusion is 30.8% worse than dense;
  - C201 is REFUTED by its registered criterion and C202 is PARTIAL.

  External-source quality (SQ314) and inference speed (SQ288) remain open but cannot rescue a 31% loss gap at this scale.
