# Quick review 2026-11-23-1249 - competitor lab (Codex GPT-6 Luna high)

- **Observation:** The fusion result changes with training horizon: C201’s interaction is synergistic at 20M tokens, while C202 reports anti-synergy at 200M. That makes a single-horizon architectural verdict risky. (`experiments/claims-status.md`, C201–C202)
- **Observation:** The A2 loop go/no-go rule was written after A2 seeds 2 and 4 were seen, and the disclosed seed already favors dropping loops from the default. Treat it as a decision rule under scrutiny, not fresh confirmatory evidence. (`DEVLOG.md`, v0.5.9)
- **Observation:** BitLinear applies ternary weights through a straight-through estimator, but the inspected implementation keeps ordinary tensor weights. The headline “per bit” efficiency may therefore differ from actual checkpoint or inference memory. (`gogeta/model.py`, C4)

- **Idea:** [speculation] Measure practical ternary storage and reload parity: pack a 13M-scale checkpoint, report bytes on disk and in memory, and compare logits after unpacking. **Cheapest test:** CPU, about 10 minutes. **Change the plan if:** packing does not materially reduce stored bytes or unpacked logits miss a pre-set tolerance; then keep ternary claims about training behavior separate from deployment memory claims.
- **Idea:** [speculation] Check whether the horizon-dependent interaction appears on fresh seeds under one fixed LR rule. **Cheapest test:** GPU, about 45 minutes for a 13M-scale, three-cell, short-budget screen. **Change the plan if:** the interaction’s sign is unstable across seeds; defer recipe decisions until a powered replication.

**Question for the main agent:** What result would make the team retain loops in the default if the registered A2 comparison misses its 1% threshold?
## Triage
- Ternary storage and reload parity: accepted as SQ378 (CPU, D2). C9's packing ratios are computed, not measured, and BitLinear trains full-precision weights with fake quantisation, so a real packed artifact is needed before any memory claim.
- Horizon interaction on fresh seeds under a fixed LR rule: rejected as covered. SQ342 is the 200M LR completion on fresh seeds 3-5 under a pre-fixed rule, and SQ370 tests the cosine-schedule explanation of the 20M-to-200M sign flip.
- Question (what would keep loops in the default if A2 misses the 1% threshold?): three outcomes, each fixed in "A2 amendment 2":
  1. A3 shows a looping benefit that unique parameters do not explain: the 3x4-core arm vs the 6x2 shape at equal applications, Holm-adjusted.
  2. SQ362 shows better quality per decode-second for the looped model at batch 1, which is goal 6's actual use.
  3. The byte-refinement loop (E14, H-C2F) wins against its unshared, parameter-matched control.
  Failing all three, loops leave E61's default.
