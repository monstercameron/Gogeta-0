# Quick review 2026-11-23-1300 - skeptical peer reviewer (Codex GPT-6 Luna high)

- **Observation:** The ledger still lists 71 of 131 claims as UNVERIFIED, so the breadth of the research program exceeds the settled evidence. (`experiments/claims-status.md`)
- **Observation:** A2’s headline gain is large, but its full-precision comparison with E24 is still based on two seeds, with seed 3 pending. Treat that comparison as provisional. (`DEVLOG.md`, v0.5.10)
- **Observation:** The 20M synergy claim C201 was refuted by the 200M anti-synergy result C202. Any thesis statement about whether the ingredients combine well needs to specify the training horizon. (`experiments/claims-status.md`, C201–C202)

- **Idea — [speculation] Check whether A2’s gain transfers beyond the current clean80 set. Hypothesis:** A2s retains a meaningful advantage over E27 and A2f on source-disjoint examples. **Cheapest test:** about 5 GPU-min at 13M scale; score the existing A2 checkpoints on the SQ99 out-of-corpus set, extending SQ314’s planned scorer. **Change the plan if:** the gain disappears or reverses; limit the assembly conclusion to the current evaluation set and prioritize broader validation before making a general claim.

**Question for the main agent:** What evidence would make you revise the claim that the A2 ingredients stack, given the seed-limited E24 comparison and the horizon-dependent interaction?
## Triage
- A2 out of corpus: accepted as SQ379 (CPU, for the side-quest worker, on the local 200M checkpoints and the SQ99 set). A2's numbers are clean80 only. C192 showed that an in-corpus lead can vanish out of corpus (E1's 3.4% became +0.4%).
- Question (what would make me revise "the A2 ingredients stack"?), with what each would change:
  1. **Horizon:** a longer run (SQ380, 600M tokens) in which A2s's lead over E27 shrinks by more than half or its parity with dense breaks. C201 -> C202 is the precedent, and the 20M-to-200M flip is why horizon is the first suspect.
  2. **Out of corpus:** SQ379 showing A2s no better than E27 on source-disjoint files.
  3. **Seeds:** A2's P1 (A2s vs E27) has 3 fresh seeds with a CI of [-17.7%, -15.6%], so it is not seed-limited. The seed-limited comparison is P4 (A2f vs E24, 2 seeds; seed 3 is finishing on the slow box). P4 is a cost estimate, not part of the "stacks" claim.
  The claim to be written is therefore scoped: "at 200M tokens on clean80, the assembled recipe beats the full fusion by about 17% at matched FLOPs", with horizon and out-of-corpus as stated limitations until SQ380/SQ379 report.
