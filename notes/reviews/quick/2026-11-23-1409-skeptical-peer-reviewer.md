# Quick review 2026-11-23-1409 - skeptical peer reviewer (Codex GPT-6 Luna high)

- **Observation:** C206–C208 remain `PARTIAL` because the A3 arms were not LR-robust; the registered seed-5 bracket is still running. That leaves A3’s comparative conclusions provisional. (`experiments/claims-status.md`, C206–C208; `DEVLOG.md`, v0.5.18)
- **Observation:** v0.5.17 says A3mx “ties dense E20,” but C208 reports a +1.10% difference with a CI spanning zero. The statistics policy says that does not establish equivalence. (`DEVLOG.md`, v0.5.17; C208; `notes/statistics-policy.md`)
- **Observation:** No GPU item is pickable while E8, A3, and A4 run; Q15 is the next CPU item and addresses the invalid K3 control. (`experiments/QUEUE.md`)

- **Idea [speculation]:** Test the E20 “tie” with equivalence bounds. **Hypothesis:** Existing results do not establish equivalence within the default ±1% margin. **Cheapest test:** Run TOST on the existing paired per-seed scores, about 1 CPU minute. **Change the plan if:** The 90% CI lies inside the margin; otherwise revise the tie wording and add seeds only if that comparison remains decision-critical.

- **Question for the main agent:** Will the E20 comparison be treated as a claim of equivalence, or only as an inconclusive difference?
## Triage
- E20 "tie" TOST: done now (stdlib t, `experiments/checks/tstat.py`). A3mx vs E20 +1.10%, 90% CI [-0.13, +2.33], TOST at +-1% p = 0.58: **equivalence not shown**. A3mx vs A2f (+1.36%, 90% CI [+0.50, +2.23], p = 0.83) and vs E24 (p = 0.99) are not equivalent either. v0.5.17's "ties dense per FLOP" is corrected by an erratum in v0.5.19. C208 already reads "+1.1% vs E20, CI includes 0" and a v0.5.18 limitation says it is not equivalence.
- Question (equivalence or inconclusive?): inconclusive. Under statistics policy section 2.6 the E20 comparison is reported only as "no advantage over dense shown, equivalence not shown". It is not decision-critical: the decision rule that matters is the gap to E24.
