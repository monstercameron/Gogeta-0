# Quick review 2026-11-23-1759 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- SQ75 is still queued to measure E0/E1/E2 kernel shares; C119 remains unverified. The profile’s bucket names use regex heuristics, so the saved top kernel names will matter when checking the result. [experiments/QUEUE.md; experiments/checks/sq_hw_step_profile.py; C119]
- The latest hardware notes describe training as bandwidth-bound, but label the bandwidth model as a hand count with about ±30% error. SQ75 may test the kernel mix, but that alone does not measure achieved memory bandwidth. [notes/hardware-spark.md]
- C190’s recorded timing is batch-1 and launch-bound, while SQ75 profiles B=32 training steps. Those measurements describe different operating regimes and shouldn’t be treated as contradictory. [C190; experiments/checks/sq_hw_step_profile.py]

## Ideas

- **[speculation] Check profiler sensitivity to arm order.** **Hypothesis:** thermal or clock changes across sequential E0/E1/E2 profiling could shift wall time and busy fraction. **Cheapest test:** profile a seeded ~13M model for about 8 GPU minutes in forward and reverse arm order; compare per-arm wall and busy times. **Change the plan if:** any arm shifts by over 10%; randomize or interleave arms in later comparisons.
- **[speculation] Validate bucket classifications against an independent timeline.** **Hypothesis:** regex buckets could put material kernels in the wrong category and alter the roofline verdict. **Cheapest test:** on one ~13M arm, compare the saved top-30 names and bucket totals with a GPU timeline capture; about 5–10 GPU minutes. **Change the plan if:** reclassification moves a decision threshold across C119’s pass/refute criteria.

## Question

- Does SQ75’s verdict remain the same when GPU busy time is cross-checked against an independent timeline capture?
## Triage
- Profiler sensitivity to arm order: accepted into SQ75's queue note. It profiles E0->E2 and then E2->E0, and flags any arm shifting by more than 10%.
- Validate the regex buckets against an independent timeline: accepted into SQ75's queue note. It saves the top-30 kernel names with their bucket labels and cross-checks the bucket totals against an nsys or CUDA-event timeline on one arm, reporting whether any reclassification crosses C119's pass/refute criteria.
- Question (does SQ75's verdict hold when cross-checked against an independent timeline?): unknown until it runs. SQ75 has not run, because no GPU is rented and it waits on the next batch. The cross-check is now part of its definition, so the verdict will be reported only together with that check. Observation 3 is agreed: C190 (batch-1 decode, launch-bound) and SQ75 (B = 32 training) measure different regimes.
