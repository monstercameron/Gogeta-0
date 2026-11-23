# Quick review 2026-11-23-1704 - systems/hardware engineer (Codex GPT-6 Luna high)

## Observations

- B1’s 20M-token screens used 2–3 seeds, and none of its three Holm-gated contrasts passed; treat the hardware implications of those architecture comparisons as provisional. [DEVLOG.md, v0.5.36]
- SQ75 is the next GPU pick: its 8-minute profiler run should test C119’s memory/elementwise-bound hypothesis and help rank follow-up hardware checks. [experiments/QUEUE.md; C119]
- GDN decode measured 0.036 ms per token, but the recurrent-kernel timing is batch-1 and launch-bound; the activation-memory scaling evidence also comes from a single run. [C111; C190]

## Idea

- **[speculation] Measure profiler overhead alongside SQ75.** Hypothesis: tracing changes the step time or kernel mix enough to distort bottleneck rankings. **Cheapest test:** GPU, about 2 minutes at 13M scale; compare warmed unprofiled steps with profiled steps on the same config. **Change the plan if:** tracing adds over 10% to step time; use profiler buckets for categorization only, then confirm leading bottlenecks with focused timings.

## Question

- Will SQ75 record unprofiled step time as a baseline for its kernel breakdown?
## Triage
- Profiler overhead alongside SQ75: accepted and written into SQ75's queue note (via `tools/queue.py`). SQ75 records the warmed, unprofiled median step time next to the profiled breakdown and reports the overhead. If tracing adds more than 10%, the buckets are used for categorisation only and the leading bottleneck is confirmed with focused timings.
- Question (will SQ75 record an unprofiled baseline?): yes, as of this triage; it is part of SQ75's definition now. Note that SQ75 is the next GPU pick but needs a box: no GPU is rented, and it would ride on the next batch.
