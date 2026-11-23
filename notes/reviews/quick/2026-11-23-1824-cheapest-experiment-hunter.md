# Quick review 2026-11-23-1824 - cheapest-experiment hunter (Codex GPT-6 Luna high)

## Observations

- The latest entry corrected six confidence-interval scripts; C209’s interval widened to `[-17.86%, -15.46%]`, with no decision changes. [DEVLOG.md; v0.5.45]
- The claims summary lists 142 claims, including 75 UNVERIFIED and seven definitions with fields still TBD. [experiments/claims-status.md]
- SQ75 is the next GPU pick at 8 minutes; Q15 is the next CPU pick at 60 minutes. [experiments/QUEUE.md]

## Idea

- **[speculation] Check how training-step time changes with sequence length.** **Hypothesis:** dense-masked SWA makes per-token runtime grow noticeably from T=512 to T=1024, despite the fixed window. **Cheapest test:** GPU, about 5 minutes at 13M scale; profile warmed steps at both lengths with the same model and batch size. **Change the plan if:** per-token time rises by more than 50%; include measured sequence-length cost when choosing contexts for follow-up runs.

## Question

- Can the SQ75 profiler capture a second sequence length cheaply, or should this check follow it?

## Triage
- Idea (step time vs sequence length under the dense SWA mask): rejected as already measured. K1 (`experiments/checks/results/K1.json`, C12 WRONG FOR OUR CODE) found an SWA time exponent of 1.74 and a memory exponent of 1.95. That implies about 1.67x per-token time from T=512 to 1024, so the review's >50% trigger has already fired, and the plan already treats per-FLOP figures as analytic. The open follow-ups are already queued: SQ244 (wall-clock crossover at 1k-8k), SQ72 and SQ3 (flex/block-sparse SWA), and SQ310 and SQ424 (equal-time replots).
- Question (can SQ75 capture a second length?): no need. SQ75 is a per-kernel breakdown at the training length; length scaling belongs to SQ244 and SQ72, both cheap candidates for the next GPU batch (SQ244 is about 1 GPU-min).
- Observations: agreed, nothing new (v0.5.45 erratum; ledger counts; queue heads).
