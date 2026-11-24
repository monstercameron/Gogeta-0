# Quick review 2026-11-23-1925 - skeptical peer reviewer (Codex GPT-6 Luna high)

## Observations

- C209’s reported gain is large, but its paired interval is based on three training seeds, and A2s has about 15.1M unique non-embedding parameters versus E27’s 9.5M. The claim is therefore specific to this recipe and comparison, not yet evidence of a parameter-matched advantage. [experiments/claims-status.md, C209]
- The C1 prediction and decision rules were amended in v0.5.51 before training began, according to the devlog. That preserves the result’s preregistration chronology; the added LR check is explicitly only a one-seed check. [DEVLOG.md, v0.5.51]
- The queue marks SQ419 running and lists no pickable GPU work; Q15 is the next CPU item. The devlog’s GPU-time figures are estimates based on prior throughput, so they do not establish SQ419’s actual progress. [experiments/QUEUE.md; DEVLOG.md, v0.5.51]

## Idea

- **[speculation] Add document uncertainty to C209’s seed interval.** Hypothesis: resampling both training seeds and clean80 documents may reveal more uncertainty than the three-seed interval alone. **Cheapest test:** CPU, about 10 minutes at the 13M-parameter scale; use existing per-document scores for A2s and E27 and run a paired seed-by-document bootstrap. **Change the plan if:** the combined interval crosses zero; qualify C209 and avoid treating its seed-only interval as the full uncertainty.

## Question

- Is C209 intended to support a conclusion about training-seed variation alone, or about variation across both seeds and evaluation documents?

## Triage
- Idea (seed x document bootstrap for C209): accepted as an extension of SQ213's two-way bootstrap, which already covered C214 since the 17:14 statistician review. It now also covers A2s vs E27 on clean80, from SQ239's per-document re-score. A crossing of 0 is very unlikely, because the gap is about 16.7% and every per-seed effect lies between -16.1% and -17.1%. The interval width is still worth reporting.
- Question (seed variation alone, or seeds and documents?): as stated, C209 is about training-seed variation on a fixed evaluation set (clean80 of the v2 val split). Generalisation to other documents is a separate claim, tested out of corpus (C213 and C214, the SQ99 set). The SQ213 extension adds the document component for the in-corpus set, so both intervals are reported.
- Observation 3: right. The C1 timings are estimates, and the controller's pull and the readout will give the measured wall time.
