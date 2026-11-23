# Quick review 2026-11-23-1659 - competitor lab (Codex GPT-6 Luna high)

## Observations

- B1’s 20M-token contrasts use only 2–3 seeds; none of its three Holm-gated contrasts passed. Treat its architectural readings as screening evidence, not confirmed wins. [DEVLOG.md, v0.5.36]
- The queue puts E11 ahead of follow-ups on the GPU, while Q15 is next on CPU. That keeps the immediate work focused on the negative-eigenvalue GDN question and repairing the invalid MQAR control. [experiments/QUEUE.md]
- The claims ledger has 74 `UNVERIFIED` claims, and several definitions still lack scope or refutation criteria. That makes precise claim boundaries a major reproducibility risk. [experiments/claims-status.md]

## Idea

- **[speculation] Validate whether short screens predict long runs.** Hypothesis: 20M-token rankings may not reliably forecast 200M outcomes. **Cheapest test:** CPU, about 5–10 minutes at 13M scale; compare early and final paired-seed rankings for architectures with existing run curves. **Change the plan if:** rank correlation is below 0.5 or more than a quarter of pairwise orderings flip; then use short screens only to reject candidates, not select winners.

## Question

- What evidence threshold will promote a 20M screen into a full-horizon architecture decision?
## Triage
- Observation 2 (the queue puts E11 next on GPU): a stale queue item, confirmed and fixed. E11 ran in Phase 2b and is settled as C196 (VERIFIED: equivalent to E1, TOST p 0.002, 3 seeds, v0.4.79). E11 and its seed follow-up Q17 are now marked done, and the next GPU pick is SQ75.
- Do 20M screens predict 200M outcomes: accepted as SQ414 (CPU, from existing curves). The pairs already in hand: A3m vs A2s (-1.0% at 20M, -3.3% at 200M) and A2f vs E24 (+1.8% vs +3.2%) keep their sign, while the F2 interaction flipped (C201 -> C202).
- Question (what evidence promotes a 20M screen into a full-horizon decision?): none on its own. A 20M screen can only open a registered 200M follow-up, through its pre-stated gate (for example SQ404's "closes more than half the gap"). Architecture decisions and claim promotions need the 200M horizon with at least 3 fresh seeds and a CI excluding 0 beyond the practical margin (statistics policy section 2). If SQ414 finds 20M rankings unreliable, screens will be used only to reject candidates.
