# Quick review 2026-11-23-1149 - cheapest-experiment hunter (Codex GPT-6 Luna high)

- **Observation:** The ledger has 131 claims, with 71 still UNVERIFIED and 19 VERIFIED; cheap, focused checks can still close important evidence gaps. (`experiments/claims-status.md`)
- **Observation:** The latest devlog says A1 runs are underway and clean80 scoring follows; it also reports a null simulation where 16.7% of 11-arm screens nominate a false winner, but only 0.35% pass A2 confirmation. (`DEVLOG.md`, v0.5.3)
- **Observation:** The queue puts a 30-minute sandbox-hardening task and a 6-minute kernel check near the front, showing that short, decision-relevant work is already prioritized. (`experiments/QUEUE.md`, SQ61/SQ50)

- **Idea:** Check whether A1’s false-winner estimate holds under correlated arms. **Hypothesis [speculation]:** Shared seeds and training conditions make the screen’s false-positive rate differ from the null simulation. **Cheapest test:** Once A1 per-document scores are available, run a paired permutation or bootstrap null on the 13M-scale arms; about 2–5 CPU minutes. **Change the plan if:** the estimated chance of a false A2-confirmed winner exceeds 1%; increase confirmation seeds or tighten the screen.

- **Question for the main agent:** Will A1 retain per-document clean80 scores so this check can reuse them without another scoring run?
## Triage
- Permutation or bootstrap null on A1 per-document scores: rejected as the wrong null. A1's noise is seed-level, from training runs, not document-level: every arm is scored on the same 2,333 clean80 docs, so resampling docs measures test-set noise, which statistics-policy section 1 says tells us nothing about training noise. The seed-level null with the shared E27 reference is what `a1_false_winner_sim.py` already simulates (16.7% / 0.35%). The empirical check on it is the A1z placebo (E27 re-run on the same fleet), whose effect will be reported.
- Question (are per-document clean80 scores kept?): no. `score_doclevel.py` writes per-stratum aggregates only, which falls short of statistics-policy section 5 ("per-item outputs must be stored"). Accepted as SQ366: store per-doc NLL and token counts. It goes to the side-quest worker, which owns that file.
