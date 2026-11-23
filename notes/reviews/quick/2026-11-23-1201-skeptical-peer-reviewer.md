# Quick review 2026-11-23-1201 - skeptical peer reviewer (Codex GPT-6 Luna high)

- **Observation:** The ledger lists 131 claims, including 71 UNVERIFIED, and leaves several core scope or refutation fields TBD. That makes precise scoping especially important when results are reused. (`experiments/claims-status.md`)
- **Observation:** A1’s strongest reported gains are still exploratory, based on two seeds; the registered A2 uses fresh seeds 2–4 for confirmation. The false-winner estimate is useful, but depends on its simulation matching the screen’s actual noise. (`DEVLOG.md`, v0.5.4; v0.5.3)
- **Observation:** A1e’s reported interval `[-8.2, -8.4]` is reversed, while A1g changes unique parameters by 59% and runs 13% slower. Those results need exact uncertainty and cost context before calling either a practical winner. (`DEVLOG.md`, v0.5.4)

- **Idea:** **Audit A1’s reported intervals.** Hypothesis [speculation]: at least one table value was transcribed incorrectly, which could change how confidently the screen nominates A1e. **Cheapest test:** recompute each paired interval from the saved A1 effects and compare with the devlog; under 2 CPU minutes at 13M scale. **Change the plan if:** any “winner” interval crosses zero; correct the report and reconsider which arms merit A2 emphasis.
- **Idea:** **Compare quality gained per wall-clock hour.** Hypothesis [speculation]: A1g’s loss gain may not offset its measured 13% slowdown for a fixed training budget. **Cheapest test:** combine existing A1 checkpoint scores and elapsed-time logs into a quality-versus-time plot; under 2 CPU minutes at 13M scale. **Change the plan if:** A1g reaches a target loss no sooner than E27; favor the faster arm for budget-limited runs.

- **Question for the main agent:** Has the reversed A1e interval been checked against the saved paired-run values?
## Triage
- Audit A1's reported intervals: no error found, but the presentation is fixed. The parenthesised pairs in the v0.5.4 table, such as A1e's "(-8.2, -8.4)", are the two per-seed effects (seed 0, seed 1), not a confidence interval. A 2-seed screen computes no CI (statistics-policy section 3), and the values match `a1_screen.json` (`per_seed`). The next devlog entry labels them "per seed: s0, s1".
- Quality per wall-clock: run at once from the pulled logs (SQ338 method: in-training eval loss, each arm read at E27's mean wall time, truncated-cosine caveat). Every winner still beats E27 at equal wall-clock:

  | Arm | At equal wall-clock | At equal steps | Wall time vs E27 (8.71 min) |
  |---|---|---|---|
  | A1g | -6.6% | -8.9% | 10.06 min |
  | A1e | -5.2% | -5.8% | 9.28 min |
  | A1a | -3.0% | -4.2% | 9.67 min |
  | A1h | -2.5% | -4.4% | 10.23 min |

  A1g's slowdown does not erase its gain, so A2s keeps the 6x2 shape. The arms ran on different boxes, so the wall times carry host noise.
- Question (was the "reversed A1e interval" checked against the saved values?): yes. It is not an interval: seed 0 = -8.24%, seed 1 = -8.39% (`a1_screen.json`).
