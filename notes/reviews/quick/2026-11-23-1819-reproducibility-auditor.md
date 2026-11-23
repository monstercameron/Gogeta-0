# Quick review 2026-11-23-1819 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- **v0.5.45:** Six scripts replaced an approximate t quantile with an exact one. C209’s interval widened to `[-17.86%, -15.46%]`, with no recorded decision changes; the correction has a saved check result. [DEVLOG.md; v0.5.45]
- **Claim scope remains a reproducibility gap:** the claims summary lists seven definitions with fields still marked TBD, and 75 claims remain UNVERIFIED. [experiments/claims-status.md]
- **The queue makes the next actions auditable:** SQ75 is the next GPU pick, with a launch command; Q15 is the next CPU pick, but is listed as agent work without a launcher command. [experiments/QUEUE.md]

## Idea

- **[speculation] Independently verify the corrected confidence intervals.** **Hypothesis:** recomputing intervals from saved per-seed values with an independent exact-t implementation reproduces the shared helper’s results. **Cheapest test:** CPU, about 2 minutes; use saved 13M-scale run values and compare C209 plus the other corrected cases against SciPy. **Change the plan if:** any bound differs by more than 0.01 percentage points; audit affected intervals before relying on them.

## Question

- Can the corrected interval check be rerun from the saved per-seed values without relying on the shared helper?

## Triage
- Idea (independent check of the corrected intervals) and the question: **done this tick.** `experiments/checks/tq_independent.py` (SciPy 1.18.1, WSL venv gogeta-eval) compares `tstat.t_quantile` with `scipy.stats.t.ppf` at 11 df values, including fractional Welch df (max diff 1.6e-14). It also recomputes all seven A2 paired intervals from the saved per-seed ratios using SciPy only, without the shared helper; the largest bound difference is 2.8e-15 percentage points. ALL PASS -> `results/tq_independent.json`. Goes in with the next outside-review version.
- Observation 2 (seven definitions with TBD fields, 75 UNVERIFIED): agreed, known; UNVERIFIED is the ledger's default for claims without saved-output evidence, and TBD scope fields are filled when a claim gets evidence. No new action.
- Observation 3 (Q15 has no launcher command): by design; Q15 is a CPU authoring item done by the main agent, not a launched job.
