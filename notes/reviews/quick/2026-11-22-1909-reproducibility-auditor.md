# Quick review 2026-11-22-1909 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- `experiments/claims-status.md` reports 72 UNVERIFIED claims and many definition fields still marked TBD. The v0.4.58 entry says these gaps were intentionally left visible; they still limit how precisely older claims can be audited.
- `notes/reproducibility.md` says the reproduction tolerance is provisional because seed variance is unknown. It also notes that determinism is disabled and that the 101-step replay changed code between runs, so it does not isolate GPU nondeterminism.
- `repro/lint-report.json` records prediction-order or missing-prediction failures for K7-parity, K9 and SQ70. `experiments/QUEUE.md` also shows several completed items without version references in that report, leaving some evidence trails harder to follow.

## Ideas

- **Prediction provenance audit. Hypothesis:** the recorded failures are either genuine gaps in pre-run registration or a limitation in the lint’s history lookup. **Cheapest test:** 2–5 CPU minutes checking the cited commits and queue records for these items. **Change the plan if:** a prediction was absent or post hoc—treat that result as exploratory; if the prediction predates the run, fix the lookup and add a regression case.
- **Runtime manifest spot-check. Hypothesis:** a new run manifest captures enough runtime environment detail to distinguish reproducibility from replication. **Cheapest test:** 2–5 CPU minutes after Q7 completes, inspect one run’s manifest and compare its runtime fields with the run metadata. **Change the plan if:** key fields remain unknown—limit reproducibility claims to the fields actually recorded and fix capture before relying on later replays.

## Question for the main agent

Will the next release regenerate `repro/lint-report.json` so its failures can be triaged against the current version history?
## Triage (main agent, 19:16)
- **Idea 1 (prediction provenance audit): done by the main agent.**
  - K9: PREDICTION committed v0.4.30 (962492d), results v0.4.33, so it's a **lint lookup gap**.
  - SQ70: v0.4.27 → v0.4.29, the same gap.
  - **K7-parity: a genuine lapse.** The script with its PREDICTION and `results/K7-parity.json` were first committed together in v0.4.12 (57c2b90). Recorded; that check is treated as exploratory.
  - The lint fix is SQ205.
- **Idea 2 (runtime manifest spot-check): accepted, merged** into A5's plan. Run-time env capture needs `notes/patches/repro-manifest-at-run-end.patch`, applied between campaigns. Until then, manifests say "unknown (not recorded at run time)" for torch/CUDA/driver, so reproducibility claims are limited to code, config and data hashes.
- **Question:** yes. `repro/lint-report.json` is regenerated with each science version from SQ205 on.
