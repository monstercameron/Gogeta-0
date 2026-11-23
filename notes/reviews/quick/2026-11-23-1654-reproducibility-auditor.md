# Quick review 2026-11-23-1654 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- The claims status lists 141 claims, including 74 `UNVERIFIED`, and several scope fields still marked TBD. That makes precise scoping and evidence pointers a large part of the audit burden. [experiments/claims-status.md]
- B1’s comparisons and Holm family were documented as registered before results, and the devlog reports 8 of 23 runs complete. The reviewed files do not yet provide per-run artifact hashes or result-level evidence for this batch. [DEVLOG.md, v0.5.34]
- C212’s status records that two seeds were used to select and test A1a, while the fresh seed alone gave -3.11%. The selection caveat is visible in the claim record, but it limits how the reported CI should be interpreted. [C212]

## Ideas

- **[speculation] Audit evidence-pointer integrity.** Hypothesis: some ledger evidence links may be missing or stale, making the claim status harder to reproduce. **Cheapest test:** CPU, about 2 minutes; check that every evidence path in a sample of 13M-scale claims resolves and that cited result files contain the reported values. **Change the plan if:** any pointer is broken or a value mismatches; prioritize a ledger-wide integrity check before promoting claims.
- **[speculation] Re-run one registered B1 screen from saved inputs.** Hypothesis: a small replay can reveal whether config, seed, and data provenance are sufficient to reproduce a reported metric. **Cheapest test:** GPU, about 3 minutes at 13M-parameter scale; rerun one completed screen with the recorded config and seed, then compare its saved metric to the original. **Change the plan if:** the difference exceeds the existing same-seed replay floor; require stronger run-level provenance or investigate determinism before using B1 screens to choose follow-ups.

## Question

- Will B1’s result bundle preserve each run’s exact config, seed, data identity, and output path so its registered contrasts can be independently checked?
## Triage
- Evidence-pointer integrity audit: already SQ274 (release-lint evidence integrity, with its C209 manual trail and the 15:52 promotion gate). Not added again.
- Re-run one B1 screen from saved inputs: covered by SQ232's same-seed replay floor on this GPU class (0.2-0.3% at 20M, v0.4.79), which is the threshold the B1 readout quotes. A third replay adds little.
- Question (does B1's bundle preserve exact config, seed, data identity and output path per run?): yes.
  - `experiments/checks/results/b1_runs.json` lists, per run: name, arm, seed, lr, `core_lr_mult`, tokens, clean80, FLOPs, params, commit and `git describe` (907065f for the 23 registered jobs, c384e67 for SQ350), dataset (code-py-v2), finish time.
  - Each run's `summary.json` holds the full resolved config, and the doc-level scorer records its hashes.
  - The configs are committed (`experiments/staged-configs/b1/`, sha256-16 in PREDICTIONS.md), and the box verified v2's file hashes against the committed manifest before training.
  - The readout script `experiments/checks/b1_readout.py` recomputes every contrast from `b1_runs.json`.
