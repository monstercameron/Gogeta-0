# Quick review 2026-11-23-0144 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- The ledger reports 71 of 127 claims as `UNVERIFIED`, and many definitions still have `TBD` scope fields. A claim’s status alone may not give a reproducer enough information to reconstruct its conditions. (`experiments/claims-status.md`)
- The v0.4.92 entry records the F2 design’s aggregate allocation and notes that the primary contrasts use 37 shared seeds. It makes a key design limitation visible without reporting cell outcomes. (`DEVLOG.md`, v0.4.92)
- The v0.4.93 entry reports F2 stage 2 running while the queue’s next local GPU pick is E9. That distinction is useful, but readers need to know the queue is not a full view of active fleet work. (`DEVLOG.md`, v0.4.93; `experiments/QUEUE.md`)

## Ideas

- **Scope completeness spot check.** [speculation] Hypothesis: resolving or explicitly retaining unknown scope fields for high-priority claims will prevent later comparisons from silently changing what a claim means. **Cheapest test:** 3 CPU-min to inspect the definitions and evidence pointers for three queued claim updates. **Change the plan if:** a claim’s dataset, architecture, or model-size scope cannot be reconstructed; record the gap before updating its status.
- **Seeded replay smoke.** [speculation] Hypothesis: a fresh process using the same commit, config, data order, and seed reproduces early metrics within a stated tolerance on a 13M-parameter run. **Cheapest test:** 5 GPU-min for two short, same-seed launches and a step-by-step metric comparison. **Change the plan if:** the divergence exceeds tolerance; document nondeterminism and avoid relying on exact replay as a validity check.

## Question

- Could a future result record include a compact source commit, config hash, data version, and seed bundle so these can be checked together during reproduction?

## Triage
- Scope completeness: accepted as SQ319. 117 of 127 ledger definitions have a TBD scope field (counted from `experiments/claims/ledger/*.jsonl`). The fill uses `claims_list.py erratum --kind tbd_fill`, settled claims first.
- Seeded replay smoke: rejected as already measured. SQ307 re-ran SQ243's E10 d=256 seeds 2 and 3 (same config and seed, later commit, different box): clean80 0.3325 -> 0.3320 and 0.3369 -> 0.3367, so GPU replay differs by <= 0.15%, which is nondeterministic kernels, not bit-exact. CPU bit-identity of the training path is pinned by `tests/test_options.py` (state_dict sha256 + 8 losses as float hex). The 0.15% replay noise should be quoted next to sub-0.5% effects.
- Question (commit + config hash + data version + seed bundle): mostly present. Each run's `meta.json` records the git commit and dirty flags, the full resolved config, `cli_overrides`, the seed and `dataset_meta`. Doclevel score files name the checkpoint and run. Missing: the dataset manifest sha256 in the run record (tracked in SQ299) and a config sha in the run summary. Worth adding with SQ299.
