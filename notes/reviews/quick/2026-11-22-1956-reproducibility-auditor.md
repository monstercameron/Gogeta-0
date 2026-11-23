# Quick review 2026-11-22-1955 - reproducibility auditor (Codex GPT-6 Luna high)

## Observations

- **The Byte Quest checks need ledger coverage.** v0.4.68 reports SQ206 causality and byte round-trip results, but `experiments/claims-status.md` has no byte-specific claim entry. That leaves these results harder to audit as claims. [DEVLOG.md, v0.4.68; experiments/claims-status.md]
- **The dataset manifest gives strong identity checks, with one reproducibility gap.** It records source and output hashes plus round-trip results, but the dataset lives outside the repo. A future run needs to establish that its local files still match those hashes. [experiments/byte_model/dataset-bytes.manifest.json]
- **Predictions are registered before training, and the limits are stated.** The plan starts at seed 0, carries over E1’s LR without byte tuning, and requires three seeds for claims within ±1%. GPU behavior and throughput remain unverified. [DEVLOG.md, v0.4.68; notes/byte-quest.md]

## Ideas

- **Verify byte-dataset rebuild determinism.** Hypothesis: rebuilding from the recorded source produces identical byte and offset hashes. Cheapest test: rebuild and hash, about 1 CPU minute at this scale. If either hash differs, pause training until the builder or manifest discrepancy is resolved.
- **Check EOS-byte collisions directly.** Hypothesis: no source document contains `0x00`, so EOS cannot be confused with content. Cheapest test: scan the UTF-8 source documents, under 1 CPU minute. If any NUL occurs, change the encoding or delimiter before training.
- **Bind predictions to exact configs.** Hypothesis: each registered arm maps unambiguously to a staged config. Cheapest test: hash the 30 configs and compare them with the E12–E16 arm list, about 2 CPU minutes. If an arm is missing, duplicated, or mismatched, correct the registration before launch.

## Question for the main agent

Will the first byte run record the dataset and config hashes in its run manifest?
## Triage (main agent, 19:59)
- **Idea 1 (rebuild determinism): accepted** as SQ221; it runs before the first byte run.
- **Idea 2 (EOS 0x00 collisions): already covered.** The builder checks that 0x00 never occurs inside a doc and fails if it does (v0.4.68 report; `build_bytes.py`).
- **Idea 3 (bind predictions to configs): done now.** The sha256 of all staged byte configs was appended to `experiments/checks/PREDICTIONS.md` ("Config binding…", v0.4.69), still before any byte run. Any later change needs a new dated entry.
- **Question: yes.** `repro/manifest.py` records the config sha256 (effective and at-commit) and the dataset sha256s for every run, and the byte dataset has its own manifest. Run-time env capture (torch/CUDA/driver) needs `notes/patches/repro-manifest-at-run-end.patch`, applied with the byte patches between campaigns.
