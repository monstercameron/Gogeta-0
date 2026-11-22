# Proposed ledger rows C180-C189 (A5, reproducibility)

Not merged: the main agent appends these to `experiments/claims.md` (append-only ledger) when it cuts the version.
Statuses are what the evidence supports today; evidence paths exist in this repo.

| ID | Claim | Source | Status | Evidence / check |
|---|---|---|---|---|
| C180 | Every finished run (12: E0, E1 x6, E2, E10 x4) has a generated manifest with code, config, dataset and checkpoint hashes, and for all 12 the exact code is recoverable (commit present, `code_dirty` false at start) | `repro/manifest.py` | `VERIFIED` | `repro/manifests/*.json` (12 files), `repro/test_repro.py` selftests (13 OK). Skipped on purpose: runs without a `done` summary (E9 first attempt, in-flight E10 seed, smoke) |
| C181 | The run-time torch / CUDA / driver / GPU / pip-freeze of the existing 12 runs is unrecoverable: `gogeta/tracking.py` did not record it, so those manifest fields read "unknown (not recorded at run time)". Only the at-manifest-time environment is on record | manifest `environment` blocks | `VERIFIED` (a limitation) | `repro/manifests/E0-20261122-122059-s0.json`; fix for future runs: `notes/patches/repro-manifest-at-run-end.patch` |
| C182 | A CPU fp32 replay from E0's manifest reproduces the logged step-0 training loss (9.0899 vs 9.0900), so seed handling, model init and data order at HEAD match the original run. It tests no training step | `repro/cpu_replay.py` (65 s, 1.6 GB RAM) | `VERIFIED` (step 0 only) | run on 2026-11-22 against `E0-20261122-122059-s0`; also supports the init/data part of C91 |
| C183 | Two GPU runs with the same seed and the same code are NOT bitwise identical under the default (fast) settings | C91 replay (2.8370 vs 2.8460 at step 100) is suggestive but the code differed | `UNVERIFIED` | SQ140 |
| C184 | The provisional reproduction tolerance abs(d val_loss) <= 0.004 is wider than the same-seed run-to-run spread and narrower than the effects we claim | `notes/reproducibility.md` tolerance block | `UNVERIFIED` | SQ140 (same-seed spread), Phase 2b / A2 (seed sd) |
| C185 | With `--deterministic` (CUBLAS workspace + `use_deterministic_algorithms` + cudnn flags) an attention-only config (E0) reproduces bitwise on this GPU | torch determinism docs | `UNVERIFIED` | SQ140 arm E0-det |
| C186 | The fla Triton GDN kernels stay nondeterministic (or autotune-dependent) even under `torch.use_deterministic_algorithms`, so E1-type runs cannot be bitwise today | Triton autotune / atomics reasoning | `UNVERIFIED` | SQ140 arm E1-det |
| C187 | The C91 step-100 gap (2.8370 vs 2.8460) is GPU nondeterminism, not a change to the loop body | C91 | `UNVERIFIED` | SQ141 (same code twice; old code vs new code) |
| C188 | The registry has internal inconsistencies today: 12 lint failures at HEAD 642e455 (E0/E1/E2 lack a DEVLOG `### <ID>` block; K7-parity results were committed before its PREDICTION section; K9 and SQ70 have no PREDICTION section; 6 done queue items cite no version) | `repro/lint_science.py` | `VERIFIED` | `repro/lint-report.json` |
| C189 | A blind replicator with a fresh clone and fresh venv reproduces E0's headline within the registered tolerances using manifests alone | `notes/reproducibility.md` section 5 | `UNVERIFIED` | SQ142 (CPU tier now), full GPU tier later |
