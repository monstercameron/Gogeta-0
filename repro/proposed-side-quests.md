# Proposed side quests SQ140-SQ146 (A5, reproducibility)

Rows for `experiments/side-quests.md` (the main agent merges). GPU items need `notes/patches/repro-manifest-at-run-end.patch`
applied and committed first (it adds `--deterministic`, run-time env capture and in-run manifests).

| ID | Side quest | Why | Status | Notes |
|---|---|---|---|---|
| SQ140 | **Same-seed determinism**: E0 and E1, default vs `--deterministic`, each run twice (seed 0, 10M tokens); compare step losses, final val loss, checkpoint sha256, per-tensor max diff; one extra E0 pair with `--compile` off | C183/C185/C186, and the data behind the 0.004 tolerance (C184) | `idea` (accepted) | GPU ~25-30 min; prediction in `repro/proposed-predictions.md`; queue item in `repro/proposed-queue.json` |
| SQ141 | **C91 closed**: replay E0's first 100 steps twice with the SAME current code and once with the v0.3.0 code (`git archive` into scratch, no worktree); is 2.8370 vs 2.8460 inside the same-code spread? | C187 | `idea` (accepted) | GPU ~8 min; can ride in SQ140 (its E0-default pair at step 100) |
| SQ142 | **Blind replication dry run, CPU tier**: fresh clone + fresh venv from the lock files, `reproduce.py` then `cpu_replay.py` for E0; report every MATCH/MISMATCH incl. the pip-freeze delta | C189 | `idea` (accepted) | CPU ~20 min incl. venv build (a network install: a main-agent/human step) |
| SQ143 | Fill `seed_sd_val_loss` in the tolerance block from the Phase 2b replication (3 seeds E10, E1) and re-issue the block as version 2 | C184 | `idea` (blocked on A2 / Phase 2b) | analysis only |
| SQ144 | First run that writes its own manifest (patch applied): check it equals what `manifest.py` would back-fill, except the env block | validates the patch | `idea` | free: any run after the patch |
| SQ145 | fla autotune pinning: fix the Triton configs of chunk GDN and re-run the E1 pair of SQ140 | only if C186 is confirmed | `idea` (conditional) | GPU ~15 min |
| SQ146 | Re-hash all dataset files and checkpoints periodically with `reproduce.py --all-data` (bit-rot / accidental-edit guard) | provenance maintenance | `idea` | CPU, minutes |
