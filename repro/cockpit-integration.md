# Cockpit integration notes (A5)

The cockpit can surface reproducibility without new data plumbing; everything is a JSON file.

* `repro/manifests/<run>.json`, one per finished run. Useful columns: `code.code_exactly_recoverable`,
  `deterministic_mode.enabled`, `environment.recorded_at_run_time.torch` (reads "unknown (not recorded at run time)" for
  the 12 back-filled runs), `artifacts.checkpoint.sha256`, `manifest_sha256`. Show the sha prefix beside each run and a
  red badge when `code_exactly_recoverable` is false.
* `repro/lint-report.json` (`python3 repro/lint_science.py --out repro/lint-report.json`, about 2 min on /mnt/c):
  `summary` per check (E experiments, C claims, P pre-registration, Q queue) and `findings[]` with `level` pass/warn/fail.
  A "registry health" tile.
* `python3 repro/reproduce.py <manifest> --json` prints the verify results (`checks`: name, MATCH|MISMATCH|UNVERIFIABLE|INFO,
  detail); with `--compare <run>` it adds `compare.verdict` (BITWISE / STATISTICAL / NOT REPRODUCED / NOT COMPARABLE).
  A "Reproduce" button can run it: it is read-only and launches nothing.
* Once `notes/patches/repro-manifest-at-run-end.patch` is applied, each new run folder holds `<run>.json` beside
  `summary.json`, and `meta.json["env"]` carries torch/CUDA/driver/GPU/pip hash: the cockpit can show run-time env per run.
* The patch applies to HEAD and also after the three cockpit patches (`cockpit-logging-train`, `cockpit-logging-tracking`,
  `cockpit-resilience-train`); checked in a scratch copy with `git apply --check`.
