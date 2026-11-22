#!/usr/bin/env bash
# Mirror every finished training run's small record files into the repo, so results live in git
# (AGENTS.md section 5: "every run's record is in the repo"). Copies only meta.json, summary.json and
# samples.txt from ~/fusiondance-data/runs/<run>/ into experiments/runs/<run>/. Never checkpoints or data.
# Runs without a summary.json (still running or killed) are skipped unless --all is given.
# Refuses to copy anything that looks like a token. Usage (WSL): bash experiments/runs/sync.sh [--all]
set -euo pipefail
SRC="${GOGETA_RUNS:-/root/fusiondance-data/runs}"
DST="$(cd "$(dirname "$0")" && pwd)"
ALL=0; [[ "${1:-}" == "--all" ]] && ALL=1
n=0; skipped=0
for d in "$SRC"/*/; do
  run="$(basename "$d")"
  if [[ ! -f "$d/summary.json" && $ALL -eq 0 ]]; then skipped=$((skipped+1)); continue; fi
  if grep -qiE 'hf_[A-Za-z0-9]{10,}|write_token' "$d"/*.json "$d"/*.txt 2>/dev/null; then
    echo "REFUSED $run: token-like string found" >&2; continue
  fi
  mkdir -p "$DST/$run"
  for f in meta.json summary.json samples.txt; do
    [[ -f "$d/$f" ]] && cp -p "$d/$f" "$DST/$run/$f"
  done
  n=$((n+1))
done
echo "synced $n run(s) into $DST (skipped $skipped without summary.json)"
