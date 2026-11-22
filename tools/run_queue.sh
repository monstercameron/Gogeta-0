#!/usr/bin/env bash
# Run experiment configs one after another (inside WSL), each refusing to start on dirty code.
#   bash tools/run_queue.sh configs/e0_dense.json configs/e1_hybrid.json@1 ...
# An optional "@<seed>" suffix overrides train.seed for that entry.
# Logs: ~/fusiondance-data/logs/<config>[-s<seed>].log
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
LOGS=${GOGETA_DATA:-$HOME/fusiondance-data}/logs
mkdir -p "$LOGS"
for entry in "$@"; do
  cfg=${entry%@*}
  seed_args=()
  name=$(basename "$cfg" .json)
  if [[ "$entry" == *@* ]]; then
    seed=${entry##*@}
    seed_args=(--seed "$seed")
    name="$name-s$seed"
  fi
  echo "[$(date +%H:%M:%S)] START $entry"
  $PY -m gogeta.train --config "$cfg" --compile --require-clean "${seed_args[@]}" > "$LOGS/$name.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $entry rc=$rc :: $(grep -E '^(FINAL|run:)' "$LOGS/$name.log" | tr '\n' ' ')"
done
