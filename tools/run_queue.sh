#!/usr/bin/env bash
# Run experiment configs one after another (inside WSL), each refusing to start on dirty code.
#   bash tools/run_queue.sh configs/e0_dense.json configs/e1_hybrid.json ...
# Logs: ~/fusiondance-data/logs/<config>.log
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
LOGS=${GOGETA_DATA:-$HOME/fusiondance-data}/logs
mkdir -p "$LOGS"
for cfg in "$@"; do
  name=$(basename "$cfg" .json)
  echo "[$(date +%H:%M:%S)] START $cfg"
  $PY -m gogeta.train --config "$cfg" --compile --require-clean > "$LOGS/$name.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] END   $cfg rc=$rc :: $(grep -E '^(FINAL|run:)' "$LOGS/$name.log" | tr '\n' ' ')"
done
