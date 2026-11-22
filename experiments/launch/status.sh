#!/usr/bin/env bash
# What is running right now (inside WSL): GPU slot, GPU temperature and processes, training runs,
# CPU jobs (n/8) and the dashboard. Read-only: changes nothing.
#
#   bash experiments/launch/status.sh [-n LINES]     # LINES of each live log to show (default 5)
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
CPU_DIR=$RUN_DIR/cpu
GPU_LOCK=$RUN_DIR/gpu.lock
MAX_CPU=8
N=5
[[ "${1:-}" == "-n" ]] && N=${2:-5}

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null | sed 's/ $//'; }
alive_with() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && [[ "$(cmdline "$1")" == *"$2"* ]]; }
etime() { ps -o etime= -p "$1" 2>/dev/null | tr -d ' '; }
field() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -1; }
tail_log() { [[ -f "$1" ]] && tail -n "$N" "$1" | sed 's/^/        | /'; }

NVSMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
[[ -x "$NVSMI" ]] || NVSMI=""

echo "== GPU  (budget: 1 GPU job at a time)"
holder=""
if [[ -f "$GPU_LOCK" ]]; then
  holder=$(field "$GPU_LOCK" pid)
  if alive_with "$holder" gogeta-gpu-slot; then
    echo "  slot : HELD by $(field "$GPU_LOCK" experiment) seed $(field "$GPU_LOCK" seed), pid $holder, running $(etime "$holder")"
    echo "         log $(field "$GPU_LOCK" log)"
  else
    echo "  slot : free (stale lock from dead pid ${holder:-?}; the next launch removes it)"
    holder=""
  fi
else
  echo "  slot : free"
fi
if [[ -n "$NVSMI" ]]; then
  IFS=, read -r temp util mem <<<"$("$NVSMI" --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')"
  note=""
  if [[ "$temp" =~ ^[0-9.]+$ ]]; then
    awk -v t="$temp" 'BEGIN { exit !(t >= 85) }' && note="  HOT: launches refused until < 75 C"
  fi
  echo "  temp : ${temp:-?} C   util ${util:-?}%   mem ${mem:-?} MiB$note"
  apps=$("$NVSMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r')
  if [[ -z "$apps" ]]; then
    echo "  procs: none"
  else
    while IFS=, read -r apid _; do
      apid=$(tr -dc '0-9' <<<"$apid"); [[ -n "$apid" ]] || continue
      pg=$(ps -o pgid= -p "$apid" 2>/dev/null | tr -d ' ')
      if [[ -n "$holder" && "$pg" == "$holder" ]]; then who="launcher slot holder"
      else who="NOT via the launcher: counts against the 1-GPU-job budget; launches will wait for it"; fi
      echo "  proc : pid $apid  up $(etime "$apid")  $(cmdline "$apid" | cut -c1-90)"
      echo "         ($who)"
    done <<<"$apps"
  fi
else
  echo "  nvidia-smi not found: no temperature / process info"
fi

echo
echo "== Training runs (PID files in $RUN_DIR)"
shown=0
for f in $(ls -1t "$RUN_DIR"/train-*.pid 2>/dev/null); do
  base=${f%.pid}; name=$(basename "$base"); name=${name#train-}
  p=$(tr -dc '0-9' <"$f"); log=$(field "$base.info" log)
  if alive_with "$p" gogeta-gpu-slot; then
    echo "  RUNNING  $name  pid $p  up $(etime "$p")"
    echo "           log $log"; tail_log "$log"
  elif (( shown < 5 )); then
    if [[ -f "$base.exit" ]]; then echo "  ended    $name  exit $(cat "$base.exit")   log $log"
    else echo "  killed   $name  (no exit code)   log $log"; fi
  fi
  shown=$((shown + 1))
done
(( shown )) || echo "  none launched yet"
other=""
for p in $(pgrep -f -- '-m gogeta[.]train|tools/run_queue[.]sh' 2>/dev/null); do
  pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
  [[ -n "$holder" && "$pg" == "$holder" ]] && continue          # the slot holder's own training
  [[ "$(cmdline "$p")" == *gogeta-gpu-slot* ]] && continue
  other+="    pid $p  $(cmdline "$p" | cut -c1-100)"$'\n'
done
[[ -n "$other" ]] && { echo "  outside the launcher (no PID file, no GPU slot):"; printf '%s' "$other"; }

echo
alive=0 lines=""
for f in $(ls -1t "$CPU_DIR"/*.pid 2>/dev/null); do
  base=${f%.pid}; p=$(tr -dc '0-9' <"$f")
  if alive_with "$p" gogeta-cpu-job; then
    alive=$((alive + 1))
    lines+="  RUNNING  $(basename "$base")  pid $p  up $(etime "$p")"$'\n'
    lines+="           $(field "$base.info" cmd | cut -c1-100)"$'\n'
    lines+="$(tail_log "$base.log" | tail -n 2)"$'\n'
  fi
done
echo "== CPU jobs: $alive/$MAX_CPU alive  ($CPU_DIR)"
[[ -n "$lines" ]] && printf '%s' "$lines"
recent=$(ls -1t "$CPU_DIR"/*.exit 2>/dev/null | head -3)
for e in $recent; do echo "  ended    $(basename "${e%.exit}")  exit $(cat "$e")"; done

echo
echo "== Dashboard"
bash "$REPO/tools/launch_tracker.sh" status 2>&1 | sed 's/^/  /'
