#!/usr/bin/env bash
# Stop a run started by tools/launch_experiment.sh or a job started by tools/launch_cpu_job.sh
# (inside WSL). Kills only processes recorded in the launchers' own PID files.
#
#   bash experiments/launch/stop.sh E3                    # the live training run of E3
#   bash experiments/launch/stop.sh E3-s1-20261122-1830   # a specific run (name from status.sh)
#   bash experiments/launch/stop.sh cpu:k4                # live CPU job(s) whose name starts with k4
#
# Sends SIGTERM to the job's process group, waits up to 20 s, then SIGKILL. The training wrapper
# frees the GPU slot on SIGTERM; after a SIGKILL this script removes the lock itself.
set -uo pipefail

DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
CPU_DIR=$RUN_DIR/cpu
GPU_LOCK=$RUN_DIR/gpu.lock

target=${1:-}
[[ -n "$target" && "$target" != -h && "$target" != --help ]] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null; }

if [[ "$target" == cpu:* ]]; then
  pattern="$CPU_DIR/${target#cpu:}*.pid"; marker=gogeta-cpu-job
else
  t=${target^^}; [[ "$t" =~ ^E[0-9]+$ ]] && t="$t-s"
  pattern="$RUN_DIR/train-${t}*.pid"; marker=gogeta-gpu-slot
  [[ "$target" =~ ^[Ee][0-9]+- ]] && pattern="$RUN_DIR/train-${target^}*.pid"
fi

matches=()
for f in $pattern; do
  [[ -e "$f" ]] || continue
  p=$(tr -dc '0-9' <"$f")
  if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && [[ "$(cmdline "$p")" == *"$marker"* ]]; then
    matches+=("$f:$p")
  fi
done
if (( ${#matches[@]} == 0 )); then echo "No live job matches '$target'. See: bash experiments/launch/status.sh"; exit 1; fi
if [[ "$target" != cpu:* && ${#matches[@]} -gt 1 ]]; then
  echo "More than one live run matches '$target'; name one:"; for m in "${matches[@]}"; do echo "  $(basename "${m%%:*}" .pid)"; done; exit 1
fi

for m in "${matches[@]}"; do
  f=${m%%:*} p=${m##*:}
  echo "Stopping $(basename "$f" .pid) (pid $p, process group $p)"
  kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"
  for _ in $(seq 1 200); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$p" 2>/dev/null; then
    echo "  still alive after 20 s: SIGKILL"; kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p"; sleep 0.5
  fi
  if [[ "$marker" == gogeta-gpu-slot && -f "$GPU_LOCK" ]] && grep -qx "pid=$p" "$GPU_LOCK"; then
    rm -f "$GPU_LOCK"; echo "  removed GPU slot lock"
  fi
  echo "  stopped. Record it in the registry / lab notebook if the run is abandoned (AGENTS.md 5)."
done
