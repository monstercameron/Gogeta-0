#!/usr/bin/env bash
# Run a small CPU-only job detached (inside WSL), within the compute budget (AGENTS.md 5):
# at most 8 CPU jobs alive at once, each limited to 2 threads, and no GPU.
#
#   bash tools/launch_cpu_job.sh [--name NAME] [--dry-run] -- <command...>
#   e.g. bash tools/launch_cpu_job.sh --name k4 -- ~/.venvs/gogeta/bin/python experiments/checks/k4_ternary_gap_vs_size.py
#
# - Env: CUDA_VISIBLE_DEVICES="" (no GPU), OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=2, PYTHONUNBUFFERED=1.
# - Runs from the repo root, detached (setsid nohup), so it survives the session that started it.
# - Files in $DATA/run/cpu/ ($DATA = $GOGETA_DATA, default ~/fusiondance-data):
#     <name>-<timestamp>.pid / .log / .info / .exit (exit code, written when the job ends)
# - Refuses when 8 CPU jobs are already alive; stale PID files (dead PIDs) are cleaned up.
# - Refuses GPU training (`gogeta.train`): that goes through tools/launch_experiment.sh.
#
# Exit codes: 0 started (or dry run that would start); 1 error; 2 bad usage; 3 refused (budget).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
CPU_DIR=$DATA/run/cpu
MAX_JOBS=8
THREADS=2

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "launch_cpu_job: $1" >&2; exit "${2:-1}"; }

NAME="" DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)    NAME=${2:-}; shift 2 || die "--name needs a value" 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; break ;;
    *)         usage >&2; die "unknown option: $1 (put the command after --)" 2 ;;
  esac
done
[[ $# -gt 0 ]] || { usage >&2; die "missing command after --" 2; }
CMD=("$@")

case " ${CMD[*]} " in
  *gogeta.train*|*launch_experiment*) die "GPU training must go through tools/launch_experiment.sh (one GPU slot), not the CPU queue" 2 ;;
esac

if [[ -z "$NAME" ]]; then
  # default name: the script a python command runs, else the program name
  NAME=$(basename "${CMD[0]}")
  if [[ "$NAME" == python* && ${#CMD[@]} -gt 1 ]]; then
    for a in "${CMD[@]:1}"; do [[ "$a" == -* ]] && continue; NAME=$(basename "$a" .py); break; done
  fi
fi
NAME=$(tr -c 'A-Za-z0-9._-' '_' <<<"$NAME" | sed 's/_$//' | cut -c1-48)
[[ -n "$NAME" ]] || NAME=job
STAMP=$(date +%Y%m%d-%H%M%S)
BASE=$CPU_DIR/$NAME-$STAMP
PIDFILE=$BASE.pid LOG=$BASE.log EXITFILE=$BASE.exit INFOFILE=$BASE.info

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null | sed 's/ $//'; }

# Lists live jobs as "pidfile<TAB>pid"; removes stale PID files unless this is a dry run.
STALE=()
live_jobs() {
  local f p
  STALE=()
  for f in "$CPU_DIR"/*.pid; do
    [[ -e "$f" ]] || continue
    p=$(tr -dc '0-9' <"$f")
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && [[ "$(cmdline "$p")" == *gogeta-cpu-job* ]]; then
      printf '%s\t%s\n' "$f" "$p"
    else
      STALE+=("$f")
    fi
  done
}

ENV_ARGS=(CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
          OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1)

print_plan() {
  echo "== Gogeta-0 CPU job: $NAME  [$1]"
  echo "cpu jobs   : $NLIVE/$MAX_JOBS alive$( (( ${#STALE[@]} )) && echo ", ${#STALE[@]} stale PID file(s) $( (( DRY )) && echo '(a real launch removes them)' || echo 'removed')")"
  echo "cwd        : $REPO"
  echo "env        : CUDA_VISIBLE_DEVICES=\"\" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS PYTHONUNBUFFERED=1"
  printf 'command    :'; printf ' %q' "${CMD[@]}"; echo
  echo "log        : $LOG"
  echo "pid file   : $PIDFILE"
}

if (( DRY )); then
  LIVE=$(live_jobs); live_jobs >/dev/null   # second call fills STALE in this shell
  NLIVE=$(grep -c . <<<"$LIVE")
  print_plan "DRY RUN"
  if (( NLIVE >= MAX_JOBS )); then
    echo "result     : DRY RUN, nothing executed. A real launch would REFUSE: $NLIVE/$MAX_JOBS CPU jobs alive."
    exit 3
  fi
  echo "result     : DRY RUN, nothing executed. A real launch would START."
  exit 0
fi

mkdir -p "$CPU_DIR" || die "cannot create $CPU_DIR"
exec 9>"$CPU_DIR/.launch.lock"
flock -w 30 9 || die "another CPU-job launch is in progress (lock $CPU_DIR/.launch.lock)"
LIVE=$(live_jobs); live_jobs >/dev/null
NLIVE=$(grep -c . <<<"$LIVE")
for f in "${STALE[@]}"; do rm -f "$f"; done
if (( NLIVE >= MAX_JOBS )); then
  print_plan "REFUSED" >&2
  echo "result     : not started: $NLIVE/$MAX_JOBS CPU jobs already alive (AGENTS.md 5 compute budget)." >&2
  exit 3
fi

{
  echo "name=$NAME"; echo "log=$LOG"; echo "started=$(date -Is)"
  printf 'cmd='; printf ' %q' "${CMD[@]}"; echo
} >"$INFOFILE"
{
  echo "=== $(date -Is) launch_cpu_job: $NAME"
  printf '=== cmd:'; printf ' %q' "${CMD[@]}"; echo
  echo "=== env: CUDA_VISIBLE_DEVICES=\"\" OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=$THREADS"
} >>"$LOG"

# Wrapper ($0 = gogeta-cpu-job, visible in ps): writes its PID, releases the launch lock,
# runs the job, records the exit code.
WRAPPER='
echo $$ >"$CJ_PIDFILE"
{ flock -u 9; exec 9>&-; } 2>/dev/null || true
"$@"
rc=$?
echo "$rc" >"$CJ_EXITFILE"
echo "=== $(date -Is) launcher: job exited with code $rc"
exit $rc
'
cd "$REPO" || die "cannot cd to $REPO"
setsid nohup env "${ENV_ARGS[@]}" CJ_PIDFILE="$PIDFILE" CJ_EXITFILE="$EXITFILE" \
  bash -c "$WRAPPER" gogeta-cpu-job "${CMD[@]}" >>"$LOG" 2>&1 </dev/null &
for _ in $(seq 1 50); do [[ -s "$PIDFILE" ]] && break; sleep 0.1; done
PID=$(tr -dc '0-9' <"$PIDFILE" 2>/dev/null)
flock -u 9; exec 9>&-
[[ -n "$PID" ]] || die "the job did not write its PID file ($PIDFILE); see $LOG"

sleep 1
if kill -0 "$PID" 2>/dev/null; then
  echo "Started CPU job $NAME ($((NLIVE + 1))/$MAX_JOBS alive)."
else
  echo "CPU job $NAME already finished (exit code $(cat "$EXITFILE" 2>/dev/null || echo '?'))."
fi
echo "  pid      : $PID"
echo "  log      : $LOG"
echo "  follow   : tail -f $LOG"
echo "  pid file : $PIDFILE"
