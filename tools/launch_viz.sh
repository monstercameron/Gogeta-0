#!/usr/bin/env bash
# Start / stop the local architecture visualiser for Gogeta-0 (run inside WSL).
#
#   bash tools/launch_viz.sh start  [--port N] [--timeout S]
#   bash tools/launch_viz.sh stop   [--port N]
#   bash tools/launch_viz.sh status [--port N]
#   bash tools/launch_viz.sh url    [--port N]
#
# - Default port 7870. Localhost only: tools/viz/server.py binds 127.0.0.1.
# - CPU only: CUDA_VISIBLE_DEVICES is empty and threads are capped at 2 (the GPU belongs to training).
# - `start` is idempotent: if the viz already answers on the port, it prints the URL and exits 0.
# - Runs detached (setsid nohup), so it survives the shell / agent session. PID file:
#   $DATA/run/viz-<port>.pid, log: $DATA/logs/viz-<port>.log ($DATA = $GOGETA_DATA, default ~/fusiondance-data).
# - `stop` only kills the process in its own PID file.
#
# Exit codes: 0 ok / viz up; 1 error or viz down; 2 bad usage.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
LOG_DIR=$DATA/logs
PORT=${GOGETA_VIZ_PORT:-7870}
TIMEOUT=90

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "launch_viz: $*" >&2; exit 1; }

cmd=${1:-}
[[ -n "$cmd" ]] || { usage; exit 2; }
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)    PORT=${2:?--port needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "launch_viz: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "launch_viz: bad port: $PORT" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "launch_viz: bad timeout: $TIMEOUT" >&2; exit 2; }
command -v curl >/dev/null || die "curl is required (apt-get install curl)"

URL="http://127.0.0.1:$PORT"
PIDFILE=$RUN_DIR/viz-$PORT.pid
LOGFILE=$LOG_DIR/viz-$PORT.log

# 0 = our viz answers; 1 = something else answers on the port; 2 = nothing answers.
probe() {
  local body code
  body=$(curl -s -m 3 "$URL/api/health" 2>/dev/null) && code=0 || code=$?
  if [[ $code -eq 0 ]]; then
    grep -q "gogeta-viz" <<<"$body" && return 0
    return 1
  fi
  return 2
}

port_owner() {
  ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true
}

# PID from our PID file if that process is alive and still our server on this port; empty otherwise.
our_pid() {
  [[ -f "$PIDFILE" ]] || return 0
  local pid; pid=$(tr -dc '0-9' <"$PIDFILE")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
  local args; args=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$args" == *"tools/viz/server.py"* && "$args" == *"--port $PORT"* ]] && echo "$pid"
  return 0
}

do_status() {
  local pid owner rc=0
  pid=$(our_pid); owner=$(port_owner)
  probe || rc=$?
  case $rc in
    0) echo "UP    $URL  (Gogeta-0 architecture viz)" ;;
    1) echo "BUSY  port $PORT answers, but not with the viz" ;;
    *) echo "DOWN  nothing answers on $URL" ;;
  esac
  if [[ -n "$pid" ]]; then
    echo "      started by this script: pid $pid (PID file $PIDFILE)"
  elif [[ -n "$owner" ]]; then
    echo "      pid $owner: $(tr '\0' ' ' <"/proc/$owner/cmdline" 2>/dev/null | sed 's/ $//')"
    echo "      (not started by this script: no PID file; 'stop' will leave it alone)"
  fi
  [[ -f "$PIDFILE" && -z "$pid" ]] && echo "      stale PID file: $PIDFILE"
  local f p
  for f in "$RUN_DIR"/viz-*.pid; do
    [[ -e "$f" && "$f" != "$PIDFILE" ]] || continue
    p=$(tr -dc '0-9' <"$f")
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      echo "      also running: $(basename "$f" .pid) (pid $p)"
    fi
  done
  [[ $rc -eq 0 ]]
}

do_start() {
  local rc=0 pid
  probe || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "Viz already running on port $PORT; not starting another."
    echo "$URL"
    return 0
  fi
  [[ $rc -eq 1 ]] && die "port $PORT is in use by something that is not the viz (pid $(port_owner)); pick another --port"

  mkdir -p "$RUN_DIR" "$LOG_DIR"
  pid=$(our_pid)
  if [[ -z "$pid" ]]; then
    rm -f "$PIDFILE"
    [[ -x "$PY" ]] || die "venv python not found: $PY (see AGENTS.md section 6, 'Training environment setup')"
    [[ -f "$REPO/tools/viz/server.py" ]] || die "missing $REPO/tools/viz/server.py"
    echo "=== $(date -Is) launch_viz start: port $PORT" >>"$LOGFILE"
    cd "$REPO" || die "cannot cd to $REPO"
    # setsid: own session, survives this shell. The inner bash writes its PID, then execs the server (same PID).
    CUDA_VISIBLE_DEVICES="" setsid nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" \
        OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
        bash -c 'echo $$ >"$0"; exec "$@"' "$PIDFILE" \
        "$PY" tools/viz/server.py --port "$PORT" \
        >>"$LOGFILE" 2>&1 </dev/null &
    for _ in $(seq 1 50); do [[ -s "$PIDFILE" ]] && break; sleep 0.1; done
    pid=$(tr -dc '0-9' <"$PIDFILE" 2>/dev/null || true)
    [[ -n "$pid" ]] || die "viz did not write its PID file ($PIDFILE); see $LOGFILE"
    echo "Started viz: pid $pid, log $LOGFILE"
  else
    echo "Viz process $pid (from $PIDFILE) is running but not answering yet; waiting."
  fi

  local waited=0
  while (( waited < TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Viz exited during startup. Last log lines ($LOGFILE):" >&2
      tail -n 20 "$LOGFILE" >&2 || true
      rm -f "$PIDFILE"
      exit 1
    fi
    probe && { echo "Viz is up."; echo "$URL"; return 0; }
    sleep 1; waited=$((waited + 1))
  done
  die "viz (pid $pid) did not answer on $URL within ${TIMEOUT}s; it is still running. Check $LOGFILE, or stop it with: bash tools/launch_viz.sh stop --port $PORT"
}

do_stop() {
  local pid owner
  pid=$(our_pid)
  if [[ -z "$pid" ]]; then
    [[ -f "$PIDFILE" ]] && { echo "Removing stale PID file $PIDFILE"; rm -f "$PIDFILE"; }
    owner=$(port_owner)
    if [[ -n "$owner" ]]; then
      echo "Port $PORT is served by pid $owner, which this script did not start (no PID file). Leaving it alone." >&2
      echo "If you really want it gone: kill $owner" >&2
      exit 1
    fi
    echo "No viz on port $PORT; nothing to stop."
    return 0
  fi
  echo "Stopping viz pid $pid (port $PORT)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Still alive after 10s; sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$PIDFILE"
  echo "=== $(date -Is) launch_viz stop: pid $pid" >>"$LOGFILE" 2>/dev/null || true
  echo "Stopped."
}

do_url() {
  echo "$URL"
  probe || { echo "launch_viz: viz is not answering on port $PORT (start it: bash tools/launch_viz.sh start --port $PORT)" >&2; exit 1; }
}

case "$cmd" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  url)    do_url ;;
  -h|--help|help) usage ;;
  *) echo "launch_viz: unknown command: $cmd" >&2; usage; exit 2 ;;
esac
