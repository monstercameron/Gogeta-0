#!/usr/bin/env bash
# Start / stop the local Trackio dashboard for Gogeta-0 (run inside WSL).
#
#   bash tools/launch_tracker.sh start  [--port N] [--project P] [--timeout S]
#   bash tools/launch_tracker.sh stop   [--port N]
#   bash tools/launch_tracker.sh status [--port N]
#   bash tools/launch_tracker.sh url    [--port N]
#
# - Localhost only: tools/dashboard.py binds 127.0.0.1 with share disabled. This script never
#   enables share links or Spaces.
# - `start` is idempotent: if the port already serves the dashboard, it prints the URL and exits 0.
# - The dashboard runs detached (setsid nohup), so it survives the shell / agent session that
#   started it. PID file: $DATA/run/tracker-<port>.pid, log: $DATA/logs/tracker-<port>.log
#   ($DATA = $GOGETA_DATA, default ~/fusiondance-data).
# - `stop` only kills the process recorded in this script's own PID file, never a dashboard
#   that was started some other way.
#
# Exit codes: 0 ok / dashboard up; 1 error or dashboard down; 2 bad usage.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
LOG_DIR=$DATA/logs
PORT=${GOGETA_DASHBOARD_PORT:-7861}
PROJECT=gogeta-0
TIMEOUT=90

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "launch_tracker: $*" >&2; exit 1; }

cmd=${1:-}
[[ -n "$cmd" ]] || { usage; exit 2; }
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)    PORT=${2:?--port needs a value}; shift 2 ;;
    --project) PROJECT=${2:?--project needs a value}; shift 2 ;;
    --timeout) TIMEOUT=${2:?--timeout needs a value}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "launch_tracker: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || { echo "launch_tracker: bad port: $PORT" >&2; exit 2; }
[[ "$TIMEOUT" =~ ^[0-9]+$ ]] || { echo "launch_tracker: bad timeout: $TIMEOUT" >&2; exit 2; }
[[ "$PROJECT" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "launch_tracker: bad project name: $PROJECT" >&2; exit 2; }
command -v curl >/dev/null || die "curl is required (apt-get install curl)"

URL="http://127.0.0.1:$PORT"
PIDFILE=$RUN_DIR/tracker-$PORT.pid
LOGFILE=$LOG_DIR/tracker-$PORT.log

# 0 = Trackio dashboard answers; 1 = something else answers on the port; 2 = nothing answers.
probe() {
  local body code
  body=$(curl -s -m 3 "$URL/" 2>/dev/null) && code=0 || code=$?
  if [[ $code -eq 0 ]]; then
    grep -q "Trackio Dashboard" <<<"$body" && return 0
    return 1
  fi
  return 2
}

# PID listening on the port, if `ss` can see it (empty otherwise).
port_owner() {
  ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true
}

# PID from our PID file if that process is alive and is a dashboard on this port; empty otherwise.
our_pid() {
  [[ -f "$PIDFILE" ]] || return 0
  local pid; pid=$(tr -dc '0-9' <"$PIDFILE")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
  # guard against PID reuse: the process must still be our dashboard
  local args; args=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$args" == *"tools/dashboard.py"* && "$args" == *"--port $PORT"* ]] && echo "$pid"
  return 0
}

do_status() {
  local pid owner rc=0
  pid=$(our_pid); owner=$(port_owner)
  probe || rc=$?
  case $rc in
    0) echo "UP    $URL  (Trackio dashboard)" ;;
    1) echo "BUSY  port $PORT answers, but not with the Trackio dashboard" ;;
    *) echo "DOWN  nothing answers on $URL" ;;
  esac
  if [[ -n "$pid" ]]; then
    echo "      started by this script: pid $pid (PID file $PIDFILE)"
  elif [[ -n "$owner" ]]; then
    echo "      pid $owner: $(tr '\0' ' ' <"/proc/$owner/cmdline" 2>/dev/null | sed 's/ $//')"
    echo "      (not started by this script: no PID file; 'stop' will leave it alone)"
  fi
  [[ -f "$PIDFILE" && -z "$pid" ]] && echo "      stale PID file: $PIDFILE"
  # other dashboards this script manages
  local f p
  for f in "$RUN_DIR"/tracker-*.pid; do
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
    echo "Dashboard already running on port $PORT; not starting another."
    echo "$URL"
    return 0
  fi
  [[ $rc -eq 1 ]] && die "port $PORT is in use by something that is not the Trackio dashboard (pid $(port_owner)); pick another --port"

  mkdir -p "$RUN_DIR" "$LOG_DIR"
  ( umask 077; touch "$LOGFILE" )   # Trackio prints a dashboard write token into this log
  pid=$(our_pid)
  if [[ -z "$pid" ]]; then
    rm -f "$PIDFILE"
    [[ -x "$PY" ]] || die "venv python not found: $PY (see AGENTS.md section 6, 'Training environment setup')"
    [[ -f "$REPO/tools/dashboard.py" ]] || die "missing $REPO/tools/dashboard.py"
    {
      echo "=== $(date -Is) launch_tracker start: port $PORT, project $PROJECT"
    } >>"$LOGFILE"
    # setsid: own session, so closing this shell (or an agent session) does not kill it.
    # The inner bash writes its own PID, then execs the dashboard (same PID). Started as a direct
    # background child (no wrapping subshell), so no bash lingers waiting on it.
    cd "$REPO" || die "cannot cd to $REPO"
    setsid nohup env PYTHONUNBUFFERED=1 bash -c 'echo $$ >"$0"; exec "$@"' "$PIDFILE" \
        "$PY" tools/dashboard.py --project "$PROJECT" --port "$PORT" \
        >>"$LOGFILE" 2>&1 </dev/null &
    for _ in $(seq 1 50); do [[ -s "$PIDFILE" ]] && break; sleep 0.1; done
    pid=$(tr -dc '0-9' <"$PIDFILE" 2>/dev/null || true)
    [[ -n "$pid" ]] || die "dashboard did not write its PID file ($PIDFILE); see $LOGFILE"
    echo "Started dashboard: pid $pid, log $LOGFILE"
  else
    echo "Dashboard process $pid (from $PIDFILE) is running but not answering yet; waiting."
  fi

  local waited=0
  while (( waited < TIMEOUT )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Dashboard exited during startup. Last log lines ($LOGFILE):" >&2
      tail -n 20 "$LOGFILE" >&2 || true
      rm -f "$PIDFILE"
      exit 1
    fi
    probe && { echo "Dashboard is up."; echo "$URL"; return 0; }
    sleep 1; waited=$((waited + 1))
  done
  die "dashboard (pid $pid) did not answer on $URL within ${TIMEOUT}s; it is still running. Check $LOGFILE, or stop it with: bash tools/launch_tracker.sh stop --port $PORT"
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
    echo "No dashboard on port $PORT; nothing to stop."
    return 0
  fi
  echo "Stopping dashboard pid $pid (port $PORT)"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 100); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "Still alive after 10s; sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    sleep 0.5
  fi
  rm -f "$PIDFILE"
  echo "=== $(date -Is) launch_tracker stop: pid $pid" >>"$LOGFILE" 2>/dev/null || true
  echo "Stopped."
}

do_url() {
  echo "$URL"
  probe || { echo "launch_tracker: dashboard is not answering on port $PORT (start it: bash tools/launch_tracker.sh start --port $PORT)" >&2; exit 1; }
}

case "$cmd" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  url)    do_url ;;
  -h|--help|help) usage ;;
  *) echo "launch_tracker: unknown command: $cmd" >&2; usage; exit 2 ;;
esac
