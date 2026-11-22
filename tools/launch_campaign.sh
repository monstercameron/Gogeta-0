#!/usr/bin/env bash
# Run a campaign: a list of experiment launches, one after another, each through
# tools/launch_experiment.sh --foreground --wait-cool (so each run takes and releases the single
# GPU slot and passes the launcher's preflight). Inside WSL.
#
#   bash tools/launch_campaign.sh <campaign-file> [--dry-run] [--foreground] [--resume]
#                                 [--poll S] [--gap S] [--stop-on-fail]
#   bash tools/launch_campaign.sh --status [NAME]      # campaigns, their progress and current item
#   bash tools/launch_campaign.sh --wait NAME          # block until campaign NAME's runner exits
#   bash tools/launch_campaign.sh --stop NAME          # SIGTERM the runner's process group (and its run)
#
# Campaign file (experiments/launch/campaigns/<name>.campaign): one launch per line,
#   <EXPERIMENT_ID> [--seed N] [--lr X] [--max-tokens N] [--no-compile]   [# comment]
# Blank lines and lines starting with # are ignored. The runner adds --foreground --wait-cool.
#
# Behaviour:
#   - Detached by default (setsid nohup): survives the shell or agent session that started it
#     (not `wsl --shutdown`, a reboot, or sleep that stops the WSL VM). One campaign at a time.
#   - Before each item it waits (polling every --poll s, default 60) while the GPU is busy: the
#     GPU slot is held, a training run is alive, nvidia-smi lists any compute process, or the
#     GPU is >= 85 C. After each item it sleeps --gap s (default 60) to let the GPU cool.
#   - Launcher exit 3 (preflight refused): if the GPU became busy (a race), wait and retry;
#     otherwise (dirty code paths, no prediction, ...) every later item would fail too, so the
#     campaign aborts. Any other non-zero exit marks the item failed and moves on (or aborts
#     with --stop-on-fail).
#   - The campaign file is snapshotted at start, so editing it later does not change a running
#     campaign. --resume skips items that already finished with exit 0 in the state file.
#
# Files ($DATA = $GOGETA_DATA, default ~/fusiondance-data; NAME = file name without extension):
#   runner log  $DATA/logs/campaign-NAME-<stamp>.log        (progress, one line per event)
#   item output $DATA/logs/campaign-NAME-<stamp>-<i>.out    (launcher output; training also has its own log)
#   PID file    $DATA/run/campaign-NAME.pid  (+ .info; .exit when the runner ends)
#   state       $DATA/run/campaign-NAME.state  (one line per finished item: idx rc start end args)
#
# Exit codes: 0 started / all items exit 0; 1 error; 2 bad usage; 3 aborted on a preflight
#             failure; 5 finished with failed items.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SELF="$REPO/tools/launch_campaign.sh"
LAUNCHER="$REPO/tools/launch_experiment.sh"
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
LOG_DIR=$DATA/logs
GPU_LOCK=$RUN_DIR/gpu.lock
TEMP_HOT=85

usage() { sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "launch_campaign: $1" >&2; exit "${2:-1}"; }
ts() { date '+%Y-%m-%d %H:%M:%S'; }
cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null | sed 's/ $//'; }
alive_with() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && [[ "$(cmdline "$1")" == *"$2"* ]]; }
NVSMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
[[ -x "$NVSMI" ]] || NVSMI=""

# ----------------------------------------------------------------------------- status / stop / wait
campaign_pid() { local f=$RUN_DIR/campaign-$1.pid p; [[ -f "$f" ]] || return 1; p=$(tr -dc '0-9' <"$f")
                 alive_with "$p" launch_campaign && echo "$p"; }

show_status() {
  local f n p any=0
  for f in "$RUN_DIR"/campaign-*.pid; do
    [[ -e "$f" ]] || continue
    n=$(basename "$f" .pid); n=${n#campaign-}
    [[ -n "${1:-}" && "$n" != "$1" ]] && continue
    any=1
    if p=$(campaign_pid "$n"); then echo "RUNNING  $n  pid $p  up $(ps -o etime= -p "$p" | tr -d ' ')"
    else echo "ended    $n  exit $(cat "$RUN_DIR/campaign-$n.exit" 2>/dev/null || echo '? (killed)')"; fi
    sed -n 's/^/         /p' "$RUN_DIR/campaign-$n.info" 2>/dev/null
    if [[ -f "$RUN_DIR/campaign-$n.state" ]]; then
      echo "         finished items (idx rc start end args):"
      sed 's/^/           /' "$RUN_DIR/campaign-$n.state"
    fi
    local log; log=$(sed -n 's/^log=//p' "$RUN_DIR/campaign-$n.info" 2>/dev/null)
    [[ -f "$log" ]] && { echo "         last events:"; tail -n 4 "$log" | sed 's/^/           | /'; }
  done
  (( any )) || echo "no campaigns${1:+ named $1}"
}

case "${1:-}" in
  --status) show_status "${2:-}"; exit 0 ;;
  --stop)   [[ -n "${2:-}" ]] || die "--stop needs a campaign name" 2
            p=$(campaign_pid "$2") || die "campaign $2 is not running"
            echo "stopping campaign $2 (process group $p): SIGTERM"; kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p"
            for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || { echo "stopped"; exit 0; }; sleep 1; done
            echo "still alive after 30 s; sending SIGKILL"; kill -KILL -- "-$p" 2>/dev/null || kill -KILL "$p"; exit 0 ;;
  --wait)   [[ -n "${2:-}" ]] || die "--wait needs a campaign name" 2
            p=$(campaign_pid "$2") || { echo "campaign $2 is not running (exit $(cat "$RUN_DIR/campaign-$2.exit" 2>/dev/null || echo '?'))"; exit 0; }
            while kill -0 "$p" 2>/dev/null; do sleep 30; done
            rc=$(cat "$RUN_DIR/campaign-$2.exit" 2>/dev/null || echo '?'); echo "campaign $2 ended, exit $rc"
            [[ "$rc" =~ ^[0-9]+$ ]] && exit "$rc"; exit 1 ;;
  -h|--help|"") usage; [[ -n "${1:-}" ]]; exit $? ;;
esac

# ----------------------------------------------------------------------------- arguments
FILE="" DRY=0 FG=0 RESUME=0 POLL=60 GAP=60 STOPFAIL=0 CHILD=0 STAMP="" CNAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)      DRY=1; shift ;;
    --foreground)   FG=1; shift ;;
    --resume)       RESUME=1; shift ;;
    --poll)         POLL=${2:-}; shift 2 || die "--poll needs a value" 2 ;;
    --gap)          GAP=${2:-}; shift 2 || die "--gap needs a value" 2 ;;
    --stop-on-fail) STOPFAIL=1; shift ;;
    --_child)       CHILD=1; STAMP=${2:-}; CNAME=${3:-}; shift 3 || die "--_child needs STAMP NAME" 2 ;;   # internal
    -*)             usage >&2; die "unknown option: $1" 2 ;;
    *)              [[ -z "$FILE" ]] || die "unexpected argument: $1" 2; FILE=$1; shift ;;
  esac
done
[[ -n "$FILE" ]] || { usage >&2; die "missing campaign file" 2; }
[[ "$POLL" =~ ^[0-9]+$ && "$GAP" =~ ^[0-9]+$ ]] || die "--poll/--gap must be whole seconds" 2
[[ "$FILE" = /* ]] || FILE="$PWD/$FILE"
[[ -f "$FILE" ]] || die "campaign file not found: $FILE"
NAME=$(basename "$FILE"); NAME=${NAME%.*}
(( CHILD )) && NAME=$CNAME   # the runner reads a snapshot; keep the campaign's own name
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "campaign name must be [A-Za-z0-9._-]: $NAME" 2
[[ -x "$LAUNCHER" || -f "$LAUNCHER" ]] || die "launcher not found: $LAUNCHER"

# ----------------------------------------------------------------------------- parse + validate
ITEMS=()
parse_file() {
  local src=$1 n=0 line t eid i
  ITEMS=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    n=$((n + 1))
    line=${line%%#*}; line=$(echo "$line" | tr -s ' \t' ' ' | sed 's/^ //; s/ $//')
    [[ -z "$line" ]] && continue
    read -ra t <<<"$line"
    eid=${t[0]^^}
    [[ "$eid" =~ ^E[0-9]+$ ]] || die "$src line $n: must start with an experiment id (E10, E1 ...): $line" 2
    i=1
    while (( i < ${#t[@]} )); do
      case "${t[$i]}" in
        --seed|--lr|--max-tokens) (( i + 1 < ${#t[@]} )) || die "$src line $n: ${t[$i]} needs a value" 2
                                  i=$((i + 2)) ;;
        --no-compile)             i=$((i + 1)) ;;
        *) die "$src line $n: option ${t[$i]} not allowed in a campaign (allowed: --seed --lr --max-tokens --no-compile)" 2 ;;
      esac
    done
    if [[ " $line " == *" --lr "* ]] && ! grep -q -- '--lr)' "$LAUNCHER"; then
      die "$src line $n uses --lr, but $LAUNCHER has no --lr option yet (see experiments/launch/campaigns/README.md)" 2
    fi
    ITEMS+=("$eid ${t[*]:1}")
  done <"$src"
  (( ${#ITEMS[@]} )) || die "$src has no launches" 2
}
parse_file "$FILE"

# ----------------------------------------------------------------------------- GPU-busy test
# Prints the reason and returns 0 when the GPU is busy; returns 1 when it is free.
gpu_busy() {
  local holder f p apps t
  if [[ -f "$GPU_LOCK" ]]; then
    holder=$(sed -n 's/^pid=//p' "$GPU_LOCK" | head -1)
    alive_with "$holder" gogeta-gpu-slot && { echo "GPU slot held by $(sed -n 's/^experiment=//p' "$GPU_LOCK") (pid $holder)"; return 0; }
  fi
  for f in "$RUN_DIR"/train-*.pid; do
    [[ -e "$f" ]] || continue
    p=$(tr -dc '0-9' <"$f")
    alive_with "$p" gogeta-gpu-slot && { echo "training run alive: $(basename "$f" .pid)"; return 0; }
  done
  p=$(pgrep -f -- '-m gogeta[.]train' | head -1)
  [[ -n "$p" ]] && { echo "gogeta.train process pid $p"; return 0; }
  if [[ -n "$NVSMI" ]]; then
    apps=$("$NVSMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -dc '0-9\n' | grep . | tr '\n' ' ')
    [[ -n "$apps" ]] && { echo "GPU compute process(es): pid $apps"; return 0; }
    t=$("$NVSMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
    [[ "$t" =~ ^[0-9.]+$ ]] && awk -v a="$t" -v b="$TEMP_HOT" 'BEGIN { exit !(a >= b) }' && { echo "GPU at ${t} C (>= $TEMP_HOT C)"; return 0; }
  fi
  return 1
}

# ----------------------------------------------------------------------------- dry run
if (( DRY )); then
  echo "== campaign $NAME: ${#ITEMS[@]} launch(es) from $FILE  [DRY RUN, nothing executed]"
  i=0
  for it in "${ITEMS[@]}"; do
    i=$((i + 1))
    read -ra a <<<"$it"
    out=$(bash "$LAUNCHER" "${a[@]}" --foreground --wait-cool --dry-run 2>&1); rc=$?
    echo "-- [$i/${#ITEMS[@]}] $it   (launcher dry run: exit $rc)"
    grep -E '^(config|command) ' <<<"$out" | sed 's/^/     /'
    grep -E '\[(FAIL|WARN)\]' <<<"$out" | sed 's/^/     /'
  done
  if why=$(gpu_busy); then echo "GPU now: busy ($why); a real campaign would wait."; else echo "GPU now: free"; fi
  echo "A real campaign runs these in order, waiting for the GPU before each one."
  exit 0
fi

mkdir -p "$RUN_DIR" "$LOG_DIR" || die "cannot create $RUN_DIR / $LOG_DIR"
PIDFILE=$RUN_DIR/campaign-$NAME.pid
STATE=$RUN_DIR/campaign-$NAME.state

# ----------------------------------------------------------------------------- start (detach)
if (( ! CHILD )); then
  for f in "$RUN_DIR"/campaign-*.pid; do
    [[ -e "$f" ]] || continue
    n=$(basename "$f" .pid); n=${n#campaign-}
    p=$(campaign_pid "$n") && die "campaign $n is already running (pid $p); one campaign at a time. Stop it: bash tools/launch_campaign.sh --stop $n" 3
  done
  STAMP=$(date +%Y%m%d-%H%M%S)
  SNAP=$RUN_DIR/campaign-$NAME-$STAMP.items
  cp "$FILE" "$SNAP" || die "cannot snapshot $FILE"
  (( RESUME )) || { [[ -f "$STATE" ]] && mv "$STATE" "$STATE.$STAMP.old"; }
  rm -f "$RUN_DIR/campaign-$NAME.exit" "$PIDFILE"   # a stale PID file must not be mistaken for the new runner
  LOG=$LOG_DIR/campaign-$NAME-$STAMP.log
  {
    echo "campaign=$NAME"; echo "file=$FILE"; echo "snapshot=$SNAP"; echo "log=$LOG"; echo "started=$(date -Is)"
    echo "git=$(git -C "$REPO" describe --tags --always --dirty 2>/dev/null)"; echo "items=${#ITEMS[@]}"
  } >"$RUN_DIR/campaign-$NAME.info"
  ARGS=("$SNAP" --_child "$STAMP" "$NAME" --poll "$POLL" --gap "$GAP")
  (( RESUME )) && ARGS+=(--resume)
  (( STOPFAIL )) && ARGS+=(--stop-on-fail)
  if (( FG )); then
    echo "campaign $NAME: running in the foreground (log $LOG)"
    bash "$SELF" "${ARGS[@]}" 2>&1 </dev/null | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
  fi
  setsid nohup bash "$SELF" "${ARGS[@]}" >>"$LOG" 2>&1 </dev/null &
  for _ in $(seq 1 30); do [[ -s "$PIDFILE" ]] && break; sleep 0.2; done
  p=$(tr -dc '0-9' <"$PIDFILE" 2>/dev/null)
  [[ -n "$p" ]] || die "the runner did not start; see $LOG"
  echo "Campaign $NAME started (detached): ${#ITEMS[@]} launch(es)."
  echo "  pid     : $p"
  echo "  log     : $LOG"
  echo "  status  : bash tools/launch_campaign.sh --status $NAME     (and bash experiments/launch/status.sh)"
  echo "  wait    : bash tools/launch_campaign.sh --wait $NAME"
  echo "  stop    : bash tools/launch_campaign.sh --stop $NAME"
  exit 0
fi

# ----------------------------------------------------------------------------- the runner
echo $$ >"$PIDFILE"
CUR=""
on_term() { log "received SIGTERM: stopping${CUR:+ (current item: $CUR)}"; echo 143 >"$RUN_DIR/campaign-$NAME.exit"; exit 143; }
trap on_term TERM INT
log() { echo "[$(ts)] $*"; }
done_ok() { [[ -f "$STATE" ]] && awk -v i="$1" -v a="$2" -F '\t' '$1 == i && $2 == 0 && $5 == a { f = 1 } END { exit !f }' "$STATE"; }

log "campaign $NAME: ${#ITEMS[@]} launch(es), poll ${POLL}s, gap ${GAP}s, git $(git -C "$REPO" describe --tags --always --dirty 2>/dev/null)"
NFAILED=0 i=0
for it in "${ITEMS[@]}"; do
  i=$((i + 1))
  if (( RESUME )) && done_ok "$i" "$it"; then log "[$i/${#ITEMS[@]}] $it: already done (exit 0), skipped"; continue; fi
  read -ra a <<<"$it"
  attempts=0
  while :; do
    last=""
    while why=$(gpu_busy); do
      [[ "$why" != "$last" ]] && log "[$i/${#ITEMS[@]}] waiting: $why"
      last=$why; sleep "$POLL"
    done
    OUT=$LOG_DIR/campaign-$NAME-$STAMP-$i.out
    CUR=$it; start=$(date -Is)
    log "[$i/${#ITEMS[@]}] launch: $it  (output $OUT)"
    bash "$LAUNCHER" "${a[@]}" --foreground --wait-cool >"$OUT" 2>&1 </dev/null
    rc=$?
    tlog=$(sed -n 's/.*in the foreground (log: \(.*\))\..*/\1/p' "$OUT" | head -1)
    if (( rc == 3 )); then
      attempts=$((attempts + 1))
      if why=$(gpu_busy) && (( attempts < 20 )); then
        log "[$i/${#ITEMS[@]}] preflight refused while the GPU became busy ($why); retrying (attempt $attempts)"
        continue
      fi
      log "[$i/${#ITEMS[@]}] preflight REFUSED and the GPU is free: a persistent problem. Aborting the campaign. Failed checks:"
      grep -E '\[FAIL\]|launch_experiment:' "$OUT" | sed 's/^/      /'
      printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$rc" "$start" "$(date -Is)" "$it" >>"$STATE"
      echo 3 >"$RUN_DIR/campaign-$NAME.exit"; exit 3
    fi
    break
  done
  CUR=""
  printf '%s\t%s\t%s\t%s\t%s\n' "$i" "$rc" "$start" "$(date -Is)" "$it" >>"$STATE"
  if (( rc == 0 )); then
    log "[$i/${#ITEMS[@]}] done: $it  (training log ${tlog:-?}); $(grep -m1 '^FINAL ' "$OUT" | cut -c1-160)"
  else
    NFAILED=$((NFAILED + 1))
    log "[$i/${#ITEMS[@]}] FAILED with exit $rc: $it  (training log ${tlog:-?}). Last lines:"
    tail -n 6 "$OUT" | sed 's/^/      | /'
    if (( STOPFAIL )); then log "--stop-on-fail: aborting"; echo 5 >"$RUN_DIR/campaign-$NAME.exit"; exit 5; fi
  fi
  (( i < ${#ITEMS[@]} && GAP > 0 )) && sleep "$GAP"
done
rc=0; (( NFAILED )) && rc=5
log "campaign $NAME finished: ${#ITEMS[@]} item(s), $NFAILED failed"
echo "$rc" >"$RUN_DIR/campaign-$NAME.exit"
exit "$rc"
