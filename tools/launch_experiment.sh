#!/usr/bin/env bash
# Launch one registered experiment's training run (inside WSL), the same way every time.
#
#   bash tools/launch_experiment.sh <EXPERIMENT_ID> [--seed N] [--dry-run] [--foreground]
#                                   [--no-compile] [--max-tokens N] [--wait-cool]
#
# What it does:
#   1. Looks up the experiment's config in experiments/experiments.json (fails if none).
#   2. Preflight: venv python + `import torch, gogeta` (+ CUDA, real launches only); dataset
#      meta.json; non-empty `prediction` in the registry (AGENTS.md 0/6: no prediction, no run);
#      clean code paths (what --require-clean enforces); Trackio dashboard reachable (warning
#      only); no other training run; the single GPU slot is free (AGENTS.md 5 "Compute budget":
#      at most 1 GPU job at a time); no foreign GPU compute process; GPU temperature < 85 C.
#   3. Launches `python -m gogeta.train --config <cfg> --compile --require-clean [--seed N]`
#      DETACHED (setsid nohup) with GOGETA_TRACKER=trackio, so it survives the shell or agent
#      session that started it. A wrapper holds the GPU slot and releases it when training exits.
#
# Options:
#   --seed N        override train.seed (default: the config's seed)
#   --dry-run       print the command, env, cwd, log path and preflight results; execute nothing
#   --foreground    run attached (output to the terminal and the log); still takes the GPU slot
#   --no-compile    drop --compile
#   --max-tokens N  override train.max_tokens (a shorter run than the registered config!)
#   --wait-cool     if the GPU is >= 85 C, poll until it is < 75 C instead of refusing
#
# Files ($DATA = $GOGETA_DATA, default ~/fusiondance-data):
#   log       $DATA/logs/<EID>-s<seed>-<timestamp>.log
#   PID file  $DATA/run/train-<EID>-s<seed>-<timestamp>.pid   (+ .info, and .exit when it ends)
#   GPU slot  $DATA/run/gpu.lock   (pid/experiment/seed/log of the holder)
#
# Exit codes: 0 launched (or dry run that would launch); 1 error; 2 bad usage;
#             3 preflight refused; 4 the run died right after starting.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-$HOME/.venvs/gogeta/bin/python}
DATA=${GOGETA_DATA:-$HOME/fusiondance-data}
RUN_DIR=$DATA/run
LOG_DIR=$DATA/logs
GPU_LOCK=$RUN_DIR/gpu.lock
REGISTRY=$REPO/experiments/experiments.json
DASH_PORT=${GOGETA_DASHBOARD_PORT:-7861}
DASH_URL="http://127.0.0.1:$DASH_PORT"
TEMP_REFUSE=85   # refuse to start at or above this (C)
TEMP_RESUME=75   # --wait-cool waits until below this (C)
COOL_POLL=30     # seconds between temperature readings while waiting
CODE_PATHS=(gogeta configs tools tests requirements.txt)   # same list as gogeta/tracking.py

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "launch_experiment: $1" >&2; exit "${2:-1}"; }

# ----------------------------------------------------------------------------- arguments
EID="" SEED="" SEED_GIVEN=0 DRY=0 FG=0 COMPILE=1 MAXTOK="" WAITCOOL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)       SEED=${2:-}; SEED_GIVEN=1; shift 2 || die "--seed needs a value" 2 ;;
    --max-tokens) MAXTOK=${2:-}; shift 2 || die "--max-tokens needs a value" 2 ;;
    --dry-run)    DRY=1; shift ;;
    --foreground) FG=1; shift ;;
    --no-compile) COMPILE=0; shift ;;
    --wait-cool)  WAITCOOL=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    -*)           usage >&2; die "unknown option: $1" 2 ;;
    *)            [[ -z "$EID" ]] || die "unexpected argument: $1" 2; EID=$1; shift ;;
  esac
done
[[ -n "$EID" ]] || { usage >&2; die "missing EXPERIMENT_ID (e.g. E3)" 2; }
EID=${EID^^}
[[ "$EID" =~ ^E[0-9]+$ ]] || die "experiment id must look like E3, got: $EID" 2
[[ -z "$SEED" || "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer" 2
[[ -z "$MAXTOK" || "$MAXTOK" =~ ^[1-9][0-9]*$ ]] || die "--max-tokens must be a positive integer" 2

# ----------------------------------------------------------------------------- registry lookup
command -v python3 >/dev/null || die "python3 not found (needed to read the registry)"
[[ -f "$REGISTRY" ]] || die "registry not found: $REGISTRY"
# Prints: title, status, config, prediction length, prediction preview (one per line).
REG=$(python3 -I - "$REGISTRY" "$EID" <<'PY'
import json, sys
path, eid = sys.argv[1], sys.argv[2]
try:
    reg = json.load(open(path, encoding="utf-8"))
except Exception as e:
    print(f"ERROR cannot parse {path}: {e}"); sys.exit(1)
exps = reg["experiments"] if isinstance(reg, dict) else reg
hit = [e for e in exps if str(e.get("id", "")).upper() == eid]
if not hit:
    print(f"ERROR {eid} is not in {path} (known: {', '.join(e.get('id', '?') for e in exps)})"); sys.exit(1)
e = hit[0]
one = lambda s: " ".join(str(s or "").split())
pred = one(e.get("prediction"))
print(one(e.get("title")) or "-")
print(one(e.get("status")) or "-")
print(one(e.get("config")))
print(len(pred))
print(pred[:110] + ("..." if len(pred) > 110 else ""))
PY
) || die "${REG#ERROR }"
{ read -r TITLE; read -r REG_STATUS; read -r CONFIG; read -r PRED_LEN; read -r PRED_PREVIEW; } <<<"$REG"
[[ -n "$CONFIG" ]] || die "$EID has no config in the registry: not designed yet. Add a config and a prediction first." 3
[[ -f "$REPO/$CONFIG" ]] || die "$EID's config does not exist: $REPO/$CONFIG"

# dataset + default seed from the config
CFGINFO=$(python3 -I - "$REPO/$CONFIG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
t = c.get("train", {})
print(c.get("experiment", ""))
print(t.get("dataset", ""))
print(t.get("seed", 0))
print(t.get("max_tokens", ""))
PY
) || die "cannot parse config $CONFIG"
{ read -r CFG_EID; read -r DATASET; read -r CFG_SEED; read -r CFG_MAXTOK; } <<<"$CFGINFO"
SEED=${SEED:-$CFG_SEED}

STAMP=$(date +%Y%m%d-%H%M%S)
RUNNAME=$EID-s$SEED-$STAMP
LOG=$LOG_DIR/$RUNNAME.log
PIDFILE=$RUN_DIR/train-$RUNNAME.pid
EXITFILE=$RUN_DIR/train-$RUNNAME.exit
INFOFILE=$RUN_DIR/train-$RUNNAME.info

CMD=("$PY" -m gogeta.train --config "$CONFIG")
(( COMPILE )) && CMD+=(--compile)
CMD+=(--require-clean)
(( SEED_GIVEN )) && CMD+=(--seed "$SEED")
[[ -n "$MAXTOK" ]] && CMD+=(--max-tokens "$MAXTOK")

# ----------------------------------------------------------------------------- helpers
CHECKS=() NFAIL=0 NWARN=0
pass() { CHECKS+=("  [PASS] $1: $2"); }
warn() { CHECKS+=("  [WARN] $1: $2"); NWARN=$((NWARN + 1)); }
fail() { CHECKS+=("  [FAIL] $1: $2"); NFAIL=$((NFAIL + 1)); }
skip() { CHECKS+=("  [SKIP] $1: $2"); }

cmdline() { tr '\0' ' ' <"/proc/$1/cmdline" 2>/dev/null | sed 's/ $//'; }
# true if $1 is alive and its command line contains $2 (guards against PID reuse)
alive_with() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && [[ "$(cmdline "$1")" == *"$2"* ]]; }
lock_field() { sed -n "s/^$1=//p" "$GPU_LOCK" 2>/dev/null | head -1; }

NVSMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
[[ -x "$NVSMI" ]] || NVSMI=""
gpu_temp() { [[ -n "$NVSMI" ]] && "$NVSMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r'; }
is_num() { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }

# Concurrency checks (run before the temperature wait, and again under the launch lock).
check_concurrency() {
  local f p holder others="" apps a apid
  # 1) other training runs started by this launcher
  for f in "$RUN_DIR"/train-*.pid; do
    [[ -e "$f" ]] || continue
    p=$(tr -dc '0-9' <"$f")
    alive_with "$p" gogeta-gpu-slot && others+=" $(basename "$f" .pid)(pid $p)"
  done
  # 2) training processes started some other way (run_queue.sh, by hand)
  for p in $(pgrep -f -- '-m gogeta[.]train' 2>/dev/null) $(pgrep -f -- 'tools/run_queue[.]sh' 2>/dev/null); do
    [[ "$p" == "$$" ]] && continue
    [[ "$(cmdline "$p")" == *gogeta-gpu-slot* ]] && continue                  # listed above
    pg=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    alive_with "$pg" gogeta-gpu-slot && continue                              # a launcher run's child
    others+=" pid $p ($(cmdline "$p" | cut -c1-80))"
  done
  if [[ -n "$others" ]]; then fail "no other training run" "active:$others"
  else pass "no other training run" "none active"; fi

  # 3) the single GPU slot
  if [[ -f "$GPU_LOCK" ]]; then
    holder=$(lock_field pid)
    if alive_with "$holder" gogeta-gpu-slot; then
      fail "GPU slot" "held by $(lock_field experiment) seed $(lock_field seed) (pid $holder, since $(lock_field started)); log $(lock_field log)"
    elif (( DRY )); then
      warn "GPU slot" "stale lock $GPU_LOCK (pid ${holder:-?} is dead); a real launch removes it"
    else
      rm -f "$GPU_LOCK"; pass "GPU slot" "free (removed stale lock of dead pid ${holder:-?})"
    fi
  else
    pass "GPU slot" "free ($GPU_LOCK absent)"
  fi

  # 4) GPU jobs that bypass the slot (another agent's check, a notebook...): still a GPU job
  if [[ -n "$NVSMI" ]]; then
    apps=$("$NVSMI" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | tr -d '\r')
    if [[ -n "$apps" ]]; then
      a=""
      while IFS=, read -r apid _; do
        apid=$(tr -dc '0-9' <<<"$apid"); [[ -n "$apid" ]] || continue
        a+=" pid $apid ($(cmdline "$apid" | cut -c1-70))"
      done <<<"$apps"
      fail "GPU idle" "compute process(es) on the GPU outside the launcher:${a:- (unreadable)}. One GPU job at a time (AGENTS.md 5)"
    else
      pass "GPU idle" "no compute processes (nvidia-smi)"
    fi
  else
    warn "GPU idle" "nvidia-smi not found; cannot see GPU processes"
  fi
}

check_temperature() {
  local t; t=$(gpu_temp)
  if ! is_num "$t"; then
    if (( DRY )); then warn "GPU temperature" "cannot read it (nvidia-smi: ${t:-no output})"
    else fail "GPU temperature" "cannot read it (nvidia-smi: ${t:-no output}); refusing rather than risk overheating"; fi
    return
  fi
  if ! ge "$t" "$TEMP_REFUSE"; then
    pass "GPU temperature" "${t} C (< ${TEMP_REFUSE} C)"; return
  fi
  if (( WAITCOOL && !DRY )); then
    echo "GPU is at ${t} C (>= ${TEMP_REFUSE} C). --wait-cool: polling every ${COOL_POLL}s until < ${TEMP_RESUME} C ..."
    while :; do
      sleep "$COOL_POLL"; t=$(gpu_temp)
      echo "  $(date +%H:%M:%S)  ${t:-?} C"
      is_num "$t" && ! ge "$t" "$TEMP_RESUME" && break
    done
    pass "GPU temperature" "${t} C after cooling (< ${TEMP_RESUME} C)"
  elif (( WAITCOOL )); then
    fail "GPU temperature" "${t} C (>= ${TEMP_REFUSE} C); a real launch with --wait-cool would wait until < ${TEMP_RESUME} C"
  else
    fail "GPU temperature" "${t} C (>= ${TEMP_REFUSE} C). Let it cool below ${TEMP_RESUME} C, or re-run with --wait-cool"
  fi
}

# ----------------------------------------------------------------------------- preflight
# venv + imports
if [[ -x "$PY" ]]; then
  pass "venv python" "$PY"
  if out=$(cd "$REPO" && "$PY" -c "import torch, gogeta; print(torch.__version__)" 2>&1); then
    pass "import torch, gogeta" "torch $(tail -n1 <<<"$out")"
  else
    fail "import torch, gogeta" "$(tail -n1 <<<"$out")"
  fi
  if (( DRY )); then
    skip "CUDA available" "dry run (not touching the GPU)"
  elif out=$("$PY" -c "import torch, sys; ok = torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'no CUDA device'); sys.exit(0 if ok else 1)" 2>&1); then
    pass "CUDA available" "$(tail -n1 <<<"$out")"
  else
    fail "CUDA available" "$(tail -n1 <<<"$out")"
  fi
else
  fail "venv python" "not found: $PY (AGENTS.md 6, 'Training environment setup')"
fi

# config <-> registry agreement
if [[ -n "$CFG_EID" && "${CFG_EID^^}" != "$EID" ]]; then
  fail "config experiment id" "$CONFIG says \"experiment\": \"$CFG_EID\", not $EID (runs would be logged under $CFG_EID)"
fi

# dataset
if [[ -z "$DATASET" ]]; then
  fail "dataset" "$CONFIG has no train.dataset"
elif [[ -f "$DATA/datasets/$DATASET/meta.json" ]]; then
  pass "dataset" "$DATA/datasets/$DATASET/meta.json"
else
  fail "dataset" "missing $DATA/datasets/$DATASET/meta.json (build it with tools/prepare_data.py)"
fi

# prediction first (AGENTS.md 0 and 6)
if [[ "$PRED_LEN" -gt 0 ]]; then
  pass "prediction written" "$PRED_LEN chars: \"$PRED_PREVIEW\""
else
  fail "prediction written" "$EID has an empty 'prediction' in the registry. No prediction, no run (AGENTS.md 6)"
fi

# clean code paths (gogeta.train --require-clean enforces the same thing)
if dirty=$(git -C "$REPO" status --porcelain -- "${CODE_PATHS[@]}" 2>&1); then
  if [[ -z "$dirty" ]]; then
    pass "code paths clean" "${CODE_PATHS[*]} match HEAD ($(git -C "$REPO" describe --tags --always 2>/dev/null))"
  else
    n=$(wc -l <<<"$dirty")
    fail "code paths clean" "$n uncommitted change(s), --require-clean will refuse: $(head -n 4 <<<"$dirty" | sed 's/^ *//' | tr '\n' ';' | sed 's/;$//')$( (( n > 4 )) && echo ' ...')"
  fi
else
  fail "code paths clean" "git status failed: $dirty"
fi

# registry status (informational)
case "$REG_STATUS" in
  done)    warn "registry status" "$EID is already 'done'; this is an extra seed / re-run" ;;
  running) warn "registry status" "$EID is marked 'running' in the registry" ;;
  *)       pass "registry status" "$REG_STATUS (set it to 'running' in experiments.json when the run starts, AGENTS.md 5)" ;;
esac
[[ -n "$MAXTOK" ]] && warn "max tokens" "override $MAXTOK (config says $CFG_MAXTOK): not the registered run"

# dashboard (warning only: logging works without it)
if curl -s -m 3 "$DASH_URL/" 2>/dev/null | grep -q "Trackio Dashboard"; then
  pass "Trackio dashboard" "$DASH_URL"
else
  warn "Trackio dashboard" "not answering on $DASH_URL (runs still log). Start it: bash tools/launch_tracker.sh start"
fi
TRACKIO_EFFECTIVE=${TRACKIO_DIR:-$DATA/trackio}
[[ "$TRACKIO_EFFECTIVE" != "$DATA/trackio" ]] && warn "TRACKIO_DIR" "set to $TRACKIO_DIR, but the dashboard reads $DATA/trackio"

check_concurrency
check_temperature

ENV_DESC="GOGETA_TRACKER=trackio PYTHONUNBUFFERED=1 TRACKIO_DIR=$TRACKIO_EFFECTIVE GOGETA_DATA=$DATA"

print_plan() {
  local mode=$1
  echo "== Gogeta-0 launch: $EID seed $SEED  [$mode]"
  echo "experiment : $EID  \"$TITLE\"  (registry status: $REG_STATUS)"
  echo "config     : $CONFIG  (dataset $DATASET, max_tokens ${MAXTOK:-$CFG_MAXTOK})"
  echo "preflight  :"
  printf '%s\n' "${CHECKS[@]}"
  echo "cwd        : $REPO"
  echo "env        : $ENV_DESC"
  printf 'command    :'; printf ' %q' "${CMD[@]}"; echo
  echo "mode       : $( (( FG )) && echo 'foreground (attached; output also teed to the log)' || echo 'detached (setsid nohup)')"
  echo "log        : $LOG"
  echo "pid file   : $PIDFILE"
  echo "GPU slot   : $GPU_LOCK (held by the run's wrapper, released when training exits)"
  echo "follow     : tail -f $LOG"
  echo "dashboard  : $DASH_URL"
}

# ----------------------------------------------------------------------------- dry run
if (( DRY )); then
  print_plan "DRY RUN"
  if (( NFAIL )); then
    echo "result     : DRY RUN, nothing executed. A real launch would REFUSE: $NFAIL failed check(s), $NWARN warning(s)."
    exit 3
  fi
  echo "result     : DRY RUN, nothing executed. A real launch would START ($NWARN warning(s))."
  exit 0
fi

if (( NFAIL )); then
  print_plan "REFUSED"
  echo "result     : not launched: $NFAIL failed preflight check(s)." >&2
  exit 3
fi

# ----------------------------------------------------------------------------- launch
mkdir -p "$RUN_DIR" "$LOG_DIR" || die "cannot create $RUN_DIR / $LOG_DIR"
exec 9>"$RUN_DIR/launch.lock"
flock -w 60 9 || die "another launch is in progress (could not lock $RUN_DIR/launch.lock in 60s)"
# re-check under the lock: someone may have started a run while we were checking or cooling down
CHECKS=() NFAIL=0
check_concurrency
if (( NFAIL )); then
  printf '%s\n' "${CHECKS[@]}" >&2
  die "not launched: the GPU became busy during preflight" 3
fi

{
  echo "=== $(date -Is) launch_experiment: $EID seed $SEED ($RUNNAME)"
  echo "=== git: $(git -C "$REPO" describe --tags --always --dirty 2>/dev/null)  config: $CONFIG"
  printf '=== cmd:'; printf ' %q' "${CMD[@]}"; echo
  echo "=== env: $ENV_DESC"
} >>"$LOG"
{
  echo "experiment=$EID"; echo "seed=$SEED"; echo "config=$CONFIG"; echo "log=$LOG"
  echo "started=$(date -Is)"; printf 'cmd='; printf ' %q' "${CMD[@]}"; echo
} >"$INFOFILE"

# The wrapper ($0 = gogeta-gpu-slot, visible in ps) owns the GPU slot for its lifetime:
# it writes its PID file and gpu.lock, runs training, records the exit code, and removes
# gpu.lock on any exit (normal, error, SIGTERM/SIGINT). A SIGKILLed wrapper leaves a stale
# lock, which the next preflight removes because the PID is dead.
WRAPPER='
set -u
echo $$ >"$GL_PIDFILE"
printf "pid=%s\nexperiment=%s\nseed=%s\nlog=%s\nstarted=%s\n" "$$" "$GL_EID" "$GL_SEED" "$GL_LOG" "$(date -Is)" >"$GL_LOCK.tmp.$$"
mv -f "$GL_LOCK.tmp.$$" "$GL_LOCK"
{ flock -u 9; exec 9>&-; } 2>/dev/null || true   # slot recorded: let the next launch check it
release() { if [ -f "$GL_LOCK" ] && grep -qx "pid=$$" "$GL_LOCK"; then rm -f "$GL_LOCK"; fi; }
trap release EXIT
trap "exit 143" TERM
trap "exit 130" INT
"$@"
rc=$?
echo "$rc" >"$GL_EXITFILE"
echo "=== $(date -Is) launcher: training exited with code $rc; GPU slot released"
exit $rc
'
LAUNCH_ENV=(env GOGETA_TRACKER=trackio PYTHONUNBUFFERED=1 TRACKIO_DIR="$TRACKIO_EFFECTIVE" GOGETA_DATA="$DATA"
            GL_PIDFILE="$PIDFILE" GL_LOCK="$GPU_LOCK" GL_EXITFILE="$EXITFILE"
            GL_EID="$EID" GL_SEED="$SEED" GL_LOG="$LOG")
cd "$REPO" || die "cannot cd to $REPO"

if (( FG )); then
  echo "Starting $EID seed $SEED in the foreground (log: $LOG). Ctrl-C stops it and frees the GPU slot."
  # The wrapper unlocks fd 9 once it has written gpu.lock (tee gets no copy of fd 9).
  "${LAUNCH_ENV[@]}" bash -c "$WRAPPER" gogeta-gpu-slot "${CMD[@]}" 2>&1 </dev/null | tee -a "$LOG" 9>&-
  rc=${PIPESTATUS[0]}
  exec 9>&-
  echo "Run finished with exit code $rc. Log: $LOG"
  exit "$rc"
fi

setsid nohup "${LAUNCH_ENV[@]}" bash -c "$WRAPPER" gogeta-gpu-slot "${CMD[@]}" >>"$LOG" 2>&1 </dev/null &
for _ in $(seq 1 50); do [[ -s "$PIDFILE" ]] && break; sleep 0.1; done
PID=$(tr -dc '0-9' <"$PIDFILE" 2>/dev/null)
flock -u 9; exec 9>&-
[[ -n "$PID" ]] || die "the run did not write its PID file ($PIDFILE); see $LOG" 4

# catch immediate failures (bad config, dirty code, CUDA errors) before reporting success
for _ in $(seq 1 8); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
if ! kill -0 "$PID" 2>/dev/null; then
  echo "The run exited right after starting (exit code $(cat "$EXITFILE" 2>/dev/null || echo '?')). Last log lines:" >&2
  tail -n 25 "$LOG" >&2
  exit 4
fi

echo "Launched $EID seed $SEED (detached)."
echo "  pid       : $PID   (wrapper; training runs in its process group)"
echo "  log       : $LOG"
echo "  follow    : tail -f $LOG"
echo "  pid file  : $PIDFILE"
echo "  GPU slot  : held until training exits ($GPU_LOCK)"
echo "  dashboard : $DASH_URL"
echo "  status    : bash experiments/launch/status.sh     stop: bash experiments/launch/stop.sh $EID"
echo "  reminder  : set $EID to \"running\" in experiments/experiments.json (AGENTS.md 5)"
