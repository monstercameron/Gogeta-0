#!/usr/bin/env bash
# Gogeta-0 cockpit launcher (run inside WSL).
#   bash cockpit/cockpit.sh            TUI
#   bash cockpit/cockpit.sh --mcp      MCP server on stdio (what .mcp.json runs)
#   bash cockpit/cockpit.sh --setup    create/update ~/.venvs/gogeta-cockpit from requirements-cockpit.txt
#   bash cockpit/cockpit.sh --test     unit tests (stdlib) + MCP probe + headless TUI check
#   bash cockpit/cockpit.sh --ingest   backfill the SQLite log store and print table counts
#   bash cockpit/cockpit.sh --watchdog start the watchdog daemon (idempotent) and print status; log: ~/fusiondance-data/cockpit/incidents.log
#   HUMAN ONLY (interactive terminal, typed confirmation; refused without a TTY or from an agent shell):
#   bash cockpit/cockpit.sh --approve-spend --max-usd 30 --max-usd-per-hour 1.00 --max-instances 1 --gpus "RTX 3090" --hours 12 --purpose "..."
#   bash cockpit/cockpit.sh --revoke-spend APR-...      revoke a spend approval (the watchdog then destroys its instances)
#   bash cockpit/cockpit.sh --revoke-ssh-key            delete the cockpit's SSH key (~/.ssh/gogeta_vast.pub) from the Vast account
#   bash cockpit/cockpit.sh --cloud approvals|offers|instances|balance|brief   read-only cloud queries
#   bash cockpit/cockpit.sh --reconnect        after a session/WSL restart: start the watchdog if it is down, then print
#                                              hold state, GPU, campaigns, open incidents and every live cloud instance + job
#   bash cockpit/cockpit.sh --hold "reason"    operator hold: no campaign resume (manual or watchdog auto-resume) until released
#   bash cockpit/cockpit.sh --release-hold     HUMAN ONLY (TTY): remove the hold
#   The TUI (no flag) also starts the watchdog if it is down, so opening the cockpit is enough to reconnect.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
VENV=${COCKPIT_VENV:-$HOME/.venvs/gogeta-cockpit}
PY=$VENV/bin/python
export PYTHONDONTWRITEBYTECODE=1
case "${1:-}" in
  --setup) exec bash "$HERE/setup_venv.sh" ;;
  --test)  cd "$REPO"; python3 -m unittest cockpit.test_core cockpit.test_watchdog -v; "$PY" "$HERE/mcp_probe.py"; "$PY" "$HERE/tui_check.py"; exit ;;
  --watchdog) python3 "$HERE/watchdog.py" --start; echo "tail -F $HOME/fusiondance-data/cockpit/incidents.log"; exit ;;
  --ingest) cd "$REPO"; python3 -m cockpit.logdb ingest; python3 -m cockpit.logdb stats; exit ;;
  --approve-spend) shift; cd "$REPO"; exec python3 "$HERE/cloud.py" approve "$@" ;;
  --revoke-spend) shift; cd "$REPO"; exec python3 "$HERE/cloud.py" revoke "$@" ;;
  --revoke-ssh-key) shift; cd "$REPO"; exec python3 "$HERE/cloud.py" revoke-ssh-key "$@" ;;
  --cloud) shift; cd "$REPO"; exec python3 "$HERE/cloud.py" "$@" ;;
  --hold) shift; mkdir -p "$HOME/fusiondance-data/cockpit"
          echo "HOLD $(date -Is): ${*:-operator hold}. No campaign resume until a human runs: bash cockpit/cockpit.sh --release-hold" \
            > "$HOME/fusiondance-data/cockpit/hold.flag"; cat "$HOME/fusiondance-data/cockpit/hold.flag"; exit ;;
  --release-hold)
          [[ -t 0 && -t 1 ]] || { echo "release-hold is human-only: run it in an interactive terminal" >&2; exit 2; }
          F="$HOME/fusiondance-data/cockpit/hold.flag"; [[ -f "$F" ]] || { echo "no hold set"; exit 0; }
          cat "$F"; read -r -p "Type RELEASE to allow campaign resumes again: " a
          [[ "$a" == "RELEASE" ]] && rm -f "$F" && echo "hold released" || echo "kept"; exit ;;
  --reconnect)
          set +e +o pipefail   # a report: one failing section must not hide the others
          cd "$REPO"; echo "== watchdog"; python3 "$HERE/watchdog.py" --start | cut -c1-200
          echo "== hold"; cat "$HOME/fusiondance-data/cockpit/hold.flag" 2>/dev/null || echo "none"
          echo "== local GPU, runs, CPU jobs"; bash "$REPO/experiments/launch/status.sh" 2>&1 | sed -n '1,6p;/CPU jobs/,/^$/p'
          echo "== campaigns"; bash "$REPO/tools/launch_campaign.sh" --status 2>&1 | tail -n 6
          echo "== open incidents"; python3 "$HERE/watchdog.py" --status | cut -c1-200
          echo "== cloud (live instances, jobs, spend; ip/port never to be pasted or committed)"
          python3 "$HERE/cloud.py" brief 2>&1 | head -n 20; exit 0 ;;
esac
[[ -x "$PY" ]] || { echo "cockpit: venv missing, creating it (bash cockpit/cockpit.sh --setup)" >&2; bash "$HERE/setup_venv.sh" >&2; }
case "${1:-}" in
  --mcp) exec "$PY" "$HERE/mcp_server.py" ;;
  *)     python3 "$HERE/watchdog.py" --start >/dev/null 2>&1 || echo "cockpit: watchdog start failed (bash cockpit/cockpit.sh --watchdog)" >&2
         exec "$PY" "$HERE/tui.py" ;;
esac
