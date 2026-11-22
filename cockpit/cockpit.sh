#!/usr/bin/env bash
# Gogeta-0 cockpit launcher (run inside WSL).
#   bash cockpit/cockpit.sh            TUI
#   bash cockpit/cockpit.sh --mcp      MCP server on stdio (what .mcp.json runs)
#   bash cockpit/cockpit.sh --setup    create/update ~/.venvs/gogeta-cockpit from requirements-cockpit.txt
#   bash cockpit/cockpit.sh --test     unit tests (stdlib) + MCP probe + headless TUI check
#   bash cockpit/cockpit.sh --ingest   backfill the SQLite log store and print table counts
#   bash cockpit/cockpit.sh --watchdog start the watchdog daemon (idempotent) and print status; log: ~/fusiondance-data/cockpit/incidents.log
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
esac
[[ -x "$PY" ]] || { echo "cockpit: venv missing, creating it (bash cockpit/cockpit.sh --setup)" >&2; bash "$HERE/setup_venv.sh" >&2; }
case "${1:-}" in
  --mcp) exec "$PY" "$HERE/mcp_server.py" ;;
  *)     exec "$PY" "$HERE/tui.py" ;;
esac
