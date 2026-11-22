#!/usr/bin/env bash
# Create the dedicated cockpit venv (WSL). Never touches the gogeta / gogeta-rl / gogeta-eval venvs.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${COCKPIT_VENV:-$HOME/.venvs/gogeta-cockpit}
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet -r "$REPO/requirements-cockpit.txt"
"$VENV/bin/pip" list 2>/dev/null | grep -iE '^(mcp|textual) '
