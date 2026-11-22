#!/usr/bin/env bash
# E1: Linear-time mixers
# Config: configs/e1_hybrid.json   (looked up in experiments/experiments.json by the launcher)
#
#   bash experiments/launch/E1.sh [--seed N] [--dry-run] [--foreground] [--no-compile] [--max-tokens N] [--wait-cool]
#
# All checks (prediction written, clean code, dataset, GPU slot, GPU temperature...) live in
# tools/launch_experiment.sh, so every experiment starts the same way and logs to Trackio.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E1 "$@"
