#!/usr/bin/env bash
# E11: Negative-eigenvalue Gated DeltaNet for language modelling
# Config: configs/e11_negeig.json   (looked up in experiments/experiments.json by the launcher)
#
#   bash experiments/launch/E11.sh [--seed N] [--dry-run] [--foreground] [--no-compile] [--max-tokens N] [--wait-cool]
#
# All checks (prediction written, clean code, dataset, GPU slot, GPU temperature...) live in
# tools/launch_experiment.sh, so every experiment starts the same way and logs to Trackio.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E11 "$@"
