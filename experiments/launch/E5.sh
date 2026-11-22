#!/usr/bin/env bash
# E5: Adaptive loops per token
# NOT DESIGNED YET: no config and no prediction in experiments/experiments.json.
#
# To make this launchable: add configs/e5_<name>.json, set "config" and write the
# "prediction" for E5 in the registry (commit both before the first run, AGENTS.md 6), then
# replace the two lines below with the standard body used by E3.sh:
#   exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E5 "$@"
echo "E5 (Adaptive loops per token) is not designed yet: add a config and a prediction first (experiments/experiments.json, AGENTS.md 6). Nothing was launched." >&2
exit 3
