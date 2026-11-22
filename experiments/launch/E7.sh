#!/usr/bin/env bash
# E7: Loops vs chain-of-thought
# NOT DESIGNED YET: no config and no prediction in experiments/experiments.json.
#
# To make this launchable: add configs/e7_<name>.json, set "config" and write the
# "prediction" for E7 in the registry (commit both before the first run, AGENTS.md 6), then
# replace the two lines below with the standard body used by E3.sh:
#   exec bash "$(dirname "${BASH_SOURCE[0]}")/../../tools/launch_experiment.sh" E7 "$@"
echo "E7 (Loops vs chain-of-thought) is not designed yet: add a config and a prediction first (experiments/experiments.json, AGENTS.md 6). Nothing was launched." >&2
exit 3
